import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import neurite as ne

from . import layers
from . import utils


def labels_to_image_model(
    in_shape,
    in_label_list,
    out_label_list=None,
    out_shape=None,
    num_chan=1,
    mean_min=None,
    mean_max=None,
    std_min=None,
    std_max=None,
    warp_shape_factor=[16],
    warp_std_dev=0.5,
    warp_modulate=True,
    bias_shape_factor=40,
    bias_std_dev=0.3,
    bias_modulate=True,
    blur_background=True,
    blur_std_dev=1,
    blur_modulate=True,
    normalize=True,
    gamma_std_dev=0.25,
    dc_offset=0,
    one_hot=True,
    return_vel=False,
    return_def=False,
    id=0,
):
    '''Build model that augments label maps and synthesizes images from them.

    Arguments:
        in_shape: List of the spatial dimensions of the input label maps.
        in_label_list: List of all possible input labels.
        out_label_list (optional): List of labels in the output label maps. If
            a dictionary is passed, it will be used to convert labels, e.g. to
            GM, WM and CSF. All labels not included will be converted to
            background with value 0. If 0 is among the output labels, it will be
            one-hot encoded. Defaults to the input labels.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit.
            Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized.
            Defaults to 1.
        mean_min (optional): List of lower bounds on the means drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            25 for all other labels.
        mean_max (optional): List of upper bounds on the means drawn to generate
            the intensities for each label. Defaults to 225 for each label.
        std_min (optional): List of lower bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            5 for all other labels.
        std_max (optional): List of upper bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 25 for each label.
            25 for all other labels.
        warp_shape_factor (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the SVF is drawn.
            Defaults to 16.
        warp_std_dev (float, optional): Upper bound on the SDs used when drawing
            the SVF. Defaults to 0.5.
        warp_modulate (bool, optional): Whether to draw the SVF with random SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        bias_shape_factor (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the bias field is
            drawn. Defaults to 40.
        bias_std_dev (float, optional): Upper bound on the SDs used when drawing
            the bias field. Defaults to 0.3.
        bias_modulate (bool, optional): Whether to draw the bias field with
            random SDs. If disabled, each batch will use the maximum SD.
            Defaults to True.
        blur_std_dev (float, optional): Upper bound on the SD of the kernel used
            for Gaussian image blurring. Defaults to 1.
        blur_modulate (bool, optional): Whether to draw random blurring SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        blur_background (bool, optional): Whether the background is blurred as
            all other labels. Defaults to True.
        normalize (bool, optional): Whether the image is min-max normalized.
            Defaults to True.
        gamma_std_dev (float, optional): SD of random global intensity
            exponentiation, i.e. gamma augmentation. Defaults to 0.25.
        dc_offset (float, optional): Upper bound on global DC offset drawn and
            added to the image after normalization. Defaults to 0.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to True.
        return_vel (bool, optional): Whether to append the half-resolution SVF
            to the model outputs. Defaults to False.
        return_def (bool, optional): Whether to append the combined displacement
            field to the model outputs. Defaults to False.
        id (int, optional): Model identifier used to create unique layer names
            for including this model multiple times. Defaults to 0.

    Returns:
        tf.Keras.Model: Label-deformation and image-synthesis model.
    '''
    if out_shape is None:
        out_shape = in_shape
    in_shape, out_shape = map(np.asarray, (in_shape, out_shape))
    num_dim = len(in_shape)

    # Transform labels into [0, 1, ..., N-1].
    in_label_list = np.unique(in_label_list).astype('int32')
    num_in_labels = len(in_label_list)
    new_in_label_list = np.arange(num_in_labels)
    in_lut = np.zeros(np.max(in_label_list) + 1, dtype='float32')
    for i, lab in enumerate(in_label_list):
        in_lut[lab] = i
    labels_input = KL.Input(shape=(*in_shape, 1), name=f'labels_input_{id}')
    labels = KL.Lambda(lambda x: tf.gather(in_lut, tf.cast(x, dtype='int32')))(labels_input)

    vel_shape = (*out_shape // 2, num_dim)
    if warp_std_dev > 0:
        # Velocity field.
        vel_scale = np.asarray(warp_shape_factor) / 2
        vel_draw = lambda x: tf_perlin(vel_shape, scales=vel_scale, max_std=warp_std_dev, modulate=warp_modulate)
        vel_field = KL.Lambda(lambda x: tf.map_fn(vel_draw, x, dtype='float32'), name=f'vel_{id}')(labels) # One per batch.
        # Deformation field.
        def_field = layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        def_field = ne.layers.RescaleValues(2)(def_field)
        def_field = ne.layers.Resize(2, interp_method='linear', name=f'def_{id}')(def_field)
    else:
        draw_zeros = lambda x, d: tf.zeros((tf.shape(x)[0], *d), dtype='float32')
        vel_field = KL.Lambda(lambda x: draw_zeros(x, vel_shape), name=f'vel_{id}')(labels)
        def_field = KL.Lambda(lambda x: draw_zeros(x, (*out_shape, num_dim)), name=f'def_{id}')(labels)

    # Resampling.
    labels = layers.SpatialTransformer(interp_method='nearest', fill_value=0, name=f'trans_{id}')([labels, def_field])
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)

    # Intensity means and standard deviations.
    if mean_min is None:
        mean_min = [0] + [25] * (num_in_labels - 1)
    if mean_max is None:
        mean_max = [225] * num_in_labels
    if std_min is None:
        std_min = [0] + [5] * (num_in_labels - 1)
    if std_max is None:
        std_max = [25] * num_in_labels
    int_range = [mean_min, mean_max, std_min, std_max]
    for i, x in enumerate(int_range):
        x = np.asarray(x)
        int_range[i] = x[...,None] if np.ndim(x) == 1 else x
    m0, m1, s0, s1 = int_range
    mean_draw = lambda x: tf.random.uniform((tf.shape(x)[0], num_in_labels, num_chan), minval=m0, maxval=m1)
    std_draw = lambda x: tf.random.uniform((tf.shape(x)[0], num_in_labels, num_chan), minval=s0, maxval=s1)
    means = KL.Lambda(mean_draw)(labels)
    stds = KL.Lambda(std_draw)(labels)

    # Synthetic image.
    image = KL.Lambda(lambda x: tf.random.normal(tf.shape(x)),  name=f'sample_normal_{id}')(labels)
    im_cat = lambda x: tf.concat([x + num_in_labels*i for i in range(num_chan)], axis=-1)
    im_ind = KL.Lambda(im_cat, name=f'ind_{id}')(labels)
    im_take = lambda x: tf.gather(tf.reshape(x[0], shape=(-1,)), x[1])
    gather = KL.Lambda(lambda x: tf.map_fn(im_take, x, dtype='float32'))
    means = gather([means, im_ind])
    stds = gather([stds, im_ind])
    image = KL.Multiply(name=f'mul_std_dev_{id}')([image, stds])
    image = KL.Add(name=f'add_means_{id}')([image, means])

    # Blur.
    if blur_modulate:
        blur_draw = lambda _: tf_draw_kernel([blur_std_dev] * num_dim)
    else:
        blur_draw = lambda _: [ne.utils.gaussian_kernel(blur_std_dev)] * num_dim
    kernels = KL.Lambda(lambda x: tf.map_fn(blur_draw, x, dtype=['float32'] * num_dim))(image)
    blur_apply = lambda x: ne.utils.separable_conv(x[0], x[1])
    image = KL.Lambda(lambda x: tf.map_fn(blur_apply, x, dtype='float32'), name=f'apply_blur_{id}')([image, kernels])

    # Background voodoo.
    mask = KL.Lambda(lambda x: tf.cast(tf.greater(x, 0), 'float32'))(labels)
    channels = KL.Lambda(lambda x: tf.split(x, num_or_size_splits=num_chan, axis=-1))(image) if num_chan > 1 else [image]
    blurred_mask = None
    out = [None] * num_chan
    for i in range(num_chan):
        if blur_background:
            rand_flip = KL.Lambda(lambda x: tf.greater(tf.random.uniform((1,), 0, 1), 0.8), name=f'bool_{i}_{id}')([])
            out[i] = KL.Lambda(lambda x: K.switch(x[0], x[1] * x[2], x[1]))([rand_flip, channels[i], mask])
        else:
            if blurred_mask is None:
                blurred_mask = KL.Lambda(lambda x: tf.map_fn(blur_apply, x, dtype='float32'))([mask, kernels])
            out[i] = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()), name=f'masked_blurring_{i}_{id}')([channels[i], blurred_mask])
            bg_mean = KL.Lambda(lambda x: tf.random.uniform((1,), 0, 10), name=f'bg_mean_{i}_{id}')([])
            bg_std = KL.Lambda(lambda x: tf.random.uniform((1,), 0, 5), name=f'bg_std_{i}_{id}')([])
            rand_flip = KL.Lambda(lambda x: tf.greater(tf.random.uniform((1,), 0, 1), 0.5), name=f'boolx_{i}_{id}')([])
            bg_mean = KL.Lambda(lambda x: K.switch(x[0], tf.zeros_like(x[1]), x[1]), name=f'switch_backgd_mean_{i}_{id}')([rand_flip, bg_mean])
            bg_std = KL.Lambda(lambda x: K.switch(x[0], tf.zeros_like(x[1]), x[1]), name=f'switch_backgd_std_{i}_{id}')([rand_flip, bg_std])
            background = KL.Lambda(lambda x: tf.random.normal(tf.shape(x[0]), mean=x[1], stddev=x[2]),
                                   name=f'gaussian_bg_{i}_{id}')([channels[i], bg_mean, bg_std])
            out[i] = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]),
                              name=f'mask_blurred_image_{i}_{id}')([out[i], mask, background])
    image = KL.Concatenate(axis=-1)(out) if num_chan > 1 else out[0]

    # Bias field.
    bias_scale = bias_shape_factor
    bias_shape = (*out_shape, 1)
    bias_draw = lambda x: tf_perlin(bias_shape, scales=bias_scale, max_std=bias_std_dev, modulate=bias_modulate)
    bias_field = KL.Lambda(lambda x: tf.map_fn(bias_draw, x, dtype='float32'))(labels) # One per batch.
    bias_field = KL.Lambda(lambda x: tf.exp(x), name=f'bias_{id}')(bias_field)
    image = KL.multiply([bias_field, image], name=f'apply_bias_{id}')

    # Intensity manipulations.
    image = KL.Lambda(lambda x: tf.clip_by_value(x, 0, 255), name=f'clip_{id}')(image)
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(tf_normalize, x))(image)
    if gamma_std_dev > 0:
        gamma_apply = lambda x: tf.pow(x, tf.exp(tf.random.normal((1,), stddev=gamma_std_dev)))
        image = KL.Lambda(lambda x: tf.map_fn(gamma_apply, x, dtype='float32'), name=f'gamma_{id}')(image)
    if dc_offset > 0:
        dc_apply = lambda x: tf.add(x, tf.random.uniform((1,), maxval=dc_offset))
        image = KL.Lambda(lambda x: tf.map_fn(dc_apply, x, dtype='float32'), name=f'dc_offset_{id}')(image)

    # Lookup table for converting the index labels back to the original values,
    # setting unwanted labels to background. If the output labels are provided
    # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF.
    if out_label_list is None:
        out_label_list = in_label_list
    if isinstance(out_label_list, (tuple, list, np.ndarray)):
        out_label_list = {lab: lab for lab in out_label_list}
    out_lut = np.zeros(num_in_labels, dtype='int32')
    for i, lab in enumerate(in_label_list):
        if lab in out_label_list:
            out_lut[i] = out_label_list[lab]

    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps.
    if one_hot:
        hot_label_list = np.unique(list(out_label_list.values())) # Sorted.
        hot_lut = np.full(hot_label_list[-1] + 1, fill_value=-1, dtype='int32')
        for i, lab in enumerate(hot_label_list):
            hot_lut[lab] = i
        out_lut = hot_lut[out_lut]

    # Convert indices to output labels only once.
    out_conv = lambda x: tf.gather(out_lut, x)
    labels = KL.Lambda(out_conv, name=f'labels_back_{id}')(labels)
    if one_hot:
        depth = len(hot_label_list)
        labels = KL.Lambda(lambda x: tf.one_hot(x[...,0], depth), name=f'one_hot_{id}')(labels)

    outputs = [image, labels]
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)

    return tf.keras.Model(inputs=labels_input, outputs=outputs)


def tf_normalize(x):
    '''Min-max normalize tensor.'''
    m = tf.reduce_min(x)
    M = tf.reduce_max(x)
    return tf.compat.v1.div_no_nan(x - m, M - m)


def tf_draw_kernel(max_sigma, width=None):
    '''Draw list of 1D Gaussian kernels.'''
    if not isinstance(max_sigma, (list, tuple)):
        max_sigma = [max_sigma]
    if width is None:
        width = 2 * np.round(3 * max(max_sigma)) + 1
    center = (width - 1) / 2
    grid = np.arange(width) - center
    grid = -0.5 * (grid ** 2)
    grid = tf.constant(grid, dtype='float32')
    sigma = [tf.random.uniform((1,), minval=1e-6, maxval=x) for x in max_sigma]
    kernel = [tf.exp(grid / x ** 2) for x in sigma]
    return [x / tf.reduce_sum(x) for x in kernel]


def tf_perlin(out_shape, scales, max_std=1, modulate=True):
    '''Generate Perlin noise by drawing from Gaussian distributions at different
    resolutions, upsampling and summing. Scale 2 means half resolution. Expects
    features as last dimension.'''
    out_shape = np.asarray(out_shape, dtype='int32')
    if np.isscalar(scales):
        scales = [scales]
    out = tf.zeros(out_shape)
    for scale in scales:
        sample_shape = np.ceil(out_shape[:-1] / scale).astype(int)
        sample_shape = (*sample_shape, out_shape[-1])
        std = tf.random.uniform((1,), maxval=max_std) if modulate else max_std
        gauss = tf.random.normal(sample_shape, stddev=std)
        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else ne.utils.resize(gauss, zoom[:-1])
    return out
