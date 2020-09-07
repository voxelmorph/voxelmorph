import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
import numpy.random as npr
# from sklearn import preprocessing

from neurite import layers as nrn_layers
from .utils import add_axis, gauss_kernel, format_target_res, get_nonlin_field_shape, get_bias_field_shape, get_shapes
from .. import layers

def labels_to_image_model(labels_shape,
                          crop_shape,
                          generation_label_list,
                          segmentation_label_list,
                          n_channels=1,
                          labels_res=(1,1,1),
                          target_res=None,
                          padding_margin=None,
                          apply_affine_trans=False,
                          apply_nonlin_trans=True,
                          nonlin_shape_factor=0.0625,
                          apply_bias_field=True,
                          bias_shape_factor=0.025,
                          blur_background=True,
                          normalise=True,
                          out_div_32=False,
                          convert_back=False,
                          id=0,  # For different layer names if several models.
                          rand_blur=True):
    """
        This function builds a keras/tensorflow model to generate brain images from supplied labels.
        It returns the model as well as the shape ouf the output images without batch and channel dimensions
        (height*width*depth).
        The model takes as inputs:
            -a label image
            -a vector containing the means of the Gaussian distributions to sample for each label,
            -a similar vector for the associated standard deviations.
            -if apply_affine_deformation=True: a (n_dims+1)x(n_dims+1) affine matrix
            -if apply_non_linear_deformation=True: a small non linear field of size batch*x*y*z*n_dims that will be
             resampled to labels size
            -if apply_bias_field=True: a small bias field of size batch*x*y*z*1 that will be resampled to labels size
        The model returns:
            -the generated image
            -the corresponding label map
    :param labels_shape: should be a list or tensor with image dimension plus channel size at the end
    :param n_channels: number of channels to be synthetised
    :param labels_res: list of dimension resolutions of model's inputs
    :param target_res: list of dimension resolutions of model's outputs
    :param crop_shape: list, shape of model's outputs
    :param generation_label_list: list of all the labels in the dataset (internally converted to [0...N-1] and converted
    back to original values at the end of model)
    :param segmentation_label_list: list of all the labels in the output labels (internally converted to [0...N-1] and
    converted back to original values at the end of model)
    :param padding_margin: margin by which the input labels will be 0-padded. This step happens
    before an eventual cropping. Default is None, no padding.
    :param apply_affine_trans: whether to apply affine deformation during generation
    :param apply_nonlin_trans: whether to apply non linear deformation during generation
    :param nonlin_shape_factor: if apply_non_linear_deformation=True, factor between the shapes of the labels and of
    the non-linear field that will be sampled
    :param apply_bias_field: whether to apply a bias field to the created image during generation
    :param bias_shape_factor: if apply_bias_field=True, factor between the shapes of the labels and of the bias field
    that will be sampled
    :param blur_background: Whether background is a regular label, thus blurred with the others.
    :param normalise: whether to normalise data. Default is False.
    :param out_div_32: whether model's outputs must be of shape divisible by 32
    """

    # get shapes
    n_dims = len(labels_shape)
    target_res = format_target_res(target_res, n_dims)
    crop_shape, resample_shape, output_shape, padding_margin = get_shapes(labels_shape,
                                                                          crop_shape,
                                                                          labels_res,
                                                                          target_res,
                                                                          padding_margin,
                                                                          out_div_32)
    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_generation_labels = generation_label_list.shape[0]
    new_generation_label_list = np.arange(n_generation_labels)
    lut = np.zeros(np.max(generation_label_list).astype('int') + 1)
    for n in range(n_generation_labels):
        lut[generation_label_list[n].astype('int')] = n

    # define mandatory inputs
    labels_input = KL.Input(shape=(*labels_shape, 1), name=f'labels_input_{id}')
    means_input = KL.Input(shape=(*new_generation_label_list.shape, n_channels), name=f'means_input_{id}')
    std_devs_input = KL.Input(shape=(*new_generation_label_list.shape, n_channels), name=f'std_devs_input_{id}')
    list_inputs = [labels_input, means_input, std_devs_input]

    # convert labels to new_label_list
    labels = KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(lut, dtype='int32'),
                                           tf.cast(x, dtype='int32')), name=f'convert_labels_{id}')(labels_input)

    # pad labels
    if padding_margin is not None:
        pad = np.transpose(np.array([[0] + padding_margin + [0]] * 2))
        labels = KL.Lambda(lambda x: tf.pad(x, tf.cast(tf.convert_to_tensor(pad), dtype='int32')), name=f'pad_{id}')(labels)
        labels_shape = labels.get_shape().as_list()[1:-1]

    # cropping
    if crop_shape is not None:
        # get maximum cropping indices in each dimension
        cropping_max_val = [labels_shape[i] - crop_shape[i] for i in range(n_dims)]
        # prepare cropping indices and tensor's new shape
        idx = KL.Lambda(lambda x: tf.zeros([1], dtype='int32'), name=f'no_cropping_batch_{id}')(means_input)  # no cropping
        for val_idx, val in enumerate(cropping_max_val):  # draw cropping indices for image dimensions
            if val > 0:
                idx = KL.Lambda(lambda x: tf.concat(
                    [tf.cast(x, dtype='int32'), K.random_uniform([1], minval=0, maxval=val, dtype='int32')], axis=0),
                                name=f'pick_cropping_idx_{val_idx}_{id}')(idx)
            else:
                idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'), tf.zeros([1], dtype='int32')], axis=0),
                                name=f'pick_cropping_idx_{val_idx}_{id}')(idx)
        idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'), tf.zeros([1], dtype='int32')], axis=0),
                        name=f'no_cropping_channel_{id}')(idx)  # no cropping for channel dimension
        patch_shape_tens = KL.Lambda(lambda x: tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32'),
                                     name=f'tensor_cropping_idx_{id}')(means_input)
        # perform cropping
        labels = KL.Lambda(
            lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'), size=tf.cast(x[2], dtype='int32')),
            name=f'cropping_{id}')([labels, idx, patch_shape_tens])
    else:
        crop_shape = labels_shape

    labels = KL.Lambda(lambda x: tf.cast(x, dtype='float'))(labels)

    # if necessary, resample image and labels at target resolution
    if resample_shape is not None:
        labels = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(labels)
        zoom_fact = [r / l for r, l in zip(resample_shape, labels_shape)] 
        labels = nrn_layers.Resize(zoom_fact, interp_method='nearest', name=f'resample_labels_{id}')(labels)

    # deform labels
    if apply_affine_trans | apply_nonlin_trans:
        labels._keras_shape = tuple(labels.get_shape().as_list())
        trans_inputs = [labels]
        # add affine deformation to inputs list
        if apply_affine_trans:
            aff_in = KL.Input(shape=(n_dims + 1, n_dims + 1), name=f'aff_input_{id}')
            list_inputs.append(aff_in)
            trans_inputs.append(aff_in)
        # prepare non-linear deformation field and add it to inputs list
        if apply_nonlin_trans:
            def_field_size = get_nonlin_field_shape(crop_shape, nonlin_shape_factor)
            nonlin_field_in = KL.Input(shape=def_field_size, name=f'nonlin_input_{id}')
            list_inputs.append(nonlin_field_in)
            int_at = 2.0
            zoom = [o / d / int_at for o, d in zip(output_shape, def_field_size)] 
            vel_field = nonlin_field_in
            vel_field = nrn_layers.Resize(zoom, interp_method='linear', name=f'resize_vel_{id}')(vel_field)
            def_field = layers.VecInt(int_steps=5)(vel_field)
            # def_field = nrn_layers.RescaleValues(int_at)(def_field)
            def_field = nrn_layers.Resize(int_at, interp_method='linear', name=f'resize_def_{id}')(def_field)
            trans_inputs.append(def_field)

        # apply deformations
        labels = layers.SpatialTransformer(interp_method='nearest', name=f'trans_{id}')(trans_inputs)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)

    # sample from normal distribution
    image = KL.Lambda(lambda x: tf.random.normal(tf.shape(x)),  name=f'sample_normal_{id}')(labels)

    # build synthetic image
    f_cat = lambda x: tf.concat([x+n_generation_labels*i for i in range(n_channels)], -1)
    cat_labels = KL.Lambda(f_cat, name=f'cat_labels_{id}')(labels)
    f_gather = lambda x: tf.gather(tf.reshape(x[0], [-1]), tf.cast(x[1], dtype='int32'))
    f_map = lambda x: tf.map_fn(f_gather, x, dtype='float32')
    means = KL.Lambda(f_map)([means_input, cat_labels])
    std_devs = KL.Lambda(f_map)([std_devs_input, cat_labels])
    image = KL.Multiply(name=f'mul_std_dev_{id}')([image, std_devs])
    image = KL.Add(name=f'add_means_{id}')([image, means])

    if rand_blur:
        shape = [5] * n_dims 
        lim = [(s - 1) / 2 for s in shape]
        lim = [np.arange(-l, l+1) for l in lim]
        grid = np.meshgrid(*lim, indexing='ij')
        grid = [g ** 2 for g in grid]
        c_grid = KL.Lambda(lambda x: tf.constant(np.stack(grid), dtype='float32'))([])
        sigma = KL.Lambda(lambda x: tf.random.uniform((n_dims,), minval=1e-6, maxval=1))([])
        f = lambda x: x[0] / x[1]**2
        kernel = KL.Lambda(lambda x: tf.map_fn(f, x, dtype='float32'))([c_grid, sigma])
        kernel = KL.Lambda(lambda x: tf.exp( -tf.reduce_sum(x, axis=0)))(kernel)
        kernel = KL.Lambda(lambda x: x[..., None, None] / tf.reduce_sum(x))(kernel)
    else:
        if (target_res is None) | (labels_res == target_res):
            sigma = [0.55] * n_dims
        else:
            sigma = [0.85 * labels_res[i] / target_res[i] for i in range(n_dims)]
        kernel = KL.Lambda(lambda x: tf.convert_to_tensor(add_axis(add_axis(gauss_kernel(sigma, n_dims), -1), -1),
                                                          dtype=x.dtype), name=f'gauss_kernel_{id}')(image)

    if n_channels == 1:
        image = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1] * n_dims),
                          name=f'blur_image_{id}')([image, kernel])
        mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                            tf.zeros_like(x, dtype='float32')), name=f'masking_{id}')(labels)
        if not blur_background:
            blurred_mask = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1] * n_dims),
                                     name=f'blur_mask_{id}')([mask, kernel])
            image = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()), name=f'masked_blurring_{id}')([image, blurred_mask])
            bckgd_mean = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 10), name=f'bckgd_mean_{id}')([])
            bckgd_std = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 5), name=f'bckgd_std_{id}')([])
            rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5), name=f'bool_{id}')([])
            bckgd_mean = KL.Lambda(lambda y: K.switch(y[0],
                                                      KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                      y[1]), name=f'switch_backgd_mean_{id}')([rand_flip, bckgd_mean])
            bckgd_std = KL.Lambda(lambda y: K.switch(y[0],
                                                     KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                     y[1]), name=f'switch_backgd_std_{id}')([rand_flip, bckgd_std])
            background = KL.Lambda(lambda x: x[1] + x[2] * tf.random.normal(tf.shape(x[0])),
                                   name=f'gaussian_bckgd_{id}')([image, bckgd_mean, bckgd_std])
            image = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]),
                              name=f'mask_blurred_image_{id}')([image, mask, background])
        else:
            rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.8), name=f'bool_{id}')([])
            image = KL.Lambda(lambda y: K.switch(y[0], KL.Lambda(
                lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], tf.zeros_like(x[0])), name=f'mask_image_{id}')(
                [y[1], y[2]]), y[1]), name=f'switch_backgd_reset_{id}')([rand_flip, image, mask])

    else:
        # blur each image channel separately
        split = KL.Lambda(lambda x: tf.split(x, [1]*n_channels, axis=-1))(image)
        image = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1] * n_dims),
                          name=f'blurring_0_{id}')([split[0], kernel])
        for i in range(1, n_channels):
            temp_blurred = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1] * n_dims),
                                     name=f'blurring_{i}_{id}')([split[i], kernel])
            mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                                tf.zeros_like(x, dtype='float32')), name=f'masking_{i}_{id}')(labels)
            if not blur_background:
                blurred_mask = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1] * n_dims),
                                         name=f'blur_mask_{i}_{id}')([mask, kernel])
                temp_blurred = KL.Lambda(lambda x: x[0] / (x[1]+K.epsilon()),
                                         name=f'masked_blurring_{i}_{id}')([temp_blurred, blurred_mask])
                bckgd_mean = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 10), name=f'bckgd_mean_{i}_{id}')([])
                bckgd_std = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 5), name=f'bckgd_std_{i}_{id}')([])
                rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5), name=f'bool{i}_{id}')([])
                bckgd_mean = KL.Lambda(lambda y: K.switch(y[0],
                                                          KL.Lambda(lambda x: tf.zeros_like(x, dtype='float32'))(y[1]),
                                                          y[1]), name=f'switch_backgd_mean{i}_{id}')([rand_flip, bckgd_mean])
                bckgd_std = KL.Lambda(lambda y: K.switch(y[0],
                                                         KL.Lambda(lambda x: tf.zeros_like(x, dtype='float32'))(y[1]),
                                                         y[1]), name=f'switch_backgd_std_{i}_{id}')([rand_flip, bckgd_std])
                background = KL.Lambda(lambda x: x[1] + x[2] * tf.random.normal(tf.shape(x[0])),
                                       name=f'gaussian_bckgd_{i}_{id}')([temp_blurred, bckgd_mean, bckgd_std])
                temp_blurred = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]),
                                         name=f'mask_blurred_image_{i}_{id}')([temp_blurred, mask, background])
            else:
                rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.8), name=f'boo{i}_{id}')([])
                image = KL.Lambda(lambda y: K.switch(y[0], KL.Lambda(
                    lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], tf.zeros_like(x[0])), name=f'mask_image_{i}_{id}')(
                    [y[1], y[2]]), y[1]), name=f'switch_backgd_reset_{i}_{id}')([rand_flip, image, mask])
            image = KL.Lambda(lambda x: tf.concat([x[0], x[1]], -1),
                              name=f'cat_blurring_{i}_{id}')([image, temp_blurred])

    # apply bias field
    if apply_bias_field:
        # format bias field and add it to inputs list
        bias_field_size = get_bias_field_shape(output_shape, bias_shape_factor)
        bias_field_in = KL.Input(shape=bias_field_size, name=f'bias_input_{id}')
        list_inputs.append(bias_field_in)
        # resize bias field and apply it to image
        zoom_fact = [o / d for o, d in zip(output_shape, bias_field_size)] 
        bias_field = nrn_layers.Resize(zoom_fact, interp_method='linear', name=f'log_bias_{id}')(bias_field_in)
        bias_field = KL.Lambda(lambda x: K.exp(x), name=f'bias_field_{id}')(bias_field)
        image._keras_shape = tuple(image.get_shape().as_list())
        bias_field._keras_shape = tuple(bias_field.get_shape().as_list())
        image = KL.multiply([bias_field, image])

    # make sure image's intensities are between 0 and 255
    image = KL.Lambda(lambda x: K.clip(x, 0, 255), name=f'clipping_{id}')(image)

    # contrast stretching
    image = KL.Lambda(
        lambda x: x * tf.random.uniform([1], minval=0.6, maxval=1.4) + tf.random.uniform([1], minval=-30, maxval=30),
        name=f'stretching_{id}')(image)

    # convert labels back to original values and remove unwanted labels
    if convert_back:
        out_lut = [x if x in segmentation_label_list else 0 for x in generation_label_list]
    else:
        # Rebase wanted indices into [0, N-1] for one-hot encoding.
        n = 0
        out_lut = [None] * len(generation_label_list)
        for i, x in enumerate(generation_label_list):
            out = -1
            if x in segmentation_label_list:
                out = n
                n += 1
            out_lut[i] = out
    labels = KL.Lambda(lambda x: tf.gather(tf.cast(out_lut, dtype='int32'),
                                           tf.cast(x, dtype='int32')), name=f'labels_back_{id}')(labels)

    # normalise the produced image (include labels_out, so this layer is not removed when plugging in other keras model)
    if normalise:
        m = KL.Lambda(lambda x: K.min(x), name=f'min_{id}')(image)
        M = KL.Lambda(lambda x: K.max(x), name=f'max_{id}')(image)
        image = KL.Lambda(lambda x: (x[0]-x[1])/(x[2]-x[1]), name=f'normalisation_{id}')([image, m, M])
    else:
        image = KL.Lambda(lambda x: x[0] + K.zeros(1), name=f'dummy_{id}')([image])

    # gamma augmentation
    image = KL.Lambda(lambda x: tf.math.pow(x[0], tf.math.exp(tf.random.normal([1], mean=0, stddev=0.25))),
                      name=f'gamma_{id}')([image, labels])

    outputs = [image, labels]
    if apply_nonlin_trans:
        outputs.append(vel_field)
    brain_model = keras.Model(inputs=list_inputs, outputs=outputs)
    return brain_model, def_field_size, bias_field_size
