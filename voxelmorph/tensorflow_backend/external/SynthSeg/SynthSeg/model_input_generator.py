from scipy.interpolate import interpn
import numpy as np
import numpy.random as npr

from .utils import draw_values
from .utils import add_axis, load_volfile
from .utils import create_affine_transformation_matrix
from .utils import get_nonlin_field_shape, get_bias_field_shape


def build_model_input_generator(labels_paths,
                                generation_label_list,
                                warp_shape,
                                bias_shape,
                                n_channels=1,
                                classes_list=None,
                                means_range=None,
                                std_devs_range=None,
                                use_specific_stats_for_channel=False,
                                apply_affine_trans=False,
                                scaling_range=None,
                                rotation_range=None,
                                shearing_range=None,
                                apply_nonlin_trans=True,
                                nonlin_std_dev=3,
                                apply_bias_field=True,
                                bias_field_std_dev=0.3,
                                blur_background=True,
                                background_paths=None,
                                head=True,
                                batch_size=1,
                                same_subj=False,
                                flipping=True,
                                rand_scale=True):

    if scaling_range is None:
        scaling_range = [0.93, 1.07]
    elif isinstance(scaling_range, (int, float)):
        scaling_range = [1 - scaling_range, 1 + scaling_range]
    if shearing_range is None:
        shearing_range = [-0.01, 0.01]
    elif isinstance(shearing_range, (int, float)) != 'list':
        shearing_range = [-shearing_range, shearing_range]

    # get label info
    n_lab = np.size(generation_label_list)
    dataset = [] if isinstance(labels_paths[0], str) else labels_paths

    # Generate!
    sample = dataset[0] if dataset else load_volfile(labels_paths[0])
    n_dims = len(sample.shape)
    dims = np.arange(n_dims)
    while True:

        # Generate all images within a batch from the same file.
        if same_subj:
            indices = npr.randint(len(labels_paths), size=1).repeat(batch_size)
        else:
            indices = npr.randint(len(labels_paths), size=batch_size)

        # initialise input tensors
        y_all = []
        means_all = []
        std_devs_all = []
        aff_all = []
        nonlinear_field_all = []
        bias_field_all = []

        # Flip each segmentation in batch identically.
        is_flip = np.random.randn(n_dims) > 0 if flipping else np.full(n_dims, False)

        for idx in indices:

            # add labels to inputs
            if dataset:
                y = dataset[idx]
            else:
                y = load_volfile(labels_paths[idx], dtype='int', squeeze=True)
                if background_paths is not None:
                    idx_258 = np.where(y == 258)
                    if np.any(idx_258):
                        background_idx = npr.randint(len(background_paths))
                        background = np.squeeze(load_volfile(background_paths[background_idx], dtype='int'))
                        background_shape = background.shape
                        if np.all(np.array(background_shape) == background_shape[0]):  # flip if same dimensions
                            background = np.flip(background, tuple([i for i in range(3) if np.random.normal() > 0]))
                        assert background.shape == y.shape, \
                            'background patches should have same shape than training labels. ' \
                            'Had {0} and {1}'.format(background.shape, y.shape)
                        y[idx_258] = background[idx_258]

            y = np.flip(y, axis=dims[is_flip])
            y_all.append(add_axis(y, axis=-2))

            # add means and standard deviations to inputs
            means = np.empty((n_lab, 0))
            std_devs = np.empty((n_lab, 0))
            for channel in range(n_channels):
                # retrieve channel specifi stats if necessary
                if use_specific_stats_for_channel:
                    tmp_means_range = means_range[2*channel:2*channel + 2, :]
                    tmp_std_devs_range = std_devs_range[2*channel:2*channel + 2, :]
                else:
                    tmp_means_range = means_range
                    tmp_std_devs_range = std_devs_range
                # draw means and std devs from priors
                tmp_means, tmp_stds = means_stds_no_rules(n_lab,
                                                          tmp_means_range,
                                                          tmp_std_devs_range)
                if blur_background:
                    tmp_means[0] = np.random.uniform(low=0, high=225)
                    tmp_stds[0] = np.random.uniform(low=0, high=25)
                else:
                    tmp_means[0] = 0
                    tmp_stds[0] = 0
                means = np.concatenate([means, tmp_means], axis=1)
                std_devs = np.concatenate([std_devs, tmp_stds], axis=1)
            means_all.append(add_axis(means))
            std_devs_all.append(add_axis(std_devs))

            # add inputs according to augmentation specification
            if apply_affine_trans:
                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = npr.uniform(low=scaling_range[0], high=scaling_range[1], size=n_dims)
                rotation_angle = draw_rotation_angle(rotation_range, n_dims)
                shearing = npr.uniform(low=shearing_range[0], high=shearing_range[1], size=n_dims ** 2 - n_dims)
                aff = create_affine_transformation_matrix(n_dims, scaling, rotation_angle, shearing)
                aff_all.append(add_axis(aff))

            if apply_nonlin_trans:
                if len(warp_shape) == n_dims:
                    warp_shape = [*warp_shape, n_dims]
                scale = nonlin_std_dev * (npr.rand() if rand_scale else 1)
                nonlinear_field = npr.normal(scale=scale, size=warp_shape)
                nonlinear_field_all.append(add_axis(nonlinear_field))

            if apply_bias_field:
                if len(bias_shape) == n_dims:
                    bias_shape = [*bias_shape, 1]
                scale = bias_field_std_dev  * (npr.rand() if rand_scale else 1)
                bias_field = npr.normal(scale=scale, size=bias_shape)
                bias_field_all.append(add_axis(bias_field))

        # build list of inputs to augmentation model
        inputs_vals = [y_all, means_all, std_devs_all]
        if apply_affine_trans:
            inputs_vals.append(aff_all)
        if apply_nonlin_trans:
            inputs_vals.append(nonlinear_field_all)
        if apply_bias_field:
            inputs_vals.append(bias_field_all)

        # put images and labels (concatenated if batch_size>1) into a tuple of 2 elements: (cat_images, cat_labels)
        if batch_size > 1:
            inputs_vals = [np.concatenate(item, 0) for item in inputs_vals]
        else:
            inputs_vals = [item[0] for item in inputs_vals]

        yield inputs_vals


def means_stds_no_rules(n_lab, means_range, std_devs_range):

    # draw values
    means = draw_values(means_range, n_lab, 'means_range')
    stds = draw_values(std_devs_range, n_lab, 'std_devs_range')

    return means, stds


def means_stds_with_rl_grouping(n_sided, n_neutral, means_range, std_devs_range):

    # draw values
    means = draw_values(means_range, n_sided+n_neutral, 'means_range')
    stds = draw_values(std_devs_range, n_sided+n_neutral, 'std_devs_range')

    # regroup neutral and sided labels
    means = np.concatenate([means[:n_neutral], means[n_neutral:], means[n_neutral:]])
    stds = np.concatenate([stds[:n_neutral], stds[n_neutral:], stds[n_neutral:]])

    return means, stds


def means_stds_with_classes(classes_list, means_range, std_devs_range):

    # get unique list of classes and reorder them from 0 to N-1
    classes_lut = np.zeros(np.max(classes_list).astype('int') + 1)
    _, idx = np.unique(classes_list, return_index=True)
    unique_classes = np.sort(classes_list[np.sort(idx)])
    n_stats = len(unique_classes)

    # reformat classes_list
    for n in range(n_stats):
        classes_lut[unique_classes[n].astype('int')] = n
    classes_list = (classes_lut[classes_list]).astype('int')

    # draw values
    means = draw_values(means_range, n_stats, 'means_range')
    stds = draw_values(std_devs_range, n_stats, 'std_devs_range')

    # reorder values
    means = means[classes_list]
    stds = stds[classes_list]

    return means, stds


def means_stds_fs_labels_with_relations(means_range, std_devs_range, min_diff=15, head=True):

    # draw gm wm and csf means
    gm_wm_csf_means = np.zeros(3)
    while (abs(gm_wm_csf_means[1] - gm_wm_csf_means[0]) < min_diff) | \
          (abs(gm_wm_csf_means[1] - gm_wm_csf_means[2]) < min_diff) | \
          (abs(gm_wm_csf_means[0] - gm_wm_csf_means[2]) < min_diff):
        gm_wm_csf_means = draw_values(means_range, 3, 'means_range')

    # apply relations
    wm = gm_wm_csf_means[0]
    gm = gm_wm_csf_means[1]
    csf = gm_wm_csf_means[2]
    csf_like = csf * npr.uniform(low=0.95, high=1.05)
    alpha_thalamus = npr.uniform(low=0.4, high=0.9)
    thalamus = alpha_thalamus*gm + (1-alpha_thalamus)*wm
    cerebellum_wm = wm * npr.uniform(low=0.7, high=1.3)
    cerebellum_gm = gm * npr.uniform(low=0.7, high=1.3)
    caudate = gm * npr.uniform(low=0.9, high=1.1)
    putamen = gm * npr.uniform(low=0.9, high=1.1)
    hippocampus = gm * npr.uniform(low=0.9, high=1.1)
    amygdala = gm * npr.uniform(low=0.9, high=1.1)
    accumbens = caudate * npr.uniform(low=0.9, high=1.1)
    pallidum = wm * npr.uniform(low=0.8, high=1.2)
    brainstem = wm * npr.uniform(low=0.8, high=1.2)
    alpha_ventralDC = npr.uniform(low=0.1, high=0.6)
    ventralDC = alpha_ventralDC*gm + (1-alpha_ventralDC)*wm
    alpha_choroid = npr.uniform(low=0.0, high=1.0)
    choroid = alpha_choroid*csf + (1-alpha_choroid)*wm

    # regroup structures
    neutral_means = [np.zeros(1), csf_like, csf_like, brainstem, csf]
    sided_means = [wm, gm, csf_like, csf_like, cerebellum_wm, cerebellum_gm, thalamus, caudate, putamen, pallidum,
                   hippocampus, amygdala, accumbens, ventralDC, choroid]

    # draw std deviations
    std = draw_values(std_devs_range, 17, 'std_devs_range')
    neutral_stds = [np.zeros(1), std[1], std[1], std[2], std[3]]
    sided_stds = [std[4], std[5], std[1], std[1], std[6], std[7], std[8], std[9], std[10], std[11], std[12], std[13],
                  std[14], std[15], std[16]]

    # add means and variances for extra head labels if necessary
    if head:
        # means
        extra_means = draw_values(means_range, 2, 'means_range')
        skull = extra_means[0]
        soft_non_brain = extra_means[1]
        eye = csf * npr.uniform(low=0.95, high=1.05)
        optic_chiasm = wm * npr.uniform(low=0.8, high=1.2)
        vessel = csf * npr.uniform(low=0.7, high=1.3)
        neutral_means += [csf_like, optic_chiasm, skull, soft_non_brain, eye]
        sided_means.insert(-1, vessel)
        # std dev
        extra_std = draw_values(std_devs_range, 4, 'std_devs_range')
        neutral_stds += [std[1], extra_std[0], extra_std[1], extra_std[2], std[1]]
        sided_stds.insert(-1, extra_std[3])

    means = np.concatenate([np.array(neutral_means), np.array(sided_means), np.array(sided_means)])
    stds = np.concatenate([np.array(neutral_stds), np.array(sided_stds), np.array(sided_stds)])

    return means, stds


def means_stds_with_stats(n_sided, n_neutral, means_range, std_devs_range):

    # draw values
    means = draw_values(means_range[:, :n_sided+n_neutral], n_sided+n_neutral, 'means_range')
    stds = draw_values(std_devs_range[:, :n_sided+n_neutral], n_sided+n_neutral, 'std_devs_range')

    # regroup neutral and sided labels
    means = np.concatenate([means[:n_neutral], means[n_neutral:], means[n_neutral:]])
    stds = np.concatenate([stds[:n_neutral], stds[n_neutral:], stds[n_neutral:]])

    return means, stds


def means_stds_classes_with_stats(classes_list, means_range, std_devs_range):

    # get unique classes and corresponding stats
    unique_classes, unique_idx = np.unique(classes_list, return_index=True)
    n_unique = unique_classes.shape[0]
    unique_means_range = means_range[:, unique_idx]
    unique_std_devs_range = std_devs_range[:, unique_idx]

    # draw values
    unique_means = draw_values(unique_means_range, n_unique, 'means_range')
    unique_stds = draw_values(unique_std_devs_range, n_unique, 'std_devs_range')

    # put stats back in order
    n_classes = classes_list.shape[0]
    means = np.zeros((n_classes, 1))
    stds = np.zeros((n_classes, 1))
    for idx_class, tmp_class in enumerate(unique_classes):
        means[classes_list == tmp_class] = unique_means[idx_class]
        stds[classes_list == tmp_class] = unique_stds[idx_class]

    return means, stds


def draw_rotation_angle(rotation_range, n_dims):
    # reformat rotation_range
    if not isinstance(rotation_range, np.ndarray):
        if rotation_range is None:
            rotation_range = [-10, 10]
        elif isinstance(rotation_range, (int, float)):
            rotation_range = [-rotation_range, rotation_range]
        elif isinstance(rotation_range, (list, tuple)):
            assert len(rotation_range) == 2, 'if list, rotation_range should be of length 2.'
        else:
            raise Exception('If not numpy array, rotation_range should be None, int, float or list.')
        if n_dims == 2:
            rotation_angle = npr.uniform(low=rotation_range[0], high=rotation_range[1], size=1)
        else:
            rotation_angle = npr.uniform(low=rotation_range[0], high=rotation_range[1], size=n_dims)

    elif isinstance(rotation_range, np.ndarray):
        assert rotation_range.shape == (2, n_dims), 'rotation_range should be array of size {}'.format((2, n_dims))
        rotation_angle = npr.uniform(low=rotation_range[0, :], high=rotation_range[1, :])

    else:
        raise Exception('rotation_range should be None, int, float, list or numpy array')

    return rotation_angle
