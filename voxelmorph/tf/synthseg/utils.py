import os
import glob
import math
import pickle
import numpy as np
import nibabel as nib
import numpy.random as npr
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_edt


# IMAGE MANIPULATION AND MISCELLANEOUS FUNCTIONS

def rescale_volume(volume, use_positive_only=True, min_percentile=0.025, max_percentile=0.975):

    # sort intensities
    if use_positive_only:
        intensities = np.sort(volume[volume > 0])
    else:
        intensities = np.sort(volume.flatten())

    # define robust max and min
    robust_min = np.maximum(0, intensities[int(intensities.shape[0] * min_percentile)])
    robust_max = intensities[int(intensities.shape[0] * max_percentile)]

    # trim values outside range
    volume = np.clip(volume, robust_min, robust_max)

    # rescale image
    volume = (volume-robust_min) / (robust_max-robust_min) * 255

    return volume


def rescale_images_in_folder(image_dir, result_dir, use_positive_only=True,
                             min_percentile=0.025, max_percentile=0.975, recompute=True):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    im_list = list_images_in_folder(image_dir)
    for path in im_list:
        new_path_image = os.path.join(result_dir, os.path.basename(path))
        if (not os.path.isfile(new_path_image)) | recompute:
            im, aff, h = load_volfile(path, im_only=False)
            im = rescale_volume(im, use_positive_only, min_percentile, max_percentile)
            save_volfile(im, aff, h, new_path_image)


def divisors(n):
    divs = {1, n}
    for i in range(2, int(math.sqrt(n))+1):
        if n % i == 0:
            divs.update((i, n//i))
    return sorted(list(divs))


def find_closest_number_divisible_by_m(n, m, smaller_ans=True):
    # quotient
    q = int(n / m)
    # 1st possible closest number
    n1 = m * q
    # 2nd possible closest number
    if (n * m) > 0:
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))
    # find closest solution
    if (abs(n - n1) < abs(n - n2)) | smaller_ans:
        return n1
    else:
        return n2


def gauss_kernel(sigma, n_dims, shape=None, channels=1):
    if not isinstance(sigma, (list, tuple)):
        sigma = [float(sigma)] * n_dims
    elif len(sigma) == 1:
        sigma = sigma * n_dims
    else:
        assert len(sigma) == n_dims, \
            'sigma should be of length 1 or same length as n_dims: got {} instead of 1 or {}'.format(len(sigma), n_dims)
    if shape is None:
        shape = list()
        for s in sigma:
            sha = math.ceil(2.5*s)
            if sha % 2 == 0:
                shape.append(sha + 1)
            else:
                shape.append(sha)
    else:
        assert len(shape) == n_dims, \
            'shape should have same length as n_dims: got {} instead of {}'.format(len(shape), n_dims)
    if n_dims == 2:
        m, n = [(ss-1.)/2. for ss in shape]
        x, y = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x/(sigma[0]**2) + y*y/(sigma[1]**2)) / 2)
    elif n_dims == 3:
        m, n, p = [(ss-1.)/2. for ss in shape]
        x, y, z = np.ogrid[-m:m+1, -n:n+1, -p:p+1]
        h = np.exp(-(x*x/(sigma[0]**2) + y*y/(sigma[1])**2 + z*z/(sigma[2]**2)) / 2)
    else:
        raise Exception('dimension > 3 not supported')
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    if channels > 1:
        h = np.stack([h]*channels, axis=-1)
    return h


def add_axis(x, axis=0):
    if axis == 0:
        return x[np.newaxis, ...]
    elif axis == -1:
        return x[..., np.newaxis]
    elif axis == -2:
        return x[np.newaxis, ..., np.newaxis]
    else:
        raise Exception('axis should be 0 (first), -1 (last), or -2 (first and last)')


def get_resample_factor(target_res, header, n_dims):
    if target_res is not None:
        labels_res = header['pixdim'][1:n_dims+1]
        if not isinstance(target_res, list):
            target_res = [target_res]
        if len(target_res) == 1:
            target_res = target_res * n_dims
        else:
            assert len(target_res) == n_dims, \
                'new_resolution must be of length 1 or n_dims ({}): got {}'.format(n_dims, len(target_res))
        resample_factor = [labels_res[i] / float(target_res[i]) for i in range(n_dims)]
    else:
        resample_factor = None
    return resample_factor


def format_target_res(target_res, n_dims):
    if target_res is not None:
        # format target res
        if isinstance(target_res, (int, float)):
            target_res = [target_res] * n_dims
        if isinstance(target_res, (list, tuple)):
            if len(target_res) == 1:
                target_res = target_res * n_dims
            elif len(target_res) != n_dims:
                raise TypeError('target_res should be float, list of size 1 or {0}, or None. '
                                'Had {1}'.format(n_dims, target_res))
    return target_res


def get_padding_margin(cropping, loss_cropping):

    if (cropping is None) | (loss_cropping is None):
        padding_margin = None
    else:
        # bring cropping and loss_cropping to common format and get padding_margin
        if isinstance(cropping, (int, float)) & isinstance(loss_cropping, (int, float)):
            padding_margin = int((cropping - loss_cropping) / 2)
        elif isinstance(cropping, (list, tuple)) & isinstance(loss_cropping, (list, tuple)):
            n_dims = max(len(cropping), len(loss_cropping))
            if len(cropping) < n_dims:
                if len(cropping) == 1:
                    cropping = cropping * n_dims
                else:
                    raise ValueError('cropping should have length 1 or n_dims, had {}'.format(len(cropping)))
            if len(loss_cropping) < n_dims:
                if len(loss_cropping) == 1:
                    loss_cropping = loss_cropping * n_dims
                else:
                    raise ValueError('loss_cropping should have length 1 or n_dims, had {}'.format(len(loss_cropping)))
            padding_margin = [int((cropping[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        elif isinstance(cropping, (list, tuple)):
            if isinstance(loss_cropping, (int, float)):
                n_dims = len(cropping)
                loss_cropping = [loss_cropping] * n_dims
                padding_margin = [int((cropping[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
            else:
                raise ValueError('loss_cropping should be int, float, list, or tuple, had %s' % type(loss_cropping))
        elif isinstance(loss_cropping, (list, tuple)):
            if isinstance(cropping, (int, float)):
                n_dims = len(loss_cropping)
                cropping = [cropping] * n_dims
                padding_margin = [int((cropping[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
            else:
                raise ValueError('cropping should be int, float, list, or tuple, had {}'.format(type(cropping)))
        else:
            raise ValueError('cropping and loss cropping should be list, tuple, int or float.'
                             'Had cropping: {0} and loss_cropping: {1}.'.format(type(cropping), type(loss_cropping)))

    return padding_margin


def get_shapes(label_shape, output_shape, labels_res, target_res, padding_margin, out_div_32):

    n_dims = len(labels_res)

    # get new labels shape if padding
    if padding_margin is not None:
        # format loss_crop_shape
        padding_margin = reformat_variable(padding_margin, n_dims, dtype='int')
        label_shape = [label_shape[i] + 2*padding_margin[i] for i in range(n_dims)]

    # get resampling factor
    if target_res is not None:
        if labels_res != target_res:
            resample_factor = [labels_res[i] / float(target_res[i]) for i in range(n_dims)]
        else:
            resample_factor = None
    else:
        resample_factor = None

    # get resample and cropping shapes
    # output shape specified
    if output_shape is not None:
        # format output shape
        output_shape = reformat_variable(output_shape, n_dims, dtype='int')
        if out_div_32:
            tmp_shape = [find_closest_number_divisible_by_m(s, 32, smaller_ans=True) for s in output_shape]
            if output_shape != tmp_shape:
                print('provided output shape {0} not divisible by 32, changed to {1}'.format(output_shape, tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [int(output_shape[i]/resample_factor[i]) for i in range(n_dims)]
            resample_shape = output_shape
        else:
            cropping_shape = output_shape
            resample_shape = None
        # check that obtained cropping shape is lower than labels shape
        tmp_shape = [min(label_shape[i], cropping_shape[i]) for i in range(n_dims)]
        if cropping_shape != tmp_shape:
            cropping_shape = tmp_shape
            if resample_shape is None:
                resample_shape = output_shape

    # no output shape specified, so no cropping in that case
    else:
        cropping_shape = None
        if resample_factor is not None:
            resample_shape = [int(label_shape[i]*resample_factor[i]) for i in range(n_dims)]
            if out_div_32:
                resample_shape = [find_closest_number_divisible_by_m(s, 32, smaller_ans=False) for s in resample_shape]
            output_shape = resample_shape
        else:
            tmp_shape = [find_closest_number_divisible_by_m(s, 32, smaller_ans=False) for s in label_shape]
            if (tmp_shape != label_shape) & out_div_32:
                print('label shape {0} not divisible by 32, resampled to {1}'.format(label_shape, tmp_shape))
                resample_shape = tmp_shape
                output_shape = tmp_shape
            else:
                resample_shape = None
                output_shape = label_shape

    return cropping_shape, resample_shape, output_shape, padding_margin


def build_training_generator(gen, batch_size):
    while True:
        inputs = next(gen)
        if batch_size > 1:
            target = np.concatenate([add_axis(np.zeros(1))] * batch_size, 0)
        else:
            target = add_axis(np.zeros(1))
        yield inputs, target


def draw_values(values_range, size, atype):
    if values_range is None:
        if atype == 'means_range':
            values_range = np.array([[25] * size, [225] * size])
        else:
            values_range = np.array([[5] * size, [25] * size])
        values = add_axis(npr.uniform(low=values_range[0, :], high=values_range[1, :]), -1)
    elif isinstance(values_range, (list, tuple)):
        values_range = np.array([[values_range[0]] * size, [values_range[1]] * size])
        values = add_axis(npr.uniform(low=values_range[0, :], high=values_range[1, :]), -1)
    elif isinstance(values_range, np.ndarray):
        assert values_range.shape[1] == size, '{0} should be (2,{1}), got {2}'.format(atype, size, values_range.shape)
        n_modalities = int(values_range.shape[0] / 2)
        idx = npr.randint(n_modalities)
        values = add_axis(npr.normal(loc=values_range[2*idx, :], scale=values_range[2*idx+1, :]), -1)
    else:
        raise ValueError('{} should be a list, an array, or None'.format(atype))
    return values


# AUGMENTATION FUNCTIONS

def draw_data_augm_params(nonlinear_field_size, bias_field_size, n_dims,
                          scaling_low=0.93, scaling_high=1.07,
                          rotation_low=-10, rotation_high=10,
                          shearing_low=-0.01, shearing_high=0.01,
                          non_linear_scale=3.0,
                          bias_field_scale=0.3):

    # get affine transformation: rotate, scale, shear (translation done during random cropping)
    scaling = np.random.uniform(low=scaling_low, high=scaling_high, size=n_dims)
    if n_dims == 2:
        rotation = np.random.uniform(low=rotation_low, high=rotation_high, size=1)
    else:
        rotation = np.random.uniform(low=rotation_low, high=rotation_high, size=n_dims)
    shearing = np.random.uniform(low=shearing_low, high=shearing_high, size=n_dims**2-n_dims)
    T = create_affine_transformation_matrix(n_dims, scaling, rotation, shearing)

    # initiate non-linear deformation a nd bias fields
    nonlinear_field = np.random.normal(loc=0, scale=non_linear_scale * np.random.rand(), size=nonlinear_field_size)
    bias_field = np.random.normal(loc=0, scale=bias_field_scale * np.random.rand(), size=bias_field_size)

    # random value to determine if we flip the image
    rand_flip = np.random.random(size=[1])

    # randomly select one axis
    flipping_axis = np.random.randint(0, n_dims, size=1)

    return nonlinear_field, T, bias_field, rand_flip, flipping_axis


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    T_scaling = np.eye(n_dims + 1)
    T_shearing = np.eye(n_dims + 1)
    T_translation = np.eye(n_dims + 1)

    if scaling is not None:
        T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        T_shearing[shearing_index] = shearing

    if translation is not None:
        T_translation[np.arange(n_dims), n_dims * np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot = np.eye(n_dims + 1)
        T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[0]),
                                                                 np.sin(rotation[0]),
                                                                 np.sin(rotation[0]) * -1,
                                                                 np.cos(rotation[0])]
        return T_translation @ T_rot @ T_shearing @ T_scaling

    else:

        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot1 = np.eye(n_dims + 1)
        T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                  np.sin(rotation[0]),
                                                                  np.sin(rotation[0]) * -1,
                                                                  np.cos(rotation[0])]
        T_rot2 = np.eye(n_dims + 1)
        T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                  np.sin(rotation[1]) * -1,
                                                                  np.sin(rotation[1]),
                                                                  np.cos(rotation[1])]
        T_rot3 = np.eye(n_dims + 1)
        T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                  np.sin(rotation[2]),
                                                                  np.sin(rotation[2]) * -1,
                                                                  np.cos(rotation[2])]
        return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling


def get_nonlin_field_shape(patch_shape, factor):
    return tuple([math.ceil(s*factor) for s in patch_shape]+[len(patch_shape)])


def get_bias_field_shape(patch_shape, factor):
    return tuple([math.ceil(s*factor) for s in patch_shape]+[1])


def crop_volume(volume, margin=0, label=None, vox2ras=None):

    # crop volume around specified labels
    if label is not None:
        # reformat labels
        if isinstance(label, (list, tuple)):
            label = [int(la) for la in label]
        elif isinstance(label, (float, int)):
            label = [int(label)]
        elif label == 'hippo':
            label = [17, 53]
        else:
            raise Exception('label should be list or int')
        # build mask
        mask = np.full(volume.shape, False, dtype=bool)
        for la in label:
            mask = mask | (volume == la)
        if label == 'hippo':
            mask = mask | (volume > 20000)
        # mask volume
        volume[~mask] = 0

    # find cropping indices
    indices = np.nonzero(volume)
    min_indices = np.maximum(np.array([np.min(idx) for idx in indices]) - margin, 0)
    max_indices = np.minimum(np.array([np.max(idx) for idx in indices]) + margin, np.array(volume.shape))
    cropping = np.concatenate([min_indices, max_indices])

    # crop volume
    volume = volume[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]

    if vox2ras is not None:
        vox2ras[0:3, -1] = vox2ras[0:3, -1] + vox2ras[:3, :3] @ min_indices - 1
        return volume, cropping, vox2ras
    else:
        return volume, cropping


def crop_array_with_idx(x, crop_idx, n_dims, vox2ras=None):
    # crop image
    if n_dims == 2:
        x = x[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]]
    elif n_dims == 3:
        x = x[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]]
    else:
        raise Exception('cannot crop images of size greater than 3')
    if vox2ras is not None:
        vox2ras[0:3, -1] = vox2ras[0:3, -1] + vox2ras[:3, :3] @ crop_idx[:3] - 1
        return x, vox2ras
    else:
        return x


# LABELS EDIT FUNCTIONS

def check_labels(labels_folder):
    # prepare data files
    labels_paths = list_images_in_folder(labels_folder)
    assert len(labels_paths) > 0, "Could not find any training data"
    # loop through files
    vol_list = list()
    aff_list = list()
    res_list = list()
    uni_list = list()
    n_labels = len(labels_paths)
    for lab_idx, path in enumerate(labels_paths):
        if lab_idx == 0:
            print('processing image {}/{}'.format(1, n_labels))
        elif lab_idx % 5 == 4:
            print('processing image {}/{}'.format(lab_idx + 1, n_labels))
        vol, aff, h = load_volfile(path, im_only=False)
        aff = np.round(aff[:3, :3], 2).tolist()
        try:
            res = np.round(np.array(h['pixdim'][1:3 + 1]), 2).tolist()  # nifty image
        except ValueError:
            res = np.array(h['delta']).tolist()  # mgz image
        uni = np.unique(vol).tolist()
        if vol.shape not in vol_list:
            vol_list.append(vol.shape)
        if aff not in aff_list:
            aff_list.append(aff)
        if res not in res_list:
            res_list.append(res)
        if uni not in uni_list:
            uni_list.append(uni)
    aff_list2 = [np.around(np.array(a), 2).tolist() for a in aff_list]
    aff_list3 = list()
    for a in aff_list2:
        if a not in aff_list3:
            aff_list3.append(a)
    res_list2 = [np.around(np.array(r), 2).tolist() for r in res_list]
    res_list3 = list()
    for r in res_list2:
        if r not in res_list3:
            res_list3.append(r)
    return vol_list, aff_list3, res_list3, uni_list


def check_images_and_labels(labels_folder, images_folder):

    labels_paths = sorted([os.path.join(labels_folder, p) for p in os.listdir(labels_folder) if
                           os.path.isfile(os.path.join(labels_folder, p))])
    images_paths = sorted([os.path.join(images_folder, p) for p in os.listdir(images_folder) if
                           os.path.isfile(os.path.join(images_folder, p))])

    for lab_path, im_path in zip(labels_paths, images_paths):
        print('\n'+os.path.basename(lab_path))
        lab, aff_lab, h_lab = load_volfile(lab_path, im_only=False)
        im, aff_im, h_im = load_volfile(im_path, im_only=False)
        aff_lab_list = np.round(aff_lab, 2).tolist()
        aff_im_list = np.round(aff_im, 2).tolist()
        if aff_lab_list != aff_im_list:
            print('aff mismatch :' + lab_path)
            print(aff_lab_list)
            print(aff_im_list)
        if lab.shape != im.shape:
            print('shape mismatch :' + lab_path)
            print(lab.shape)
            print(im.shape)


def correct_labels(labels_dir, list_incorrect_labels, list_correct_labels, results_folder, recompute=False):

    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    assert len(labels_paths) > 0, "Could not find any training data"

    # create results dir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for la, path_label in enumerate(labels_paths):

        filename = os.path.basename(path_label)
        path_new_labels = os.path.join(results_folder, filename)

        if (not os.path.exists(path_new_labels)) | recompute:

            # load image
            if la == 0:
                print('processing labels {}/{}'.format(1, len(labels_paths)))
            elif (la + 1) % 100 == 0:
                print('processing labels {0}/{1}'.format(la + 1, len(labels_paths)))
            im, vox2ras, header = load_volfile(path_label, im_only=False)
            im_labels = np.unique(im)

            # correct labels for this file
            previous_correct_labels = None
            distance_map_list = None
            for incorrect_label, correct_label in zip(list_incorrect_labels, list_correct_labels):
                if incorrect_label in im_labels:
                    incorrect_voxels = np.where(im == incorrect_label)
                    if isinstance(correct_label, (int, float)):
                        im[incorrect_voxels] = correct_label
                    elif isinstance(correct_label, (tuple, list)):
                        if correct_label != previous_correct_labels:
                            distance_map_list = [distance_transform_edt(np.logical_not(im == lab))
                                                 for lab in correct_label]
                            previous_correct_labels = correct_label
                        distances_correct = np.stack([dist[incorrect_voxels] for dist in distance_map_list])
                        idx_correct_lab = np.argmin(distances_correct, axis=0)
                        im[incorrect_voxels] = np.array(correct_label)[idx_correct_lab]

            # save corrected labels
            if ('.nii.gz' in filename) | ('.mgz' in filename):
                # save new labels
                save_volfile(im, vox2ras, header, path_new_labels)
            elif '.npz' in filename:
                np.savez_compressed(path_new_labels, vol=im.astype('int'))
            else:
                raise ValueError('only support nii.gz, mgz or npz files')


def upsample_labels(labels_dir, target_res, result_folder, path_label_list, path_freesurfer='/usr/local/freesurfer/'):

    # prepare result folder
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    assert len(labels_paths) > 0, "Could not find any training data"

    # load label list and corresponding LUT to make sure that labels go from 0 to N-1
    label_list = np.load(path_label_list)
    n_labels = label_list.shape[0]
    new_label_list = np.arange(n_labels)
    lut = np.zeros(np.max(label_list).astype('int') + 1)
    for n in range(n_labels):
        lut[label_list[n].astype('int')] = n

    # get info
    labels_shape, aff, n_dims, _, h, _ = get_image_info(labels_paths[0])
    if isinstance(target_res, (int, float)):
        target_res = [target_res]*n_dims

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin')
    post_cmd = ' -voxsize ' + ' '.join([str(r) for r in target_res]) + ' -rt interpolate -odt float'

    for i, path in enumerate(labels_paths):
        print('processing image {0}/{1}'.format(i+1, len(labels_paths)))

        # load volume
        volume, aff, h = load_volfile(path, im_only=False)
        volume = lut[volume.astype('int')]

        # create individual label maps
        basefilename = os.path.basename(path).replace('.nii.gz', '').replace('.mgz', '').replace('.npz', '')
        indiv_label = os.path.join(result_folder, basefilename)
        upsample_indiv_label = os.path.join(result_folder, basefilename + '_upsampled')
        if not os.path.exists(os.path.join(indiv_label)):
            os.mkdir(indiv_label)
        if not os.path.exists(os.path.join(upsample_indiv_label)):
            os.mkdir(upsample_indiv_label)
        for label in new_label_list:
            path_mask = os.path.join(indiv_label, str(label)+'.nii.gz')
            path_mask_upsampled = os.path.join(upsample_indiv_label, str(label)+'.nii.gz')
            if not os.path.isfile(path_mask):
                mask = (volume == label) * 1.0
                save_volfile(mask, aff, h, path_mask)
            if not os.path.isfile(path_mask_upsampled):
                cmd = mri_convert + ' ' + path_mask + ' ' + path_mask_upsampled + post_cmd
                _ = os.system(cmd)

        # take argmax of upsampled probability maps
        probmax, aff, h = load_volfile(os.path.join(upsample_indiv_label, '0.nii.gz'), im_only=False)
        volume = np.zeros(probmax.shape, dtype='int')
        for label in new_label_list:
            prob = load_volfile(os.path.join(upsample_indiv_label, str(label) + '.nii.gz'))
            idx = prob > probmax
            volume[idx] = label
            probmax[idx] = prob[idx]
        save_volfile(label_list[volume], aff, h, os.path.join(result_folder, os.path.basename(path)))


def smooth_labels(labels_dir, result_folder, path_label_list, recompute=False):

    # create result folder
    if not os.path.exists(os.path.join(result_folder)):
        os.mkdir(result_folder)

    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    labels_shape, aff, n_dims, _, h, _ = get_image_info(labels_paths[0])
    label_list = np.load(path_label_list)

    kernel = np.ones(tuple([3] * n_dims))
    for i, path in enumerate(labels_paths):
        print('\nprocessing label map {0}/{1}'.format(i+1, len(labels_paths)))
        result_file = os.path.join(result_folder, os.path.basename(path))
        if (not os.path.isfile(result_file)) | recompute:
            volume, aff, h = load_volfile(path, im_only=False)
            count = np.zeros(labels_shape)
            new_volume = np.zeros(labels_shape, dtype='int')
            for la, label in enumerate(label_list):
                print_loop_info(la, len(label_list), 10)
                mask = (volume == label) * 1
                n_neighbours = convolve(mask, kernel)
                idx = n_neighbours > count
                count[idx] = n_neighbours[idx]
                new_volume[idx] = label
            save_volfile(new_volume, aff, h, result_file)


def equalise_dataset_size_by_padding(labels_dir, results_folder, padding_value=0, max_shape=None):

    # create results dir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    assert len(labels_paths) > 0, "Could not find any training data"
    assert len(labels_paths) > 1, "Cannot equalise size of only one file"

    # get maximum shape
    if max_shape is None:
        max_shape, vox2ras, n_dims, n_channels, header, _ = get_image_info(os.path.join(labels_dir, labels_paths[0]))
        for path in labels_paths[1:]:
            labels_shape, vox2ras, n_dims, n_channels, header, _ = get_image_info(os.path.join(labels_dir, path))
            max_shape = tuple(np.maximum(np.asarray(max_shape), np.asarray(labels_shape)))
        max_shape = np.asarray(max_shape)

    # pad all labels to max_shape
    for i, path in enumerate(labels_paths):
        print_loop_info(i, len(labels_paths), 5)
        # load labels
        padded_labels = padding_value * np.ones(max_shape)
        labels, labels_shape, vox2ras, _, _, header, _ = get_image_info(os.path.join(labels_dir, path),
                                                                        return_image=True)
        labels_shape = np.asarray(labels_shape)
        # pad image
        min_coor = np.round((max_shape - labels_shape) / 2).astype('int')
        max_coor = (min_coor + labels_shape).astype('int')
        padded_labels[min_coor[0]:max_coor[0], min_coor[1]:max_coor[1], min_coor[2]:max_coor[2]] = labels
        padded_labels = padded_labels.astype('int')
        # update vox2ras matrix
        vox2ras[:-1, -1] = vox2ras[:-1, -1] - vox2ras[:-1, :-1] @ min_coor
        # save new file
        save_volfile(padded_labels, vox2ras, header, os.path.join(results_folder, os.path.basename(path)))


def crop_dataset(labels_dir, results_folder, image_dir=None, image_results_folder=None, margin=5, final_cropping=None):
    """crop all labels to the minimum possible size, with a margin. This assumes all the label maps have the same size.
    If images are provided, they are cropped in the sam fashion as their corresponding label maps.
    If the label map contains extra-cerebral labels, use a small margin, if not use a bigger one."""

    # create results dir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    if image_dir is not None:
        images_paths = list_images_in_folder(image_dir)
        assert image_results_folder is not None, 'im_results_folder should be specified as well'
        if not os.path.exists(image_results_folder):
            os.mkdir(image_results_folder)
    else:
        images_paths = [None] * len(labels_paths)
    labels_shape, _, n_dims, _, _, _ = get_image_info(labels_paths[0])

    # get cropping indices
    if final_cropping is None:
        min_cropping = np.array(labels_shape, dtype='int')
        max_cropping = np.zeros(n_dims, dtype='int')
        # get maximum cropping possible
        print('getting final cropping indices')
        for i, path in enumerate(labels_paths):
            print_loop_info(i, len(labels_paths), 5)
            volume = load_volfile(path)
            _, cropping = crop_volume(volume, margin=margin)
            min_cropping = np.minimum(min_cropping, cropping[:n_dims], dtype='int')
            max_cropping = np.maximum(max_cropping, cropping[n_dims:], dtype='int')
        final_cropping = np.concatenate([min_cropping, max_cropping])

    # crop label maps (and images)
    print('\ncropping images')
    for i, (lab_path, im_path) in enumerate(zip(labels_paths, images_paths)):
        print_loop_info(i, len(labels_paths), 5)
        volume, aff, h = load_volfile(lab_path, im_only=False)
        volume, aff = crop_array_with_idx(volume, final_cropping, n_dims, vox2ras=aff)
        save_volfile(volume, aff, h, os.path.join(results_folder, os.path.basename(lab_path)))
        if im_path is not None:
            volume, aff, h = load_volfile(im_path, im_only=False)
            volume, aff = crop_array_with_idx(volume, final_cropping, n_dims, vox2ras=aff)
            save_volfile(volume, aff, h, os.path.join(image_results_folder, os.path.basename(im_path)))

    return final_cropping


def mask_dataset_with_labels(images_dir, labels_folder, results_folder):
    # create results dir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    # prepare data files
    images_paths = list_images_in_folder(images_dir)
    labels_paths = list_images_in_folder(labels_folder)
    for i, (im_path, lab_path) in enumerate(zip(images_paths, labels_paths)):
        print_loop_info(i, len(labels_paths), 5)
        im, aff, h = load_volfile(im_path, im_only=False)
        lab = load_volfile(lab_path)
        im[lab == 0] = 0
        save_volfile(im, aff, h, os.path.join(results_folder, os.path.basename(im_path)))


def subdivide_dataset(patch_shape, labels_folder, label_results_folder, no_full_back=False):

    if not os.path.exists(label_results_folder):
        os.mkdir(label_results_folder)

    lab_paths = sorted(os.listdir(labels_folder))
    n_dims = len(patch_shape)

    for im_idx, la_path in enumerate(lab_paths):

        print('processing labels {} ({}/{})'.format(la_path, im_idx+1, len(lab_paths)))

        # load labels
        la, voxla, hla = load_volfile(os.path.join(labels_folder, la_path), im_only=False, squeeze=False)
        # first crop to size divisible by patch_shape
        la_shape = np.array(la.shape)
        new_size = np.array([find_closest_number_divisible_by_m(la_shape[i], patch_shape[i]) for i in range(n_dims)])
        crop = np.round((la_shape - new_size) / 2).astype('int')
        crop = np.concatenate((crop, crop + new_size), axis=0)
        if n_dims == 2:
            la = la[crop[0]:crop[2], crop[1]:crop[3]]
        elif n_dims == 3:
            la = la[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5]]
        else:
            raise Exception('cannot segment labels of size greater than 3')
        # split labels in several pieces
        n_la = 0
        n_crop = (new_size / patch_shape).astype('int')
        for i in range(n_crop[0]):
            i *= patch_shape[0]
            for j in range(n_crop[1]):
                j *= patch_shape[1]
                new_filename = la_path.replace('.nii.gz', '_%d.nii.gz' % n_la)
                new_filename = new_filename.replace('.mgz', '_%d.nii.gz' % n_la).replace('.npz', '_%d.nii.gz' % n_la)
                if n_dims == 2:
                    temp_la = la[i:i+patch_shape[0], j:j+patch_shape[0]]
                    labels = np.unique(temp_la.astype('int'))
                    if (not np.array_equal(labels, np.zeros(1, dtype='int32'))) & no_full_back:
                        n_la += 1
                        save_volfile(temp_la, voxla, hla, os.path.join(label_results_folder, new_filename))
                    elif not no_full_back:
                        n_la += 1
                        save_volfile(temp_la, voxla, hla, os.path.join(label_results_folder, new_filename))
                if n_dims == 3:
                    for k in range(n_crop[2]):
                        k *= patch_shape[2]
                        temp_la = la[i:i + patch_shape[0], j:j + patch_shape[0], k:k+patch_shape[2]]
                        labels = np.unique(temp_la.astype('int'))
                        if (not np.array_equal(labels, np.zeros(1, dtype='int32'))) & no_full_back:
                            n_la += 1
                            save_volfile(temp_la, voxla, hla, os.path.join(label_results_folder, new_filename))
                        elif not no_full_back:
                            n_la += 1
                            save_volfile(temp_la, voxla, hla, os.path.join(label_results_folder, new_filename))


def crop_labels_dataset(labels_dir, results_dir):
    # create results dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # prepare data files
    labels_paths = list_images_in_folder(labels_dir)
    labels_shape, _, n_dims, _, _, _ = get_image_info(labels_paths[0])
    # get maximum cropping possible
    print('cropping images')
    max_shape = np.zeros(n_dims, dtype='int')
    for i, path in enumerate(labels_paths):
        print_loop_info(i, len(labels_paths), 5)
        volume, aff, h = load_volfile(path, im_only=False)
        volume, cropping, aff = crop_volume(volume, margin=5, vox2ras=aff)
        max_shape = np.maximum(max_shape, np.array(volume.shape), dtype='int')
        save_volfile(volume, aff, h, os.path.join(results_dir, os.path.basename(path)))
    print('\npadding images to maximum size: {}'.format(max_shape))
    equalise_dataset_size_by_padding(results_dir, results_dir, max_shape=max_shape)


def convert_labels_type(labels_dir, results_dir, dtype='int32'):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    list_files = sorted([file for file in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, file))])
    for file in list_files:
        path_file = os.path.join(labels_dir, file)
        path_result = os.path.join(results_dir, file)
        lab, aff, h = load_volfile(path_file, im_only=False)
        save_volfile(lab, aff, h, path_result, dtype=dtype)


# FILE HANDLING FUNCTIONS

def get_image_info(image_path, return_image=False):

    # read image
    if ('.nii.gz' in image_path) | ('.mgz' in image_path):
        im, vox2ras, header = load_volfile(image_path, im_only=False)
    elif '.npz' in image_path or '.npy' in image_path:
        im = np.load(image_path)
        if image_path.endswith('.npz'):
            im = im['vol']
        vox2ras = np.eye(4)
        header = nib.Nifti1Header()
    else:
        raise TypeError('only nii.gz, mgz, and npz files supported: got %s' % os.path.basename(image_path))

    # understand if image is rgb
    im_shape = list(im.shape)
    if im_shape[-1] == 3:
        im_shape = im_shape[:-1]
        n_channels = 3
    else:
        n_channels = 1

    # get dimensions (excluding rgb channels)
    n_dims = len(im_shape)

    # get labels res
    if '.nii.gz' in image_path:
        labels_res = np.array(header['pixdim'][1:n_dims + 1]).tolist()
    elif '.mgz' in image_path:
        labels_res = np.array(header['delta']).tolist()  # mgz image
    else:
        labels_res = [1.0] * n_dims

    # return info
    if return_image:
        return im, im_shape, vox2ras, n_dims, n_channels, header, labels_res
    else:
        return im_shape, vox2ras, n_dims, n_channels, header, labels_res


def get_list_labels(path_label_list=None, labels_folder=None, save_label_list=None, FS_sort=False):
    """
    This function reads or compute the label list necessary to use BrainGenerator.
    :param path_label_list : path of already computed label list
    :param labels_folder: path of folder containing label maps. Label list is the list of unique labels from those maps.
    :param save_label_list: path where to save computed label list
    :param FS_sort: sort labels according to the FreeSurfer classification
    :return: the label list (numpy vector) and the number of neutral labels (if FS_sort=True)
    """

    # load label list if previously computed
    if path_label_list is not None:
        if os.path.exists(path_label_list):
            print('Loading list of unique labels')
            label_list = np.load(path_label_list)
        else:
            raise Exception('{}: file does not exist'.format(path_label_list))

    # compute label list from all label files
    elif labels_folder is not None:
        print('Compiling list of unique labels')
        # prepare data files
        labels_paths = list_images_in_folder(labels_folder)
        assert len(labels_paths) > 0, "Could not find any training data"
        # go through all labels files and compute unique list of labels
        label_list = np.empty(0)
        n_labels = len(labels_paths)
        datasets = []
        for lab_idx, path in enumerate(labels_paths):
            print_loop_info(lab_idx, n_labels, 10)
            y = load_volfile(path, squeeze=True, dtype='int32')
            datasets.append(np.squeeze(y))
        label_list = np.unique(datasets)

    else:
        raise Exception('either load_label_list_file or labels_folder should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 165, 251, 252, 253, 254, 255, 258, 259,
                             331, 332, 333, 334, 335, 336, 337, 338, 339, 340]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (20100 < la < 20110):
                left.append(la)
            elif (39 < la < 72) | (20000 < la < 20010):
                right.append(la)
            else:
                raise Exception('label {} not in current our FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([neutral, left, right])
        n_neutral_labels = len(neutral)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, label_list)

    if FS_sort:
        return label_list, datasets, n_neutral_labels
    else:
        return label_list, datasets


def load_volfile(datafile, im_only=True, squeeze=True, dtype=None):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz', '.npy')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(datafile)
        image = np.asarray(x.dataobj)
        vox2ras = x.affine
        header = x.header
    else:
        image = np.load(datafile)
        if datafile.endswith('.npz'):
            image = image['vol']
        vox2ras = np.eye(4)
        header = nib.Nifti1Header()
    if dtype:
        image = image.astype(dtype=dtype)
    if squeeze:
        image = np.squeeze(image)
    return image if im_only else (image, vox2ras, header)


def save_volfile(image, affine, header, path, res=None, dtype=None, n_dims=3):
    if dtype is not None:
        image = image.astype(dtype=dtype)
    if '.npz' in path:
        np.savez_compressed(path, vol=image)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if affine is None:
            affine = np.eye(4)
        nifty = nib.Nifti1Image(image, affine, header)
        if res is not None:
            res = reformat_variable(res, n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def write_object(filepath, obj):
    with open(filepath, 'wb') as file:
        pickler = pickle.Pickler(file)
        pickler.dump(obj)


def read_object(filepath):
    with open(filepath, 'rb') as file:
        unpickler = pickle.Unpickler(file)
        return unpickler.load()


def write_model_summary(model, filepath='./model_summary.txt', line_length=150):
    with open(filepath, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=line_length)


def reformat_variable(var, n_dim, dtype=None):
    """This function takes a variable (int, float, list, tuple) and reformat it into a list of desired length (n_dim)
    and type (int, float, bool)."""
    if isinstance(var, (int, float)):
        var = [var] * n_dim
    elif isinstance(var, (list, tuple)):
        if len(var) == 1:
            var = var * n_dim
        elif len(var) != n_dim:
            raise ValueError('if var is a list/tuple, it should be of length 1 or {0}, had {1}'.format(n_dim, var))
    else:
        raise TypeError('var should be an int, float, tuple, or list; had {}'.format(type(var)))
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        else:
            raise ValueError('dtype should be "float", "int", or "bool"; had {}'.format(dtype))
    return var


def strip_extension(path):
    path = path.replace('.nii.gz', '')
    path = path.replace('.nii', '')
    path = path.replace('.mgz', '')
    path = path.replace('.npz', '')
    return path


def strip_seg_aseg(path):
    path = path.replace('_seg', '')
    path = path.replace('_seg_1', '')
    path = path.replace('_seg_2', '')
    path = path.replace('seg_1_', '')
    path = path.replace('seg_2_', '')
    path = path.replace('_aseg', '')
    path = path.replace('_aseg_1', '')
    path = path.replace('_aseg_2', '')
    path = path.replace('aseg_1_', '')
    path = path.replace('aseg_2_', '')
    return path


def list_images_in_folder(path):
    names = ('*.nii.gz', '*.nii', '*.mgz', '*.npy', '*.npz')
    if os.path.isdir(path):
        out = sum((glob.glob(os.path.join(path, n)) for n in names), [])
    else:
        out = glob.glob(path)
    return sorted(out)


def list_models_in_folder(folder):
    list_models = sorted(glob.glob(os.path.join(folder, 'dice*.h5')))
    return list_models


def convert_images_in_folder_to_nifty(folder):
    image_paths = list_images_in_folder(folder)
    for image_path in image_paths:
        im, aff, h = load_volfile(image_path, im_only=False)
        save_volfile(im, aff, h, strip_extension(image_path) + '.nii.gz')


def print_loop_info(idx, n_iterations, spacing):
    if idx == 0:
        print('processing {}/{}'.format(1, n_iterations))
    elif idx % spacing == spacing-1:
        print('processing {}/{}'.format(idx + 1, n_iterations))


if __name__ == '__main__':

    # labels = '/home/benjamin/data/mit/data_for_stats_estimation/fsm_aseg_corrected'
    # images = '/home/benjamin/data/mit/data_for_stats_estimation/fsm_T2w'
    # check_images_and_labels(images, labels)
    # check_labels(labels)

    # image_folder = '/home/benjamin/data/mit/data_for_stats_estimation/im_T2'
    # lab_folder = '/home/benjamin/data/mit/data_for_stats_estimation/labels'
    # list_labels = '/home/benjamin/PycharmProjects/brain_generator/labels_and_classes/fsm_labels.npy'
    # list_classes = '/home/benjamin/PycharmProjects/brain_generator/labels_and_classes/fsm_classes.npy'
    # folder_images_rescaled = '/home/benjamin/data/mit/data_for_stats_estimation/im_T2_rescaled'
    # stats_dir = '/home/benjamin/data/mit/data_for_stats_estimation'
    #
    # rescale_images_in_folder(image_folder, folder_images_rescaled, recompute=False)
    # mean, std_dev = get_intensity_stats_dataset(folder_images_rescaled,
    #                                             lab_folder,
    #                                             list_labels,
    #                                             path_classes_list=list_classes,
    #                                             result_dir=stats_dir,
    #                                             prefix='t2')

    # # preprocess labels
    # dir_labels = '/home/benjamin/data/T1mix/testing/asegs_full_size'
    # results_correction = '/home/benjamin/data/T1mix/testing/asegs_full_size_ccorrected'
    # results_smoothing = None  # '/home/benjamin/data/mit/testing/multiSpectral/labels_cropped_corrected_smoothed'
    # path_list_labels = './labels_and_classes/mit_segmentation_labels.npy'
    #
    # labels = get_list_labels(labels_folder='/home/benjamin/data/mit/data_for_stats_estimation/training_asegs_corrected',
    #                          save_label_list='./labels_and_classes/T1mix_ss_intensity_estimation_labels.npy',
    #                          FS_sort=True)
    # print(labels)
    #
    # incorrect = [24, 29, 30, 62, 72, 77, 80, 85, 251, 252, 253, 254, 255]
    # correct = [0, 2, 0, 41, [2, 41], [2, 41], [3, 42], 0, [2, 41], [2, 41], [2, 41], [2, 41], [2, 41]]
    # correct_labels(dir_labels, incorrect, correct, results_correction)
    #
    # labels = get_list_labels(labels_folder='/home/benjamin/data/mit/result_samseg/fsm/origs', save_label_list=None, FS_sort=True)
    # print(labels)

    # smooth_labels(results_correction, results_smoothing, path_list_labels, recompute=True)

    labels = '/home/benjamin/data/T1mix/testing/asegs_corrected'
    images = '/home/benjamin/data/T1mix/testing/origs'
    result = '/home/benjamin/data/T1mix/testing/origs_cerebral'
    # image_results = '/home/benjamin/data/mit/testing/fsm/dbs_cropped'
    # margin = 20
    # final_cropping = equalise_dataset_size_by_cropping(labels, result, image_dir=images, image_results_folder=image_results, margin=margin)
    # print(final_cropping)

    # mask_dataset_with_labels(images, labels, result)
    convert_images_in_folder_to_nifty('/home/benjamin/PycharmProjects/brain_generator/models/t1_origs/test120/T1mix_ex')
