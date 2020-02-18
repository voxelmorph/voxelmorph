import os
import numpy as np
import scipy
import csv
import functools


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def load_volfile(filename, np_var='vol', add_batch_axis=False, add_feat_axis=False, pad_shape=None, resize_factor=1, ret_affine=False):
    """
    Loads a file in nii, nii.gz, mgz, or npz format.

    Parameters:
        filename: Filename to load.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data()
        affine = img.affine
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1,  0,  0,  0],
                               [ 0,  0,  1,  0],
                               [ 0, -1,  0,  0],
                               [ 0,  0,  0,  1]], dtype=float)
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    """
    if factor == 1:
        return array
    else:
        dim_factors = [factor for _ in array.shape[:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_matrix_to_vec(matrix):
    """
    Converts an affine matrix to a transform vector (of len 12).
    """
    trf = matrix - np.eye(4)
    trf = trf[:3,:]
    trf = trf.reshape([1, 12])
    return trf


def affine_vec_to_matrix(trf):
    """
    Converts an affine transform vector (of len 12) to a matrix.
    """
    return np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)


def affine_merge(transforms, resize):
    """
    Merges a set of affine transforms and scales the matrix to account for volume resizing.
    """
    matrices = [affine_vec_to_matrix(trf) for trf in transforms]
    matrix = functools.reduce(np.matmul, matrices)
    matrix[:, -1] *= (1 / resize)
    return affine_matrix_to_vec(matrix)
