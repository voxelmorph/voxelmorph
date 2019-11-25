import os
import numpy as np
import collections
import scipy
import yaml
import inspect
import functools


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def load_volfile(filename, np_var='vol_data', add_axes=False, pad_shape=None, zoom=1):
    """
    Loads a file in nii, nii.gz, mgz, or npz format.

    Parameters:
        filename: Filename to load.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol_data'.
        add_axes: Adds an axis to the beginning and end of the array. Default is False.
        pad_shape: Zero-pad loaded volume to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
    """
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        vol = nib.load(filename).get_data()
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if zoom != 1:
        vol = resize(vol, zoom)

    if add_axes:
        vol = vol[np.newaxis, ..., np.newaxis]

    return vol


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape).astype(array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor):
    """
    Resizes an array by a given factor. 
    """
    return array if factor == 1 else scipy.ndimage.interpolation.zoom(array, [factor] * array.ndim, order=0)


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


def matrix_to_transform(matrix):
    """
    Converts an affine matrix to a transform vector (of len 12).
    """
    trf = matrix - np.eye(4)
    trf = trf[:3,:]
    trf = trf.reshape([1, 12])
    return trf


def transform_to_matrix(trf):
    """
    Converts an affine transform vector (of len 12) to a matrix.
    """
    return np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)


def merge_affines(transforms, resize):
    """
    Merges a set of affine transforms and scales the matrix to account for volume resizing.
    """
    matrices = [transform_to_matrix(trf) for trf in transforms]
    matrix = functools.reduce(np.matmul, matrices)
    matrix[:, -1] *= (1 / resize)
    return matrix_to_transform(matrix)


class NetConfig(collections.OrderedDict):
    """
    A specialized dictionary for managing network configuration parameters.

    Parameters:
        network: The function used to build the model.
        kwargs: Parameters to be forwarded to the network builder function.
    """

    def __init__(self, network, **kwargs):
        self.network = network
        self.update(kwargs)

    def build_model(self, weights=None):
        """
        Constructs a model from a configuration. Weights can be loaded into the
        model if a filename is provided.
        """
        # remove any arguments that don't exist in the function definition
        parameters = { k: self[k] for k in inspect.getfullargspec(self.network).args if k in self }
        model = self.network(**parameters)

        # load weights
        if weights:
            if get_backend() == 'pytorch':
                import torch
                model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))
            else:
                if isinstance(model, (list, tuple)):
                    model[0].load_weights(weights)
                else:
                    model.load_weights(weights)

        return model

    @classmethod
    def read(cls, filename):
        """
        Loads network parameters from a yaml configuration file.
        """
        with open(filename, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        from . import networks
        net = getattr(networks, config.pop('network'))
        if not net:
            raise ValueError('No network named "%s" found in voxelmorph.networks' % self.network)

        return cls(net, **config)

    def write(self, filename):
        """
        Saves network parameters to a yaml configuration file.
        """
        with open(filename, 'w') as file:
            file.write('network: %s\n' % self.network.__name__)
            for param, value in self.items():
                if isinstance(value, tuple):
                    value = list(value)
                file.write('%s: %s\n' % (param, value))

