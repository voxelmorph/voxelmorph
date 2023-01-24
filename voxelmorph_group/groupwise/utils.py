from __future__ import absolute_import

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import omegaconf
from sklearn.decomposition import PCA
from .t1map import MOLLIT1mapParallel

def save2onxx(model, x, in_name, out_name, out_path):
    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    out_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    input_names = [in_name],   # the model's input names
                    output_names = [out_name], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param




# NOTE: Gaussian kernel
def _gauss_1d(x, mu, sigma):
    return 1./(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def gauss_kernel_1d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = np.arange(-lw, lw+1)
    kernel_1d = _gauss_1d(x, 0., sigma)
    return kernel_1d / kernel_1d.sum()

def gauss_kernel_2d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = np.arange(-lw, lw+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kernel_2d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma)
    return kernel_2d / kernel_2d.sum()

def gauss_kernel_3d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = z = np.arange(-lw, lw+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kernel_3d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma) \
              * _gauss_1d(Z, 0., sigma)
    return kernel_3d / kernel_3d.sum()


# NOTE: Average kernel
def _average_kernel_nd(ndim, kernel_size):

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    kernel_nd = np.ones(kernel_size)
    kernel_nd /= np.sum(kernel_nd)

    return kernel_nd

def average_kernel_1d(kernel_size):
    return _average_kernel_nd(1, kernel_size)

def average_kernel_2d(kernel_size):
    return _average_kernel_nd(2, kernel_size)

def average_kernel_3d(kernel_size):
    return _average_kernel_nd(3, kernel_size)


# NOTE: Gradient kernel
def gradient_kernel_1d(method='default'):

    if method == 'default':
        kernel_1d = np.array([-1,0,+1])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_1d

def gradient_kernel_2d(method='default', axis=0):

    if method == 'default':
        kernel_2d = np.array([[0,-1,0],
                              [0,0,0],
                              [0,+1,0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1,-2,-1],
                              [0,0,0],
                              [+1,+2,+1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1,-1,-1],
                              [0,0,0],
                              [+1,+1,+1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1,-np.sqrt(2),-1],
                              [0,0,0],
                              [+1,+np.sqrt(2),+1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_2d, 0, axis)

def gradient_kernel_3d(method='default', axis=0):

    if method == 'default':
        kernel_3d = np.array([[[0, 0, 0],
                               [0, -1, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, +1, 0],
                               [0, 0, 0]]])
    elif method == 'sobel':
        kernel_3d = np.array([[[-1, -3, -1],
                               [-3, -6, -3],
                               [-1, -3, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +3, +1],
                               [+3, +6, +3],
                               [+1, +3, +1]]])
    elif method == 'prewitt':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -1, -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +1, +1],
                               [+1, +1, +1]]])
    elif method == 'isotropic':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -np.sqrt(2), -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +np.sqrt(2), +1],
                               [+1, +1, +1]]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_3d, 0, axis)


_func_conv_nd_table = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}

def spatial_filter_nd(x, kernel, mode='replicate'):
    """ N-dimensional spatial filter with padding.
    Args:
        x (~torch.Tensor): Input tensor.
        kernel (~torch.Tensor): Weight tensor (e.g., Gaussain kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.
    Returns:
        ~torch.Tensor: Output tensor
    """

    n_dim = x.dim() - 2
    conv = _func_conv_nd_table[n_dim]

    pad = [None,None]*n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k//2 for k in pad]

    return conv(F.pad(x, pad=pad, mode=mode), kernel)


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


def update_atlas(invols, method='avg', tvec=None):
    """
    Generate implicit template from input volumes.

    Args:
        invols (_type_): (N, C, H, W)
        method (str, optional): Methods to generate implicit template. Defaults to 'avg'.

    Raises:
        ValueError: input volume should be 4D

    Returns:
        atlas: implicit template (1, 1, H, W)
    """
    print(f"Mona - invols shape {invols.shape} and method {method} and tvec {tvec}")
    if len(invols.shape) == 4:
        if method == 'avg':
            atlas = torch.mean(invols, axis=0, keepdims=True)
        elif method == 'pca':
            tmp = torch.transpose(torch.squeeze(invols), (1, 2, 0))

            x, y, z = tmp.shape
            M = tmp.reshape(x*y, z)
            
            pca = PCA(n_components=1, svd_solver='full')
            img_pca = pca.fit_transform(M)
            atlas = img_pca.reshape((x, y))
            atlas = atlas[None, None, :, :]
        elif method == 't1map':
            t1map = MOLLIT1mapParallel()
            print(f"Mona - {invols.shape}")
            inversion_img, pmap, sdmap, null_index, S = t1map.mestimation_abs(np.array(tvec), torch.squeeze(invols).permute(1, 2, 0).detach().cpu().numpy())
            atlas = S[None, None, :, :]
        return atlas
    else:
        raise ValueError("Input volume should be 4D")