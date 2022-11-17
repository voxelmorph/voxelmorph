from typing import Optional, Tuple
from torch import Tensor
from torch import nn as nn

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math

import SimpleITK as sitk
from torch.nn.modules.loss import _Loss
from .utils import *


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return [x, x]

def _grad_param(ndim, method, axis):

    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())

def _gauss_param(ndim, sigma, truncate):

    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, device="cuda"):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        vol_shape = y_pred.shape[1:-1]
        ndims = len(vol_shape)
        if ndims == 3:
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        elif ndims == 2:
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            d = torch.mean(dx) + torch.mean(dy)

        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class BendingEnergy2d:
    """Calculates bending energy penalty for a 2D coordinate grid."""

    def __init__(self):
        self.dvf_input = False
    
    def grad(
        self, _, coord_grid: Tensor, vector_dim: int = -1) -> Tensor:
        """Calculates bending energy penalty for a 2D coordinate grid.

        For further details regarding this regularization please read the work by `Rueckert 1999`_.

        Args:
            coord_grid: 2D coordinate grid, i.e. a 4D Tensor with standard dimensions
            (n_samples, 2, y, x).
            vector_dim: Specifies the location of the vector dimension. Default: -1
            dvf_input: If ``True``, coord_grid is assumed a displacement vector field and
            an identity_grid will be added. Default: ``False``

        Returns:
            Bending energy per instance in the batch.

        .. _Rueckert 1999: https://ieeexplore.ieee.org/document/796284

        """
        assert coord_grid.ndim == 4, "Input tensor should be 4D, i.e. 2D images."

        if vector_dim != 1:
            coord_grid = coord_grid.movedim(vector_dim, -1)

        if self.dvf_input:
            coord_grid = coord_grid + self.identity_grid(coord_grid.shape[2:], stackdim=0)

        d_y = torch.diff(coord_grid, dim=2)
        d_x = torch.diff(coord_grid, dim=3)

        d_yy = torch.diff(d_y, dim=2)[:, :, :, :-2]
        d_yx = torch.diff(d_y, dim=3)[:, :, :-1, :-1]
        d_xx = torch.diff(d_x, dim=3)[:, :, :-2, :]
        # tmp = torch.mean(d_yy ** 2 + d_xx ** 2 + 2 * d_yx ** 2, axis=(1, 2, 3))
        
        return torch.mean(d_yy ** 2 + d_xx ** 2 + 2 * d_yx ** 2)


    def identity_grid(shape, stackdim, dtype=torch.float32, device="cpu"):
        """Create an nd identity grid."""
        tensors = (torch.arange(s, dtype=dtype, device=device) for s in shape)
        return torch.stack(
            torch.meshgrid(*tensors)[::-1], stackdim
        )  # z,y,x shape and flip for x, y, z coords


class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: Optional[Tuple[float, float]] = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        """
        Compute the normalized mutual information metric.
        Args:
            fixed: Tensor of shape (batch, N, H, W), representing the fixed image.
            warped: Tensor of shape (batch, N, H, W), representing the warped image.
        
        """
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return - self.nmi_gauss(fixed, warped, bins_fixed, bins_warped, sigma=self.sigma).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -self.nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )
    
    def nmi_gauss(self, x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"

        def gaussian_window(x, bins, sigma):
            assert x.ndim == 2, "Input tensor should be 2-dimensional."
            return torch.exp(
                -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
            ) / (math.sqrt(2 * math.pi) * sigma)

        x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
        x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
        p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
        p_XY = p_XY + e  # deal with numerical instability

        p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]

        p_X = p_XY.sum(1)
        p_Y = p_XY.sum(2)

        I = (p_XY * torch.log(p_XY / (p_X[:, None] * p_Y[:, :, None]))).sum((1, 2))

        marg_ent_0 = (p_X * torch.log(p_X)).sum(1)
        marg_ent_1 = (p_Y * torch.log(p_Y)).sum(1)

        normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

        return normalized


    def nmi_gauss_mask(x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
        def gaussian_window_mask(x, bins, sigma):

            assert x.ndim == 1, "Input tensor should be 2-dimensional."
            return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
                math.sqrt(2 * math.pi) * sigma
            )

        x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
        x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
        p_XY = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
        p_XY = p_XY + e  # deal with numerical instability

        p_XY = p_XY / p_XY.sum()

        p_X = p_XY.sum(0)
        p_Y = p_XY.sum(1)

        I = (p_XY * torch.log(p_XY / (p_X[None] * p_Y[:, None]))).sum()

        marg_ent_0 = (p_X * torch.log(p_X)).sum()
        marg_ent_1 = (p_Y * torch.log(p_Y)).sum()

        normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

        return normalized


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self, config):
        super(MILossGaussian, self).__init__()

        self.vmin = config.vmin
        self.vmax = config.vmax
        self.sample_ratio = config.sample_ratio
        self.normalised = config.normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (self.vmax - self.vmin) / config.num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = config.num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)

class Jacobian:
    def __init__(self):
        pass
    
    def loss(self, displacement):
        dis = displacement.detach().cpu().numpy()
        folding_ratio, mag_det_jac_det = self.calculate_jacobian_metrics(dis)
        return folding_ratio, mag_det_jac_det
    

    def calculate_jacobian_metrics(self, disp):
        """Calculate Jacobian related regularity metrics.
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant

        Args:
            disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

        Returns:
           folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
        """
        folding_ratio = []
        mag_grad_jac_det = []
        # print(f'Mona: the shape of disp is {disp.shape}')
        for n in range(disp.shape[0]):
            # print(f"Mona: the shape of disp[n] is {disp[n, ...].shape}")
            disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
            jac_det_n = self.calculate_jacobian_det(disp_n)
            folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
            mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
        return np.mean(folding_ratio), np.mean(mag_grad_jac_det)
    

    def calculate_jacobian_det(self, disp):
        """Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

        Args:
            disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

        Returns:
            jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
        """

        disp_img = sitk.GetImageFromArray(disp, isVector=True)
        jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
        jac_det = sitk.GetArrayFromImage(jac_det_img)
        return jac_det


class GradientDifference2d(nn.Module):
    """ Two-dimensional gradient difference
    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean'):

        super(GradientDifference2d, self).__init__()

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self.return_map = return_map
        self.reduction = reduction

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(2, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(2, self.grad_method, axis=1)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(2, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(2, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))

        # absolute difference
        diff_u = torch.abs(x_grad_u - y_grad_u)
        diff_v = torch.abs(x_grad_v - y_grad_v)

        # reshape back
        diff_u = diff_u.view(b, c, *spatial_shape)
        diff_v = diff_v.view(b, c, *spatial_shape)

        diff_map = 0.5 * (diff_u + diff_v)

        if self.reduction == 'mean':
            diff = torch.mean(diff_map)
        elif self.reduction == 'sum':
            diff = torch.sum(diff_map)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if self.return_map:
            return diff, diff_map

        return diff


class GradientCorrelation2d(GradientDifference2d):
    """ Two-dimensional gradient correlation (GC)
    https://github.com/yuta-hi/pytorch_similarity.git

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean',
                 eps=1e-8):

        super().__init__(grad_method,
                        gauss_sigma,
                        gauss_truncate,
                        return_map,
                        reduction)

        self.eps = eps


    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))

        # gradient correlation
        gc_u, gc_map_u = normalized_cross_correlation(x_grad_u, y_grad_u, True, self.reduction, self.eps)
        gc_v, gc_map_v = normalized_cross_correlation(x_grad_v, y_grad_v, True, self.reduction, self.eps)

        # reshape back
        gc_map_u = gc_map_u.view(b, c, *spatial_shape)
        gc_map_v = gc_map_v.view(b, c, *spatial_shape)

        gc_map = 0.5 * (gc_map_u + gc_map_v)
        gc = 0.5 * (gc_u + gc_v)

        if not self.return_map:
            return gc

        return gc, gc_map