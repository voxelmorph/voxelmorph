import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
from torch import Tensor
from sklearn.metrics.cluster import normalized_mutual_info_score
from skimage.metrics import normalized_mutual_information
import SimpleITK as sitk
from torch.nn.modules.loss import _Loss

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

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
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

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


class MutualInformation_v2:
    """https://matthew-brett.github.io/teaching/mutual_information.html
    """
    def __init__(self, nb_bins=20) -> None:
        self.bins = nb_bins
    
    def mutual_information(self, hgram):
        """ Mutual information for joint histogram"""
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def loss(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        hist_2d, x_edges, y_edges = np.histogram2d(y_true.ravel(), y_pred.ravel(), bins=self.bins)
        MI = self.mutual_information(hist_2d)
        # print(f"Mona debug - the mutual information is {loss}")
        # hist_2d_test, x_edges, y_edges = np.histogram2d(y_true.ravel(), y_true.ravel(), bins=self.bins)
        # loss_test = self.mutual_information(hist_2d_test)
        # # print(f"The mutual information between same image is {loss_test} and registered image is {loss}")

        return 1.0 / MI

class MutualInformation_v3:
    """https://scikit-image.org/docs/stable/api/skimage.metrics.html"""
    def __init__(self) -> None:
        pass
    
    def loss(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        batch = y_true.shape[0]
        loss = 0
        for i in range(batch):
            nmi = normalized_mutual_information(np.squeeze(y_true[i,:,:,:]), np.squeeze(y_pred[i, :, :, :])) - 1
            loss += nmi

        return - loss / batch


class MutualInformation_v4:
    """https://simpleitk.readthedocs.io/en/v1.1.0/Documentation/docs/source/registrationOverview.html"""
    def __init__(self) -> None:
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(1)
        registration_method.SetInterpolator(sitk.sitkLinear)
        self.register = registration_method
    
    def loss(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        # print(f"y_true shape is {y_true.shape}")
        batch = y_true.shape[0]
        loss = 0
        for i in range(batch):
            fixed_image = sitk.GetImageFromArray(np.squeeze(y_true[i, :, :, :]), isVector=False)
            moving_image = sitk.GetImageFromArray(np.squeeze(y_pred[i, :, :, :]), isVector=False)

            
            transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
            transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
            initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform(2,sitk.sitkIdentity)))
            self.register.SetInitialTransform(initial_transform)

            loss1 = self.register.MetricEvaluate(fixed_image, moving_image)
            loss += loss1

        return np.float32(loss/batch)


def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
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

        return -nmi_gauss(
            fixed, warped, bins_fixed, bins_warped, sigma=self.sigma
        ).mean()

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

        return -nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )


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