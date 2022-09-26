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

# class MutualInformation_v1:
#     """
#     Soft Mutual Information approximation for intensity volumes

#     More information/citation:
#     - Courtney K Guo. 
#       Multi-modal image registration with unsupervised deep learning. 
#       PhD thesis, Massachusetts Institute of Technology, 2019.
#     - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
#       SynthMorph: learning contrast-invariant registration without acquired images
#       IEEETransactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
#       https ://doi.org/10.1109/TMI.2021.3116879

#     When the image is not registered, the signal is less concentrated into a small number of bins, the mutual information has dropped.
#     https://matthew-brett.github.io/teaching/mutual_information.html
#     """

#     def __init__(self, 
#                  bin_centers=None,
#                  nb_bins=None,
#                  soft_bin_alpha=None,
#                  min_clip=None,
#                  max_clip=None) -> None:

#         self.bin_centers = None
#         if bin_centers is not None:
#             self.bin_centers = tf.convert_to_tensor(bin_centers, dtype=tf.float32)
#             assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
#             nb_bins = bin_centers.shape[0]

#         self.nb_bins = nb_bins
#         if bin_centers is None and nb_bins is None:
#             self.nb_bins = 16

#         self.min_clip = min_clip
#         if self.min_clip is None:
#             self.min_clip = -np.inf

#         self.max_clip = max_clip
#         if self.max_clip is None:
#             self.max_clip = np.inf

#         self.soft_bin_alpha = soft_bin_alpha
#         if self.soft_bin_alpha is None:
#             sigma_ratio = 0.5
#             if self.bin_centers is None:
#                 sigma = sigma_ratio / (self.nb_bins - 1)
#             else:
#                 sigma = sigma_ratio * tf.reduce_mean(tf.experimental.numpy.diff(bin_centers))
#             self.soft_bin_alpha = 1 / (2 * tf.square(sigma))

#     def loss(self, x, y):
#         x = x.cpu().detach().numpy().transpose(0, 2, 3, 1)
#         y = y.cpu().detach().numpy().transpose(0, 2, 3, 1)
#         tensor_channels_x = K.shape(x)[-1]
#         tensor_channels_y = K.shape(y)[-1]
#         msg = 'volume_mi requires two single-channel volumes. See channelwise().'
#         tf.debugging.assert_equal(tensor_channels_x, 1, msg)
#         tf.debugging.assert_equal(tensor_channels_y, 1, msg)

#         # volume mi
#         loss = K.flatten(self.channelwise(x, y))
#         print(f"Mona - debug: the mutual information loss is {loss.numpy()}")
#         return 1.0 / loss.numpy()[0]

#     def channelwise(self, x, y):
#         """
#         Mutual information for each channel in x and y. Thus for each item and channel this 
#         returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to 
#         create a soft quantization (binning) of the intensities in each channel

#         Parameters:
#             x and y:  [bs, ..., C]

#         Returns:
#             Tensor of size [bs, C]
#         """
#         # check shapes
#         tensor_shape_x = K.shape(x)
#         tensor_shape_y = K.shape(y)
#         tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

#         # reshape to [bs, V, C]
#         if tensor_shape_x.shape[0] != 3:
#             new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
#             x = tf.reshape(x, new_shape)                             # [bs, V, C]
#             y = tf.reshape(y, new_shape)                             # [bs, V, C]

#         # move channels to first dimension
#         ndims_k = len(x.shape)
#         permute = [ndims_k - 1] + list(range(ndims_k - 1))
#         cx = tf.transpose(x, permute)                                # [C, bs, V]
#         cy = tf.transpose(y, permute)                                # [C, bs, V]

#         # soft quantize
#         cxq = self._soft_sim_map(cx)                                  # [C, bs, V, B]
#         cyq = self._soft_sim_map(cy)                                  # [C, bs, V, B]

#         # get mi
#         map_fn = lambda x: self.maps(*x)
#         cout = tf.map_fn(map_fn, [cxq, cyq], dtype=tf.float32)       # [C, bs]

#         # permute back
#         return tf.transpose(cout, [1, 0])                            # [bs, C]
 
#     def _soft_sim_map(self, x):
#         """
#         See neurite.utils.soft_quantize

#         Parameters:
#             x [bs, ...]: intensity image. 

#         Returns:
#             volume with one more dimension [bs, ..., B]
#         """
        
#         return ne.utils.soft_quantize(x,
#                                       alpha=self.soft_bin_alpha,
#                                       bin_centers=self.bin_centers,
#                                       nb_bins=self.nb_bins,
#                                       min_clip=self.min_clip,
#                                       max_clip=self.max_clip,
#                                       return_log=False)              # [bs, ..., B]


#     def maps(self, x, y):
#         """
#         Computes mutual information for each entry in batch, assuming each item contains 
#         probability or similarity maps *at each voxel*. These could be e.g. from a softmax output 
#         (e.g. when performing segmentaiton) or from soft_quantization of intensity image.

#         Note: the MI is computed separate for each itemin the batch, so the joint probabilities 
#         might be  different across inputs. In some cases, computing MI actoss the whole batch 
#         might be desireable (TODO).

#         Parameters:
#             x and y are probability maps of size [bs, ..., B], where B is the size of the 
#               discrete probability domain grid (e.g. bins/labels). B can be different for x and y.

#         Returns:
#             Tensor of size [bs]
#         """

#         # check shapes
#         tensor_shape_x = K.shape(x)
#         tensor_shape_y = K.shape(y)
#         tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
#         tf.debugging.assert_non_negative(x)
#         tf.debugging.assert_non_negative(y)

#         eps = K.epsilon()

#         # reshape to [bs, V, B]
#         if tensor_shape_x.shape[0] != 3:
#             new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
#             x = tf.reshape(x, new_shape)                             # [bs, V, B1]
#             y = tf.reshape(y, new_shape)                             # [bs, V, B2]

#         # joint probability for each batch entry
#         x_trans = tf.transpose(x, (0, 2, 1))                         # [bs, B1, V]
#         # pxy = K.batch_dot(x_trans, y)                                # [bs, B1, B2]
#         pxy = np.matmul(x_trans.numpy(), y.numpy())
#         pxy = pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)   # [bs, B1, B2]

#         # x probability for each batch entry
#         px = K.sum(x, 1, keepdims=True)                              # [bs, 1, B1]
#         px = px / (K.sum(px, 2, keepdims=True) + eps)                # [bs, 1, B1]

#         # y probability for each batch entry
#         py = K.sum(y, 1, keepdims=True)                              # [bs, 1, B2]
#         py = py / (K.sum(py, 2, keepdims=True) + eps)                # [bs, 1, B2]

#         # independent xy probability
#         px_trans = K.permute_dimensions(px, (0, 2, 1))               # [bs, B1, 1]
        
#         pxpy = np.matmul(px_trans.numpy(), py.numpy())
#         # pxpy = K.batch_dot(px_trans, py)                             # [bs, B1, B2]
#         pxpy_eps = pxpy + eps

#         # mutual information
#         log_term = K.log(pxy / pxpy_eps + eps)                       # [bs, B1, B2]
#         mi = K.sum(pxy * log_term, axis=[1, 2])                      # [bs]
#         return mi


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
