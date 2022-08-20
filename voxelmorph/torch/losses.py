import torch
import torch.nn.functional as F
import numpy as np
import math


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


class MutualInfo:
    """Compute the mutual information
    https://github.com/connorlee77/pytorch-mutual-information/blob/master/MutualInformation.py
    """
    def __init__(self, sigma=0.4, num_bins=256, normalize=True) -> None:
        self.sigma = 2 * sigma ** 2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = torch.linspace(0, 1, num_bins).float()

    def marignalPdf(self, input):
        residuals = input - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def jointPdf(self, kernel1, kernel2):

        joint_kernel = torch.matmul(kernel1.transpose(1, 2), kernel2)
        normalization = torch.sum(joint_kernel, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel / normalization

        return pdf  

    def loss(self, y_true, y_pred):
        B, C, H, W = y_pred.shape
        assert((y_pred.shape == y_true.shape))

        x1 = y_true.view(B, H*W, C)
        x2 = y_pred.view(B, H*W, C)

        pdf_x1, kernel1 = self.marignalPdf(x1)
        pdf_x2, kernel2 = self.marignalPdf(x2)

        pdf_x1x2 = self.jointPdf(kernel1, kernel2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
		
        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information

