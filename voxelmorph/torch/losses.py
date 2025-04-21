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

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
    

class SoftNormalizedMutualInformation:
    """
    Soft Normalized Mutual Information
    Citation:
    Qiu, H., Qin, C., Schuh, A., Hammernik, K. &amp; Rueckert, D.. (2021).
      Learning Diffeomorphic and Modality-invariant Registration using B-splines. 
      <i>Proceedings of the Fourth Conference on Medical Imaging with Deep Learning</i>, 
      in <i>Proceedings of Machine Learning Research</i> 143:645-664 Available from https://proceedings.mlr.press/v143/qiu21a.html.
    """
    
    def __init__(self, num_bins=64, sigma=None):
        """
        Parameters:
            num_bins: number of bins, default 64; 256/bins should be integer.
            sigma: variance of gaussian distribution, default sigma ensures HMFW is one bin size.
        """
        assert 256 % num_bins == 0, "256/bins should be integer."
        
        self.num_bins = num_bins
        if(sigma == None):
            self.sigma = 256.0/num_bins/np.sqrt(2*np.log(2))
        else:
            self.sigma = sigma

    def gaussian(self, y):
        return 1/torch.sqrt(torch.tensor(2*torch.pi))/self.sigma*torch.exp(-y**2/2/self.sigma**2)
    
    def loss(self, y_true, y_pred):
        """
        Returns: -(H_f+H_m)/H_fm where H_f, H_m, H_fm are entropy for fixed image, moving image 
            and joint entropy for two images
        """
        batch_size = y_pred.shape[0]
        h = torch.zeros([batch_size, self.num_bins, self.num_bins]).to(y_true.device)
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                h[:,i,j] = torch.sum(self.gaussian(y_true - i)*self.gaussian(y_pred - j), dim=list(range(1,len(y_true.shape))))
        p = h / torch.sum(h, dim=[1,2])[:,None,None]
        p_f = torch.sum(p, dim = 2)
        p_m = torch.sum(p, dim = 1)

        eps = 1e-8
        H_fm = torch.sum(-p*torch.log(p+eps), dim = [1,2])
        H_f = torch.sum(-p_f*torch.log(p_f+eps), dim = 1)
        H_m = torch.sum(-p_m*torch.log(p_m+eps), dim = 1)
        return -torch.mean((H_f + H_m) / H_fm)