import torch
import torch.nn.functional as F
import numpy as np
import math

# Source:
# https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
# Version:
# 20250113, 13 Jan 2025

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
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt = torch.ones([5, 1, *win]).to(y_true.device)

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
        
        all_five = torch.cat((Ii, Ji, I2, J2, IJ),dim=1)
        all_five_conv = conv_fn(all_five, sum_filt, stride=stride, padding=padding, groups=5)
        I_sum, J_sum, I2_sum, J2_sum, IJ_sum = torch.split(all_five_conv, 1, dim=1)
        
        # I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        # J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        # I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        # J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        # IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # compute cross correlation
        # win_size = np.prod(win)
        # u_I = I_sum / win_size
        # u_J = J_sum / win_size

        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size


        # compute cross correlation
        win_size = np.prod(self.win)

        cross = IJ_sum - J_sum/win_size*I_sum
        I_var = I2_sum - I_sum/win_size*I_sum
        J_var = J2_sum - J_sum/win_size*J_sum

        
        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
