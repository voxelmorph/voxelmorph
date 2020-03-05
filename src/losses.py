"""
losses for VoxelMorph
"""

# main imports
import sys

# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np



def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims+1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)


class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.get_shape()) - 2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well


    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))


###############################################################################
# Experimental losses
###############################################################################


class SparseVM(object):
    '''
    SparseVM Sparse Normalized Local Cross Correlation (SLCC)
    '''
    def __init__(self, mask):
        self.mask = mask

    def conv_block(self,data, mask, conv_layer, mask_conv_layer, core_name):
        '''
        data is the data tensor
        mask is a binary tensor the same size as data

        steps:
        - set empty voxels in data using data *= mask
        - conv data and mask with the conv conv_layer
        - re-weight data
        - binarize mask
        '''

        # mask.dtype
        # data.dtype
        # make sure the data is sparse according to the mask
        wt_data = keras.layers.Lambda(lambda x: x[0] * x[1], name='%s_pre_wmult' % core_name)([data, mask])
        # convolve data
        conv_data = conv_layer(wt_data)  
    
        # convolve mask
        conv_mask = mask_conv_layer(mask)
        zero_mask = keras.layers.Lambda(lambda x:x*0+1)(mask)
        conv_mask_allones = mask_conv_layer(zero_mask) # all_ones mask to get the edge counts right.
        mask_conv_layer.trainable = False
        o = np.ones(mask_conv_layer.get_weights()[0].shape)
        mask_conv_layer.set_weights([o])
    
        # re-weight data (this is what makes the conv makes sense)
        data_norm = lambda x: x[0] / (x[1] + 1e-2)
        # data_norm = lambda x: x[0] / K.maximum(x[1]/x[2], 1)
        out_data = keras.layers.Lambda(data_norm, name='%s_norm_im' % core_name)([conv_data, conv_mask])
        mask_norm = lambda x: tf.cast(x > 0, tf.float32)
        out_mask = keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)

        return (out_data, out_mask, conv_data, conv_mask)


         
    def sparse_conv_cc3D(self, atlas_mask, conv_size = 13, sum_filter = 1, padding = 'same', activation = 'elu'):
        '''
        Sparse Normalized Local Cross Correlation (SLCC) for 3D images
        '''
        def loss(I, J):
            # pass in mask to class: e.g. Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask),
            mask = self.mask
            # need the next two lines to specify channel for source image (otherwise won't compile)
            I = I[:,:,:,:,0]
            I = tf.expand_dims(I, -1)
             
            I2 = I*I
            J2 = J*J
            IJ = I*J
            input_shape = I.shape
            # want the size without the channel and batch dimensions
            ndims = len(input_shape) -2
            strides = [1] * ndims
            convL = getattr(KL, 'Conv%dD' % ndims)
            im_conv = convL(sum_filter, conv_size, padding=padding, strides=strides,kernel_initializer=keras.initializers.Ones())
            im_conv.trainable = False
            mask_conv = convL(1, conv_size, padding=padding, use_bias=False, strides=strides,kernel_initializer=keras.initializers.Ones())
            mask_conv.trainable = False

            combined_mask = mask*atlas_mask
            u_I, out_mask_I, not_used, conv_mask_I = self.conv_block(I, mask, im_conv, mask_conv, 'u_I')
            u_J, out_mask_J, not_used, conv_mask_J = self.conv_block(J, atlas_mask, im_conv, mask_conv, 'u_J')
            not_used, not_used_mask, I_sum, conv_mask = self.conv_block(I, combined_mask, im_conv, mask_conv, 'I_sum')
            not_used, not_used_mask, J_sum, conv_mask = self.conv_block(J, combined_mask, im_conv, mask_conv, 'J_sum')
            not_used, not_used_mask, I2_sum, conv_mask = self.conv_block(I2, combined_mask, im_conv, mask_conv, 'I2_sum')
            not_used, not_used_mask, J2_sum, conv_mask = self.conv_block(J2, combined_mask, im_conv, mask_conv, 'J2_sum')
            not_used, not_used_mask, IJ_sum, conv_mask = self.conv_block(IJ, combined_mask, im_conv, mask_conv, 'IJ_sum')
    
            cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*conv_mask
            I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*conv_mask
            J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*conv_mask
            cc = cross*cross / (I_var*J_var + 1e-2) 
            return -1.0 * tf.reduce_mean(cc)
        return loss

def mutualInformation(bin_centers,
                      sigma_ratio=0.5,    # sigma for soft MI. If not provided, it will be half of a bin length
                      max_clip=1,
                      crop_background=False, # crop_background should never be true if local_mi is True
                      local_mi=False,
                      patch_size=1):
    """
    mutual information for image-image pairs.

    Author: Courtney Guo. See thesis https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)
    
    if local_mi:
        return localMutualInformation(bin_centers, sigma_ratio, max_clip, patch_size)

    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)

def globalMutualInformation(bin_centers,
                      sigma_ratio=0.5,
                      max_clip=1,
                      crop_background=False):
    """
    Mutual Information for image-image pairs

    Building from neuron.losses.MutualInformationSegmentation()    

    This function assumes that y_true and y_pred are both (batch_size x height x width x depth x nb_chanels)

    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)
        
        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss

def localMutualInformation(bin_centers,
                      vol_size,
                      sigma_ratio=0.5,
                      max_clip=1,
                      patch_size=1):
    """
    Local Mutual Information for image-image pairs
    # vol_size is something like (160, 192, 224)  

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)

    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, num_bins]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute padding sizes
        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- preterm * K.square(tf.pad(y_true, padding, 'CONSTANT')  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT')  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, num_bins])

        I_b_patch = tf.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = tf.reduce_mean(I_a_patch, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.mean(K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss

