"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import sys

# third party
import numpy as np
import keras.backend as K
from keras import losses
import tensorflow as tf

# local
from . import utils

class CategoricalCrossentropy(object):
    """
    Categorical crossentropy with optional categorical weights and spatial prior

    Adapted from weighted categorical crossentropy via wassname:
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        loss = CategoricalCrossentropy().loss # or
        loss = CategoricalCrossentropy(weights=weights).loss # or
        loss = CategoricalCrossentropy(..., prior=prior).loss
        model.compile(loss=loss, optimizer='adam')
    """

    def __init__(self, weights=None, use_float16=False, vox_weights=None, crop_indices=None):
        """
        Parameters:
            vox_weights is either a numpy array the same size as y_true,
                or a string: 'y_true' or 'expy_true'
            crop_indices: indices to crop each element of the batch
                if each element is N-D (so y_true is N+1 dimensional)
                then crop_indices is a Tensor of crop ranges (indices)
                of size <= N-D. If it's < N-D, then it acts as a slice
                for the last few dimensions.
                See Also: tf.gather_nd
        """

        self.weights = weights if (weights is not None) else None
        self.use_float16 = use_float16
        self.vox_weights = vox_weights
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)

    def loss(self, y_true, y_pred):
        """ categorical crossentropy loss """

        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        if self.use_float16:
            y_true = K.cast(y_true, 'float16')
            y_pred = K.cast(y_pred, 'float16')

        # scale and clip probabilities
        # this should not be necessary for softmax output.
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute log probability
        log_post = K.log(y_pred)  # likelihood

        # loss
        loss = - y_true * log_post

        # weighted loss
        if self.weights is not None:
            loss *= self.weights

        if self.vox_weights is not None:
            loss *= self.vox_weights

        # take the total loss
        # loss = K.batch_flatten(loss)
        mloss = K.mean(K.sum(K.cast(loss, 'float32'), -1))
        tf.verify_tensor_all_finite(mloss, 'Loss not finite')
        return mloss


class Dice(object):
    """
    Dice of two Tensors.

    Tensors should either be:
    - probabilitic for each label
        i.e. [batch_size, *vol_size, nb_labels], where vol_size is the size of the volume (n-dims)
        e.g. for a 2D vol, y has 4 dimensions, where each entry is a prob for that voxel
    - max_label
        i.e. [batch_size, *vol_size], where vol_size is the size of the volume (n-dims).
        e.g. for a 2D vol, y has 3 dimensions, where each entry is the max label of that voxel

    Variables:
        nb_labels: optional numpy array of shape (L,) where L is the number of labels
            if not provided, all non-background (0) labels are computed and averaged
        weights: optional numpy array of shape (L,) giving relative weights of each label
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft

    Usage:
        diceloss = metrics.dice(weights=[1, 2, 3])
        model.compile(diceloss, ...)

    Test:
        import keras.utils as nd_utils
        reload(nrn_metrics)
        weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        nb_labels = len(weights)
        vol_size = [10, 20]
        batch_size = 7

        dice_loss = metrics.Dice(nb_labels=nb_labels).loss
        dice = metrics.Dice(nb_labels=nb_labels).dice
        dice_wloss = metrics.Dice(nb_labels=nb_labels, weights=weights).loss

        # vectors
        lab_size = [batch_size, *vol_size]
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_1 = np.reshape(r, [*lab_size, nb_labels])
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_2 = np.reshape(r, [*lab_size, nb_labels])

        # get some standard vectors
        tf_vec_1 = tf.constant(vec_1, dtype=tf.float32)
        tf_vec_2 = tf.constant(vec_2, dtype=tf.float32)

        # compute some metrics
        res = [f(tf_vec_1, tf_vec_2) for f in [dice, dice_loss, dice_wloss]]
        res_same = [f(tf_vec_1, tf_vec_1) for f in [dice, dice_loss, dice_wloss]]

        # tf run
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(res)
            sess.run(res_same)
            print(res[2].eval())
            print(res_same[2].eval())
    """

    def __init__(self, nb_labels,
                 weights=None,
                 input_type='prob',
                 dice_type='soft',
                 approx_hard_max=True,
                 vox_weights=None,
                 crop_indices=None,
                 area_reg=0.1):  # regularization for bottom of Dice coeff
        """
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft
        approx_hard_max - see note below

        Note: for hard dice, we grab the most likely label and then compute a
        one-hot encoding for each voxel with respect to possible labels. To grab the most
        likely labels, argmax() can be used, but only when Dice is used as a metric
        For a Dice *loss*, argmax is not differentiable, and so we can't use it
        Instead, we approximate the prob->one_hot translation when approx_hard_max is True.
        """

        self.nb_labels = nb_labels
        self.weights = None if weights is None else K.variable(weights)
        self.vox_weights = None if vox_weights is None else K.variable(vox_weights)
        self.input_type = input_type
        self.dice_type = dice_type
        self.approx_hard_max = approx_hard_max
        self.area_reg = area_reg
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)

    def dice(self, y_true, y_pred):
        """
        compute dice for given Tensors

        """
        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        if self.input_type == 'prob':
            # We assume that y_true is probabilistic, but just in case:
            y_true /= K.sum(y_true, axis=-1, keepdims=True)
            y_true = K.clip(y_true, K.epsilon(), 1)

            # make sure pred is a probability
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1)

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one-hot-based matrices of size
        # [batch_size, nb_voxels, nb_labels], where for each voxel in each batch entry,
        # the entries are either 0 or 1
        if self.dice_type == 'hard':

            # if given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                if self.approx_hard_max:
                    y_pred_op = _hard_max(y_pred, axis=-1)
                    y_true_op = _hard_max(y_true, axis=-1)
                else:
                    y_pred_op = _label_to_one_hot(K.argmax(y_pred, axis=-1), self.nb_labels)
                    y_true_op = _label_to_one_hot(K.argmax(y_true, axis=-1), self.nb_labels)

            # if given predicted label, transform to one hot notation
            else:
                assert self.input_type == 'max_label'
                y_pred_op = _label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = _label_to_one_hot(y_true, self.nb_labels)

        # If we're doing soft Dice, require prob output, and the data already is as we need it
        # [batch_size, nb_voxels, nb_labels]
        else:
            assert self.input_type == 'prob', "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        sum_dim = 1
        top = 2 * K.sum(y_true_op * y_pred_op, sum_dim)
        bottom = K.sum(K.square(y_true_op), sum_dim) + K.sum(K.square(y_pred_op), sum_dim)
        # make sure we have no 0s on the bottom. K.epsilon()
        bottom = K.maximum(bottom, self.area_reg)
        return top / bottom

    def mean_dice(self, y_true, y_pred):
        """ weighted mean dice across all patches and labels """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_metric *= self.weights
        if self.vox_weights is not None:
            dice_metric *= self.vox_weights

        # return one minus mean dice as loss
        mean_dice_metric = K.mean(dice_metric)
        tf.verify_tensor_all_finite(mean_dice_metric, 'metric not finite')
        return mean_dice_metric


    def loss(self, y_true, y_pred):
        """ the loss. Assumes y_pred is prob (in [0,1] and sum_row = 1) """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # loss
        dice_loss = 1 - dice_metric

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_loss *= self.weights

        # return one minus mean dice as loss
        mean_dice_loss = K.mean(dice_loss)
        tf.verify_tensor_all_finite(mean_dice_loss, 'Loss not finite')
        return mean_dice_loss


class MeanSquaredError():
    """
    MSE with several weighting options
    """


    def __init__(self, weights=None, vox_weights=None, crop_indices=None):
        """
        Parameters:
            vox_weights is either a numpy array the same size as y_true,
                or a string: 'y_true' or 'expy_true'
            crop_indices: indices to crop each element of the batch
                if each element is N-D (so y_true is N+1 dimensional)
                then crop_indices is a Tensor of crop ranges (indices)
                of size <= N-D. If it's < N-D, then it acts as a slice
                for the last few dimensions.
                See Also: tf.gather_nd
        """
        self.weights = weights
        self.vox_weights = vox_weights
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)
        
    def loss(self, y_true, y_pred):

        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        ksq = K.square(y_pred - y_true)

        if self.vox_weights is not None:
            if self.vox_weights == 'y_true':
                ksq *= y_true
            elif self.vox_weights == 'expy_true':
                ksq *= tf.exp(y_true)
            else:
                ksq *= self.vox_weights

        if self.weights is not None:
            ksq *= self.weights

        return K.mean(ksq)


class Mix():
    """ a mix of several losses """

    def __init__(self, losses, loss_wts=None):
        self.losses = losses
        self.loss_wts = loss_wts
        if loss_wts is None:
            self.loss_wts = np.ones(len(loss_wts))

    def loss(self, y_true, y_pred):
        total_loss = K.variable(0)
        for idx, loss in enumerate(self.losses):
            total_loss += self.loss_wts[idx] * loss(y_true, y_pred)
        return total_loss


class WGAN_GP(object):
    """
    based on https://github.com/rarilurelo/keras_improved_wgan/blob/master/wgan_gp.py
    """

    def __init__(self, disc, batch_size=1, lambda_gp=10):
        self.disc = disc
        self.lambda_gp = lambda_gp
        self.batch_size = batch_size

    def loss(self, y_true, y_pred):

        # get the value for the true and fake images
        disc_true = self.disc(y_true)
        disc_pred = self.disc(y_pred)

        # sample a x_hat by sampling along the line between true and pred
        # z = tf.placeholder(tf.float32, shape=[None, 1])
        # shp = y_true.get_shape()[0]
        # WARNING: SHOULD REALLY BE shape=[batch_size, 1] !!!
        # self.batch_size does not work, since it's not None!!!
        alpha = K.random_uniform(shape=[K.shape(y_pred)[0], 1, 1, 1])
        diff = y_pred - y_true
        interp = y_true + alpha * diff

        # take gradient of D(x_hat)
        gradients = K.gradients(self.disc(interp), [interp])[0]
        grad_pen = K.mean(K.square(K.sqrt(K.sum(K.square(gradients), axis=1))-1))

        # compute loss
        return (K.mean(disc_pred) - K.mean(disc_true)) + self.lambda_gp * grad_pen


class Nonbg(object):
    """ UNTESTED
    class to modify output on operating only on the non-bg class

    All data is aggregated and the (passed) metric is called on flattened true and
    predicted outputs in all (true) non-bg regions

    Usage:
        loss = metrics.dice
        nonbgloss = nonbg(loss).loss
    """

    def __init__(self, metric):
        self.metric = metric

    def loss(self, y_true, y_pred):
        """ prepare a loss of the given metric/loss operating on non-bg data """
        yt = y_true #.eval()
        ytbg = np.where(yt == 0)
        y_true_fix = K.variable(yt.flat(ytbg))
        y_pred_fix = K.variable(y_pred.flat(ytbg))
        return self.metric(y_true_fix, y_pred_fix)


def l1(y_true, y_pred):
    """ L1 metric (MAE) """
    return losses.mean_absolute_error(y_true, y_pred)


def l2(y_true, y_pred):
    """ L2 metric (MSE) """
    return losses.mean_squared_error(y_true, y_pred)


###############################################################################
# Helper Functions
###############################################################################

def _label_to_one_hot(tens, nb_labels):
    """
    Transform a label nD Tensor to a one-hot 3D Tensor. The input tensor is first
    batch-flattened, and then each batch and each voxel gets a one-hot representation
    """
    y = K.batch_flatten(tens)
    return K.one_hot(y, nb_labels)


def _hard_max(tens, axis):
    """
    we can't use the argmax function in a loss, as it's not differentiable
    We can use it in a metric, but not in a loss function
    therefore, we replace the 'hard max' operation (i.e. argmax + onehot)
    with this approximation
    """
    tensmax = K.max(tens, axis=axis, keepdims=True)
    eps_hot = K.maximum(tens - tensmax + K.epsilon(), 0)
    one_hot = eps_hot / K.epsilon()
    return one_hot
