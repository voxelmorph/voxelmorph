"""
tensorflow/keras regularizers for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import tensorflow as tf
import keras.backend as K

from .utils import soft_delta



def soft_l0_wrap(wt = 1.):

    def soft_l0(x):
        """
        maximize the number of 0 weights
        """
        nb_weights = tf.cast(tf.size(x), tf.float32)
        nb_zero_wts = tf.reduce_sum(soft_delta(K.flatten(x)))
        return wt * (nb_weights - nb_zero_wts) / nb_weights

    return soft_l0
