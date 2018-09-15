"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils

# other vm functions
import losses


def unet_core(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    # inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = UpSampling3D()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x])


def unet(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src, tgt] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    flow = Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)

    # warp the source with the flow
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='xy')([src, flow])
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model


def miccai2018_net(vol_size, enc_nf, dec_nf, use_miccai_int=True, int_steps=7, indexing='xy'):
    """
    architecture for probabilistic diffeomoprhic VoxelMorph presented in the MICCAI 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    The stationary velocity field operates in a space (0.5)^3 of vol_size for computational reasons.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6, see unet function.
    :param use_miccai_int: whether to use the manual miccai implementation of scaling and squaring integration
            note that the 'velocity' field outputted in that case was 
            since then we've updated the code to be part of a flexible layer. see neuron.layers.VecInt
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
    :return: the keras model
    """    
    
    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=False)
    [src,tgt] = unet_model.inputs
    x_out = unet_model.outputs[-1]

    # velocity mean and logsigma layers
    flow_mean = Conv3D(3, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)

    flow_log_sigma = Conv3D(3, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10), name='log_sigma')(x_out)
    flow_params = concatenate([flow_mean, flow_log_sigma])

    # velocity sample
    flow = Lambda(sample, name="z_sample")([flow_mean, flow_log_sigma])

    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    if use_miccai_int:
        # for the miccai2018 submission, the scaling and squaring layer
        # was manually composed of a Transform and and Add Layer.
        flow = Lambda(lambda x: x, name='flow-fix')(flow)  # remanant of old code
        v = flow
        for _ in range(int_steps):
            v1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([v, v])
            v = keras.layers.add([v, v1])
        flow = v

    else:
        # new implementation in neuron is cleaner.
        # the 2**int_steps is a correcting factor left over from the miccai implementation.
        # * (2**int_steps)
        flow = Lambda(lambda x: x, name='flow-fix')(flow)
        flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=7)(flow)       

    # get up to final resolution
    flow = Lambda(interp_upsampling, output_shape=vol_size+(3,), name='pre_diffflow')(flow)
    flow = Lambda(lambda arg: arg*2, name='diffflow')(flow)

    # transform
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])

    # prepare outputs and losses
    outputs = [y, flow_params]

    # build the model
    return Model(inputs=[src, tgt], outputs=outputs)


def nn_trf(vol_size):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = nrn_layers.SpatialTransformer(interp_method='nearest', indexing='xy')
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)


# Helper functions
def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z

def interp_upsampling(V):
    """ 
    upsample a field by a factor of 2
    TODO: should switch this to use neuron.utils.interpn()
    """

    [xx, yy, zz] = nrn_utils.volshape_to_ndgrid([f*2 for f in V.get_shape().as_list()[1:4]])
    xx = tf.cast(xx, 'float32')
    yy = tf.cast(yy, 'float32')
    zz = tf.cast(zz, 'float32')
    xx = tf.expand_dims(xx/2-xx, 0)
    yy = tf.expand_dims(yy/2-yy, 0)
    zz = tf.expand_dims(zz/2-zz, 0)
    offset = tf.stack([xx, yy, zz], 4)

    # V = nrn_utils.transform(V, offset)
    V = nrn_layers.SpatialTransformer(interp_method='linear')([V, offset])

    return V

