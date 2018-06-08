"""
Networks for voxelwarp model
"""

# third party
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras
import numpy as np

# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
import losses



def unet(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size

    """

    # inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    if full_size:
        x = UpSampling3D()(x)
        x = concatenate([x, x_in])
        x = myConv(x, dec_nf[5])

        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])

    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)

    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])

    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model


def myConv(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
