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
from . import layers

# third party
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras.constraints import maxnorm


###############################################################################
# Roughly volume preserving (e.g. high dim to high dim) models
###############################################################################


def dilation_net(nb_features,
                 input_shape, # input layer shape, vector of size ndims + 1(nb_channels)
                 nb_levels,
                 conv_size,
                 nb_labels,
                 name='dilation_net',
                 prefix=None,
                 feat_mult=1,
                 pool_size=2,
                 use_logp=True,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 use_residuals=False,
                 final_pred_activation='softmax',
                 nb_conv_per_level=1,
                 add_prior_layer=False,
                 add_prior_layer_reg=0,
                 layer_nb_feats=None,
                 batch_norm=None):

    return unet(nb_features,
         input_shape, # input layer shape, vector of size ndims + 1(nb_channels)
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         use_logp=True,
         padding='same',
         activation='elu',
         use_residuals=False,
         dilation_rate_mult=dilation_rate_mult,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         add_prior_layer=False,
         add_prior_layer_reg=0,
         layer_nb_feats=None,
         batch_norm=None)


def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         use_logp=True,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         add_prior_layer=False,
         add_prior_layer_reg=0,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None):
    """
    unet-style keras model with an overdose of parametrization.

    downsampling: 

    for U-net like architecture, we need to use Deconvolution3D.
    However, this is not yet available (maybe soon, it's on a dev branch in github I believe)
    Until then, we'll upsample and convolve.
    TODO: Need to check that UpSampling3D actually does NN-upsampling!

    Parameters:
        nb_features: the number of features at each convolutional level
            see below for `feat_mult` and `layer_nb_feats` for modifiers to this number
        input_shape: input layer shape, vector of size ndims + 1 (nb_channels)
        conv_size: the convolution kernel size
        nb_levels: the number of Unet levels (number of downsamples) in the "encoder" 
            (e.g. 4 would give you 4 levels in encoder, 4 in decoder)
        nb_labels: number of output channels
        name (default: 'unet'): the name of the network
        prefix (default: `name` value): prefix to be added to layer names
        feat_mult (default: 1) multiple for `nb_features` as we go down the encoder levels.
            e.g. feat_mult of 2 and nb_features of 16 would yield 32 features in the 
            second layer, 64 features in the third layer, etc
        pool_size (default: 2): max pooling size (integer or list if specifying per dimension)
        use_logp:
        padding:
        dilation_rate_mult:
        activation:
        use_residuals:
        final_pred_activation:
        nb_conv_per_level:
        add_prior_layer:
        add_prior_layer_reg:
        layer_nb_feats:
        conv_dropout:
        batch_norm:
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm)

    # get decoder
    # use_skip_connections=1 makes it a u-net
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = conv_dec(nb_features,
                         None,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=1,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation='linear' if add_prior_layer else final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         layer_nb_feats=lnf,
                         conv_dropout=conv_dropout,
                         input_model=enc_model)

    final_model = dec_model
    if add_prior_layer:
        final_model = add_prior(dec_model,
                                [*input_shape[:-1], nb_labels],
                                name=model_name + '_prior',
                                use_logp=use_logp,
                                final_pred_activation=final_pred_activation,
                                add_prior_layer_reg=add_prior_layer_reg)

    return final_model


def ae(nb_features,
       input_shape,
       nb_levels,
       conv_size,
       nb_labels,
       enc_size,
       name='ae',
       prefix=None,
       feat_mult=1,
       pool_size=2,
       padding='same',
       activation='elu',
       use_residuals=False,
       nb_conv_per_level=1,
       batch_norm=None,
       enc_batch_norm=None,
       ae_type='conv', # 'dense', or 'conv'
       enc_lambda_layers=None,
       add_prior_layer=False,
       add_prior_layer_reg=0,
       use_logp=True,
       conv_dropout=0,
       include_mu_shift_layer=False,
       single_model=False, # whether to return a single model, or a tuple of models that can be stacked.
       final_pred_activation='softmax',
       src=None,
       src_input=None,
       do_vae=False):
    """
    Convolutional Auto-Encoder.
    Optionally Variational.
    Optionally Dense middle layer

    "Mostly" in that the inner encoding can be (optionally) constructed via dense features.

    Parameters:
        do_vae (bool): whether to do a variational auto-encoder or not.

    enc_lambda_layers functions to try:
        K.softsign

        a = 1
        longtanh = lambda x: K.tanh(x) *  K.log(2 + a * abs(x))
    """

    # naming
    model_name = name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         src=src,
                         src_input=src_input)

    # middle AE structure
    if single_model:
        in_input_shape = None
        in_model = enc_model
    else:
        in_input_shape = enc_model.output.shape.as_list()[1:]
        in_model = None
    mid_ae_model = single_ae(enc_size,
                             in_input_shape,
                             conv_size=conv_size,
                             name=model_name,
                             ae_type=ae_type,
                             input_model=in_model,
                             batch_norm=enc_batch_norm,
                             enc_lambda_layers=enc_lambda_layers,
                             include_mu_shift_layer=include_mu_shift_layer,
                             do_vae=do_vae)

    # decoder
    if single_model:
        in_input_shape = None
        in_model = mid_ae_model
    else:
        in_input_shape = mid_ae_model.output.shape.as_list()[1:]
        in_model = None
    dec_model = conv_dec(nb_features,
                         in_input_shape,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=False,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         conv_dropout=conv_dropout,
                         input_model=in_model)

    if add_prior_layer:
        dec_model = add_prior(dec_model,
                              [*input_shape[:-1],nb_labels],
                              name=model_name,
                              prefix=model_name + '_prior',
                              use_logp=use_logp,
                              final_pred_activation=final_pred_activation,
                              add_prior_layer_reg=add_prior_layer_reg)

    if single_model:
        return dec_model
    else:
        return (dec_model, mid_ae_model, enc_model)


def add_prior(input_model,
              prior_shape,
              name='prior_model',
              prefix=None,
              use_logp=True,
              final_pred_activation='softmax',
              add_prior_layer_reg=0):
    """
    Append post-prior layer to a given model
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # prior input layer
    prior_input_name = '%s-input' % prefix
    prior_tensor = KL.Input(shape=prior_shape, name=prior_input_name)
    prior_tensor_input = prior_tensor
    like_tensor = input_model.output

    # operation varies depending on whether we log() prior or not.
    if use_logp:
        # name = '%s-log' % prefix
        # prior_tensor = KL.Lambda(_log_layer_wrap(add_prior_layer_reg), name=name)(prior_tensor)
        print("Breaking change: use_logp option now requires log input!", file=sys.stderr)
        merge_op = KL.add

    else:
        # using sigmoid to get the likelihood values between 0 and 1
        # note: they won't add up to 1.
        name = '%s_likelihood_sigmoid' % prefix
        like_tensor = KL.Activation('sigmoid', name=name)(like_tensor)
        merge_op = KL.multiply

    # merge the likelihood and prior layers into posterior layer
    name = '%s_posterior' % prefix
    post_tensor = merge_op([prior_tensor, like_tensor], name=name)

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    pred_name = '%s_prediction' % prefix
    if final_pred_activation == 'softmax':
        assert use_logp, 'cannot do softmax when adding prior via P()'
        print("using final_pred_activation %s for %s" % (final_pred_activation, model_name))
        softmax_lambda_fcn = lambda x: tf.keras.activations.softmax(x, axis=-1)
        pred_tensor = KL.Lambda(softmax_lambda_fcn, name=pred_name)(post_tensor)

    else:
        pred_tensor = KL.Activation('linear', name=pred_name)(post_tensor)

    # create the model
    model_inputs = [*input_model.inputs, prior_tensor_input]
    model = Model(inputs=model_inputs, outputs=[pred_tensor], name=model_name)

    # compile
    return model


def single_ae(enc_size,
              input_shape,
              name='single_ae',
              prefix=None,
              ae_type='dense', # 'dense', or 'conv'
              conv_size=None,
              input_model=None,
              enc_lambda_layers=None,
              batch_norm=True,
              padding='same',
              activation=None,
              include_mu_shift_layer=False,
              do_vae=False):
    """
    single-layer Autoencoder (i.e. input - encoding - output)
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    if enc_lambda_layers is None:
        enc_lambda_layers = []

    # prepare input
    input_name = '%s_input' % prefix
    if input_model is None:
        assert input_shape is not None, 'input_shape of input_model is necessary'
        input_tensor = KL.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]
    input_nb_feats = last_tensor.shape.as_list()[-1]

    # prepare conv type based on input
    if ae_type == 'conv':
        ndims = len(input_shape) - 1
        convL = getattr(KL, 'Conv%dD' % ndims)
        assert conv_size is not None, 'with conv ae, need conv_size'
    conv_kwargs = {'padding': padding, 'activation': activation}



    # if want to go through a dense layer in the middle of the U, need to:
    # - flatten last layer if not flat
    # - do dense encoding and decoding
    # - unflatten (rehsape spatially) at end
    if ae_type == 'dense' and len(input_shape) > 1:
        name = '%s_ae_%s_down_flat' % (prefix, ae_type)
        last_tensor = KL.Flatten(name=name)(last_tensor)

    # recall this layer
    pre_enc_layer = last_tensor

    # encoding layer
    if ae_type == 'dense':
        assert len(enc_size) == 1, "enc_size should be of length 1 for dense layer"

        enc_size_str = ''.join(['%d_' % d for d in enc_size])[:-1]
        name = '%s_ae_mu_enc_dense_%s' % (prefix, enc_size_str)
        last_tensor = KL.Dense(enc_size[0], name=name)(pre_enc_layer)

    else: # convolution
        # convolve then resize. enc_size should be [nb_dim1, nb_dim2, ..., nb_feats]
        assert len(enc_size) == len(input_shape), \
            "encoding size does not match input shape %d %d" % (len(enc_size), len(input_shape))

        if list(enc_size)[:-1] != list(input_shape)[:-1] and \
            all([f is not None for f in input_shape[:-1]]) and \
            all([f is not None for f in enc_size[:-1]]): 

            # assert len(enc_size) - 1 == 2, "Sorry, I have not yet implemented non-2D resizing -- need to check out interpn!"
            name = '%s_ae_mu_enc_conv' % (prefix)
            last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

            name = '%s_ae_mu_enc' % (prefix)
            zf = [enc_size[:-1][f]/last_tensor.shape.as_list()[1:-1][f] for f in range(len(enc_size)-1)]
            last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
            # resize_fn = lambda x: tf.image.resize_bilinear(x, enc_size[:-1])
            # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

        elif enc_size[-1] is None:  # convolutional, but won't tell us bottleneck
            name = '%s_ae_mu_enc' % (prefix)
            last_tensor = KL.Lambda(lambda x: x, name=name)(pre_enc_layer)

        else:
            name = '%s_ae_mu_enc' % (prefix)
            last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

    if include_mu_shift_layer:
        # shift
        name = '%s_ae_mu_shift' % (prefix)
        last_tensor = layers.LocalBias(name=name)(last_tensor)

    # encoding clean-up layers
    for layer_fcn in enc_lambda_layers:
        lambda_name = layer_fcn.__name__
        name = '%s_ae_mu_%s' % (prefix, lambda_name)
        last_tensor = KL.Lambda(layer_fcn, name=name)(last_tensor)

    if batch_norm is not None:
        name = '%s_ae_mu_bn' % (prefix)
        last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # have a simple layer that does nothing to have a clear name before sampling
    name = '%s_ae_mu' % (prefix)
    last_tensor = KL.Lambda(lambda x: x, name=name)(last_tensor)
    

    # if doing variational AE, will need the sigma layer as well.
    if do_vae:
        mu_tensor = last_tensor

        # encoding layer
        if ae_type == 'dense':
            name = '%s_ae_sigma_enc_dense_%s' % (prefix, enc_size_str)
            last_tensor = KL.Dense(enc_size[0], name=name,
                                #    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                                #    bias_initializer=tf.keras.initializers.RandomNormal(mean=-5.0, stddev=1e-5)
                                   )(pre_enc_layer)

        else:
            if list(enc_size)[:-1] != list(input_shape)[:-1] and \
                all([f is not None for f in input_shape[:-1]]) and \
                all([f is not None for f in enc_size[:-1]]): 

                # assert len(enc_size) - 1 == 2, "Sorry, I have not yet implemented non-2D resizing..."
                name = '%s_ae_sigma_enc_conv' % (prefix)
                last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

                name = '%s_ae_sigma_enc' % (prefix)
                zf = [enc_size[:-1][f]/last_tensor.shape.as_list()[1:-1][f] for f in range(len(enc_size)-1)]
                last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
                # resize_fn = lambda x: tf.image.resize_bilinear(x, enc_size[:-1])
                # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

            elif enc_size[-1] is None:  # convolutional, but won't tell us bottleneck
                name = '%s_ae_sigma_enc' % (prefix)
                last_tensor = convL(pre_enc_layer.shape.as_list()[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)
                # cannot use lambda, then mu and sigma will be same layer.
                # last_tensor = KL.Lambda(lambda x: x, name=name)(pre_enc_layer)

            else:
                name = '%s_ae_sigma_enc' % (prefix)
                last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

        # encoding clean-up layers
        for layer_fcn in enc_lambda_layers:
            lambda_name = layer_fcn.__name__
            name = '%s_ae_sigma_%s' % (prefix, lambda_name)
            last_tensor = KL.Lambda(layer_fcn, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_ae_sigma_bn' % (prefix)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # have a simple layer that does nothing to have a clear name before sampling
        name = '%s_ae_sigma' % (prefix)
        last_tensor = KL.Lambda(lambda x: x, name=name)(last_tensor)

        logvar_tensor = last_tensor

        # VAE sampling 
        name = '%s_ae_sample' % (prefix)
        last_tensor = layers.SampleNormalLogVar(name=name)([mu_tensor, logvar_tensor])

    if include_mu_shift_layer:
        # shift
        name = '%s_ae_sample_shift' % (prefix)
        last_tensor = layers.LocalBias(name=name)(last_tensor)

    # decoding layer
    if ae_type == 'dense':
        name = '%s_ae_%s_dec_flat_%s' % (prefix, ae_type, enc_size_str)
        last_tensor = KL.Dense(np.prod(input_shape), name=name)(last_tensor)

        # unflatten if dense method
        if len(input_shape) > 1:
            name = '%s_ae_%s_dec' % (prefix, ae_type)
            last_tensor = KL.Reshape(input_shape, name=name)(last_tensor)

    else:

        if list(enc_size)[:-1] != list(input_shape)[:-1] and \
            all([f is not None for f in input_shape[:-1]]) and \
            all([f is not None for f in enc_size[:-1]]): 

            name = '%s_ae_mu_dec' % (prefix)
            zf = [input_shape[:-1][f]/enc_size[:-1][f] for f in range(len(enc_size)-1)]
            last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
            # resize_fn = lambda x: tf.image.resize_bilinear(x, input_shape[:-1])
            # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

        name = '%s_ae_%s_dec' % (prefix, ae_type)
        last_tensor = convL(input_nb_feats, conv_size, name=name, **conv_kwargs)(last_tensor)


    if batch_norm is not None:
        name = '%s_bn_ae_%s_dec' % (prefix, ae_type)
        last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # create the model and retun
    model = Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model




###############################################################################
# Encoders, decoders, etc.
###############################################################################


def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             convL=None,  # conv layer function
             src=None,
             src_input=None):
    """
    Fully Convolutional Encoder
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    maxpool = getattr(KL, 'MaxPooling%dD' % ndims)

    # first layer: input
    if src is None:
        name = '%s_input' % prefix
        last_tensor = KL.Input(shape=input_shape, name=name)
        input_tensor = last_tensor
    else:
        assert src_input is not None, 'need to provide src_input if given src'
        input_tensor = src_input
        last_tensor = src


    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features*feat_mult**level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**level

        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level-1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:  # no activation
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)
            
            if conv_dropout > 0:
                # conv dropout along feature space only
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1]*ndims, nb_lvl_feats]
                versions = tf.__version__.split('.')
                ver = int(versions[0])
                rev = int(versions[1])
                if ver < 2 or (ver == 2 and rev < 2): # < 2.2
                    noise_shape=None
                last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

        if use_residuals:
            convarm_layer = last_tensor

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            nb_feats_in = lvl_first_tensor.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(lvl_first_tensor)
                add_layer = last_tensor

                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1]*ndims, nb_lvl_feats]
                    versions = tf.__version__.split('.')
                    ver = int(versions[0])
                    rev = int(versions[1])
                    if ver < 2 or (ver == 2 and rev < 2): # < 2.2
                        noise_shape=None
                    last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = KL.add([add_layer, convarm_layer], name=name)

            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = KL.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             convL=None,
             input_model=None):
    """
    Fully Convolutional Decoder

    Parameters:
        ...
        use_skip_connections (bool): if true, turns an Enc-Dec to a U-Net.
            If true, input_tensor and tensors are required.
            It assumes a particular naming of layers. conv_enc...
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # if using skip connections, make sure need to use them.
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"

    # first layer: input
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]

    # vol size info
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims
    if ndims == 1 and isinstance(pool_size, tuple):
        pool_size = pool_size[0]  # 1D upsampling takes int not tuple

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(KL, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(nb_features*feat_mult**(nb_levels-2-level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**(nb_levels-2-level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor

        # merge layers combining previous layer
        # TODO: add Cropping3D or Cropping2D if 'valid' padding
        if use_skip_connections:
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = KL.concatenate([cat_tensor, last_tensor], axis=ndims+1, name=name)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level-1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1]*ndims, nb_lvl_feats]
                versions = tf.__version__.split('.')
                ver = int(versions[0])
                rev = int(versions[1])
                if ver < 2 or (ver == 2 and rev < 2): # < 2.2
                    noise_shape=None
                last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

        # residual block
        if use_residuals:

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            add_layer = up_tensor
            nb_feats_in = add_layer.get_shape()[-1]
            nb_feats_out = last_tensor.get_shape()[-1]
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)

                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1]*ndims, nb_lvl_feats]
                    last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = KL.add([last_tensor, add_layer], name=name)

            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = KL.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    if final_pred_activation == 'softmax':
        print("using final_pred_activation %s for %s" % (final_pred_activation, model_name))
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: tf.keras.activations.softmax(x, axis=ndims + 1)
        pred_tensor = KL.Lambda(softmax_lambda_fcn, name=name)(last_tensor)

    # otherwise create a layer that does nothing.
    else:
        name = '%s_prediction' % prefix
        if final_pred_activation is None:
            pred_tensor = KL.Activation('linear', name=name)(like_tensor)
        else:
            pred_tensor = KL.Activation(final_pred_activation, name=name)(like_tensor)

    # create the model and retun
    model = Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model


def design_dnn(nb_features, input_shape, nb_levels, conv_size, nb_labels,
               feat_mult=1,
               pool_size=2,
               padding='same',
               activation='elu',
               final_layer='dense-sigmoid',
               conv_dropout=0,
               conv_maxnorm=0,
               nb_input_features=1,
               batch_norm=False,
               name=None,
               prefix=None,
               use_strided_convolution_maxpool=True,
               nb_conv_per_level=2):
    """
    "deep" cnn with dense or global max pooling layer @ end...

    Could use sequential...
    """


    def _global_max_nd(xtens):
        ytens = K.batch_flatten(xtens)
        return K.max(ytens, 1, keepdims=True)


    model_name = name
    if model_name is None:
        model_name = 'model_1'
    if prefix is None:
        prefix = model_name

    ndims = len(input_shape)
    input_shape = tuple(input_shape)

    convL = getattr(KL, 'Conv%dD' % ndims)
    maxpool = KL.MaxPooling3D if len(input_shape) == 3 else KL.MaxPooling2D
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}
    if conv_maxnorm > 0:
        conv_kwargs['kernel_constraint'] = maxnorm(conv_maxnorm)

    # initialize a dictionary
    enc_tensors = {}

    # first layer: input
    name = '%s_input' % prefix
    enc_tensors[name] = KL.Input(shape=input_shape + (nb_input_features,), name=name)
    last_tensor = enc_tensors[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        for conv in range(nb_conv_per_level):
            if conv_dropout > 0:
                name = '%s_dropout_%d_%d' % (prefix, level, conv)
                enc_tensors[name] = KL.Dropout(conv_dropout)(last_tensor)
                last_tensor = enc_tensors[name]

            name = '%s_conv_%d_%d' % (prefix, level, conv)
            nb_lvl_feats = np.round(nb_features*feat_mult**level).astype(int)
            enc_tensors[name] = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            last_tensor = enc_tensors[name]

        # max pool
        if use_strided_convolution_maxpool:
            name = '%s_strided_conv_%d' % (prefix, level)
            enc_tensors[name] = convL(nb_lvl_feats, pool_size, **conv_kwargs, name=name)(last_tensor)
            last_tensor = enc_tensors[name]
        else:
            name = '%s_maxpool_%d' % (prefix, level)
            enc_tensors[name] = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)
            last_tensor = enc_tensors[name]

    # dense layer
    if final_layer == 'dense-sigmoid':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(1, name=name, activation="sigmoid")(last_tensor)

    elif final_layer == 'dense-tanh':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(1, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # Omittting BatchNorm for now, it seems to have a cpu vs gpu problem
        # https://github.com/tensorflow/tensorflow/pull/8906
        # https://github.com/fchollet/keras/issues/5802
        # name = '%s_%s_bn' % prefix
        # enc_tensors[name] = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
        # last_tensor = enc_tensors[name]

        name = '%s_%s_tanh' % prefix
        enc_tensors[name] = KL.Activation(activation="tanh", name=name)(last_tensor)

    elif final_layer == 'dense-softmax':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(nb_labels, name=name, activation="softmax")(last_tensor)

    # global max pooling layer
    elif final_layer == 'myglobalmaxpooling':

        name = '%s_batch_norm' % prefix
        enc_tensors[name] = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool' % prefix
        enc_tensors[name] = KL.Lambda(_global_max_nd, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool_reshape' % prefix
        enc_tensors[name] = KL.Reshape((1, 1), name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_sigmoid' % prefix
        enc_tensors[name] = KL.Conv1D(1, 1, name=name, activation="sigmoid", use_bias=True)(last_tensor)

    elif final_layer == 'globalmaxpooling':

        name = '%s_conv_to_featmaps' % prefix
        enc_tensors[name] = KL.Conv3D(2, 1, name=name, activation="relu")(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool' % prefix
        enc_tensors[name] = KL.GlobalMaxPooling3D(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_softmax' % prefix
        enc_tensors[name] = KL.Activation('softmax', name=name)(last_tensor)

    last_tensor = enc_tensors[name]

    # create the model
    model = Model(inputs=[enc_tensors['%s_input' % prefix]], outputs=[last_tensor], name=model_name)
    return model



def EncoderNet(nb_features,
               input_shape,
               nb_levels,
               conv_size,
               name=None,
               prefix=None,
               feat_mult=1,
               pool_size=2,
               dilation_rate_mult=1,
               padding='same',
               activation='elu',
               layer_nb_feats=None,
               use_residuals=False,
               nb_conv_per_level=2,
               conv_dropout=0,
               dense_size=256,
               nb_labels=2,
               final_activation=None,
               rescale=None,
               dropout=None, 
               batch_norm=None):
    """
    Fully Convolutional Encoder-based classifer
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    The end of the encoders/downsampling is flattened and attached to a dense
    layer with dense_size units, then to the nb_labels output nodes. For other
    parameters see conv_env documentation
    """

    # allocate the encoder arm
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm)

    # run the encoder outputs through a dense layer
    flat = KL.Flatten()(enc_model.outputs[0])
    if dropout is not None and dropout > 0:
        flat = KL.Dropout(dropout, name='dropout_flat')(flat)
    dense = KL.Dense(dense_size, name='dense')(flat)
    if dropout is not None and dropout > 0:
        dense = KL.Dropout(dropout, name='dropout_dense')(dense)

    if nb_labels <= 0:  # if labels <=0 assume a regression net
        nb_labels = 1
        if (final_activation is None):
            final_activation = 'linear'
    else:  # if labels>=1 assume a classification net
        if (final_activation is None):
            final_activation = 'softmax'

    if (rescale is not None):
        dense = layers.RescaleValues(rescale)(dense)
    out = KL.Dense(nb_labels, name='output_dense', activation=final_activation)(dense)
    model = keras.models.Model(inputs=enc_model.inputs, outputs=out)
    
    return model


def DenseLayerNet(inshape, layer_sizes, nb_labels=2, activation='relu', final_activation='softmax', dropout=None, batch_norm=None):
    """
    A densenet that connects a set of dense layers to  a classification
    output. 
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    """
    inputs = KL.Input(shape=inshape, name='input')
    prev_layer = KL.Flatten(name='flat_inputs')(inputs)
    # to prevent overfitting include some kernel and bias regularization
    kreg = keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    breg = keras.regularizers.l2(1e-4)

    # connect the list of dense layers to each other
    for lno, layer_size in enumerate(layer_sizes):
        prev_layer = KL.Dense(layer_size, name='dense%d' % lno, activation=activation,kernel_regularizer=kreg, bias_regularizer=breg)(prev_layer)
        if dropout is not None:
            prev_layer = KL.Dropout(dropout, name='dropout%d'%lno)(prev_layer)
        if batch_norm is not None:
            prev_layer = KL.BatchNormalization(name='BatchNorm%d'%lno)(prev_layer)
            
    # tie the previous dense layer to a onehot encoded output layer
    last_layer = KL.Dense(nb_labels, name='last_dense', activation=final_activation)(prev_layer)

    model = keras.models.Model(inputs=inputs, outputs=last_layer)
    return(model)


###############################################################################
# Helper function
###############################################################################
