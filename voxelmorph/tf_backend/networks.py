"""
Tensorflow networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs.
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

import sys
import numpy as np

import tensorflow as tf
import keras.backend as K
import keras.layers as KL
from keras.models import Model
from keras.layers import Layer, Conv3D, Activation, Input, UpSampling3D
from keras.layers import concatenate, LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal, Constant

# TODO: switch to nice local imports...
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron as ne

# other vm functions
from . import layers



########################################################
# transform networks
########################################################

def transform(vol_size,
             interp_method='linear',
             indexing='ij',
             nb_feats=1,
             int_steps=0,
             int_method='ss',
             vel_resize=1,
             **kwargs):  # kwargs are for VecInt
    """
    Simple transform model 

    Note: this is essentially a wrapper for the neuron.utils.transform

    TODO: have a new 'Transform' layer that is specific to VoxelMorph that 
    can be a deformation or something else.
    TODO: move SpatialTransform to voxelmorph?
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, nb_feats), name='subj_input')
    trf_input = Input((*[int(f*vel_resize) for f in vol_size], ndims) , name='trf_input')

    if int_steps > 0:
        trf = ne.layers.VecInt(method=int_method,
                               name='trf-int',
                               int_steps=int_steps,
                               **kwargs)(trf_input)
        trf = trf_resize(trf, vel_resize, name='flow') # TODO change to layers.ResizeTransform
        
    else:
        trf = trf_input

    # note the nearest neighbour interpolation method
    # use xy indexing when Guha's original code switched x and y dimensions
    nn_output = ne.layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf])
    return Model([subj_input, trf_input], nn_spatial_output)


def transform_nn(vol_size, **kwargs):
    """
    Simple transform model for nearest-neighbor based transformation
    """
    return transform(vol_size,
                     interp_method='nearest',
                     **kwargs)
    


########################################################
# core networks
########################################################

def unet_core(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x])


def vxmnet(vol_size, enc_nf, dec_nf, full_size=True, int_steps=7, indexing='ij', bidir=False, vel_resize=0.5, use_probs=False):
    """
    Initial attempt at a combined cvpr2018_net + miccai2018_net network. This is a work
    in progress and has not been tested.
    """

    # ensure correct dimensionality
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    # (1) build core unet
    unet = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)  # TODO use neuron unet
    source, target = unet.inputs
    x = unet.outputs[-1]

    # (2) transform unet output into a flow field
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)

    # optionally include probabilities
    if use_probs:
        # we're going to initialize the velocity variance very low, to start stable
        flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                        bias_initializer=Constant(value=-10),
                        name='log_sigma')(x)
        flow_params = concatenate([flow_mean, flow_logsigma])
        flow = layers.Sample(name="z_sample")([flow_mean, flow_logsigma])
    else:
        flow_params = flow_mean
        flow = flow_mean

    # (3) integrate to produce diffeomorphic warp (i.e. treat 'flow' as a stationary velocity field)
    pos_flow = ne.layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(flow)
    if bidir:
        neg_flow = layers.Negate()(flow)
        neg_flow = ne.layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(neg_flow)

    # (4) get up to final resolution
    pos_flow = trf_resize(pos_flow, vel_resize, name='diffflow')
    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # (5) warp the source with the flow field
    y_source = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([source, pos_flow])
    if bidir:
        y_target = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([target, neg_flow])

    # build the model
    outputs = [y_source, y_target, flow_params] if bidir else [y_source, flow_params] 
    return Model(inputs=[source, target], outputs=outputs)


def cvpr2018_net(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src, tgt] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model


def miccai2018_net(vol_size, enc_nf, dec_nf, int_steps=7, use_miccai_int=False, indexing='ij', bidir=False, vel_resize=1/2):
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
            **This param will be phased out (set to False behavior)**
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
            **This param will be phased out (set to 'ij' behavior)**
    :return: the keras model
    """    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=False)
    [src, tgt] = unet_model.inputs
    x_out = unet_model.outputs[-1]

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)
    # we're going to initialize the velocity variance very low, to start stable.
    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=Constant(value=-10),
                            name='log_sigma')(x_out)
    flow_params = concatenate([flow_mean, flow_log_sigma])

    # velocity sample
    flow = layers.Sample(name="z_sample")([flow_mean, flow_log_sigma])

    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    if use_miccai_int:
        # for the miccai2018 submission, the squaring layer
        # scaling was essentially built in by the network
        # was manually composed of a Transform and and Add Layer.
        v = flow
        for _ in range(int_steps):
            v1 = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([v, v])
            v = keras.layers.add([v, v1])
        flow = v

    else:
        # new implementation in neuron is cleaner.
        z_sample = flow
        flow = ne.layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(z_sample)
        if bidir:
            rev_z_sample = layers.Negate()(z_sample)
            neg_flow = ne.layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(rev_z_sample)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')

    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # transform
    y = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    if bidir:
        y_tgt = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing)([tgt, neg_flow])

    # prepare outputs and losses
    outputs = [y, flow_params]
    if bidir:
        outputs = [y, y_tgt, flow_params]

    # build the model
    return Model(inputs=[src, tgt], outputs=outputs)


def cvpr2018_net_probatlas(vol_size, enc_nf, dec_nf, nb_labels,
                           diffeomorphic=True,
                           full_size=True,
                           indexing='ij',
                           init_mu=None,
                           init_sigma=None,
                           stat_post_warp=False,  # compute statistics post warp?
                           network_stat_weight=0.001,
                           warp_method='WARP',
                           stat_nb_feats=16):
    """
    Network to do unsupervised segmentation with probabilistic atlas
    (Dalca et al., submitted to MICCAI 2019)
    """
    # print(warp_method)
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    weaknorm = RandomNormal(mean=0.0, stddev=1e-5)

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size, tgt_feats=nb_labels)
    [src_img, src_atl] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow1 = Conv(ndims, kernel_size=3, padding='same', name='flow', kernel_initializer=weaknorm)(x)
    if diffeomorphic:
        flow2 = ne.layers.VecInt(method='ss', name='flow-int', int_steps=8)(flow1)
    else:
        flow2 = flow1
    if full_size:
        flow = flow2
    else:
        flow = trf_resize(flow2, 1/2, name='diffflow')

    # warp atlas
    if warp_method == 'WARP':
        warped_atlas = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_atlas')([src_atl, flow])
    else:
        warped_atlas = src_atl

    if stat_post_warp:
        assert warp_method == 'WARP', "if computing stat post warp, must do warp... :) set warp_method to 'WARP' or stat_post_warp to False?"

        # combine warped atlas and warpedimage and output mu and log_sigma_squared
        combined = concatenate([warped_atlas, src_img])
    else:
        combined = unet_model.layers[-2].output

    conv1 = conv_block(combined, stat_nb_feats)
    conv2 = conv_block(conv1, nb_labels)
    stat_mu_vol = Conv(nb_labels, kernel_size=3, name='mu_vol',
                    kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv2)
    stat_mu = keras.layers.GlobalMaxPooling3D()(stat_mu_vol)
    stat_logssq_vol = Conv(nb_labels, kernel_size=3, name='logsigmasq_vol',
                        kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv2)
    stat_logssq = keras.layers.GlobalMaxPooling3D()(stat_logssq_vol)

    # combine mu with initializtion
    if init_mu is not None: 
        init_mu = np.array(init_mu)
        stat_mu = Lambda(lambda x: network_stat_weight * x + init_mu, name='comb_mu')(stat_mu)
    
    # combine sigma with initializtion
    if init_sigma is not None: 
        init_logsigmasq = np.array([2*np.log(f) for f in init_sigma])
        stat_logssq = Lambda(lambda x: network_stat_weight * x + init_logsigmasq, name='comb_sigma')(stat_logssq)

    # unnorm log-lik
    def unnorm_loglike(I, mu, logsigmasq, uselog=True):
        P = tf.distributions.Normal(mu, K.exp(logsigmasq/2))
        if uselog:
            return P.log_prob(I)
        else:
            return P.prob(I)

    uloglhood = KL.Lambda(lambda x:unnorm_loglike(*x), name='unsup_likelihood')([src_img, stat_mu, stat_logssq])

    # compute data loss as a layer, because it's a bit easier than outputting a ton of things, etc.
    # def logsum(ll, atl):
    #     pdf = ll * atl
    #     return tf.log(tf.reduce_sum(pdf, -1, keepdims=True) + K.epsilon())

    def logsum_safe(prob_ll, atl):
        """
        safe computation using the log sum exp trick
        e.g. https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        where x = logpdf

        note does not normalize p 
        """
        logpdf = prob_ll + K.log(atl + K.epsilon())
        alpha = tf.reduce_max(logpdf, -1, keepdims=True)
        return alpha + tf.log(tf.reduce_sum(K.exp(logpdf-alpha), -1, keepdims=True) + K.epsilon())

    loss_vol = Lambda(lambda x: logsum_safe(*x))([uloglhood, warped_atlas])

    return Model(inputs=[src_img, src_atl], outputs=[loss_vol, flow])


########################################################
# Atlas creation functions
########################################################


def diff_net(vol_size, enc_nf, dec_nf, int_steps=7, src_feats=1,
             indexing='ij', bidir=False, ret_flows=False, full_size=False,
             vel_resize=1/2, src=None, tgt=None):
    """
    diffeomorphic net, similar to miccai2018, but no sampling.

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
            **This param will be phased out (set to False behavior)**
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
            **This param will be phased out (set to 'ij' behavior)**
    :return: the keras model
    """    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size, src=src, tgt=tgt, src_feats=src_feats)
    [src, tgt] = unet_model.inputs

    # velocity sample
    # unet_model.layers[-1].name = 'vel'
    # vel = unet_model.output
    x_out = unet_model.outputs[-1]

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    vel = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)

    if full_size and vel_resize != 1:
        vel = trf_resize(vel, 1.0/vel_resize, name='flow-resize')    

    # new implementation in neuron is cleaner.
    flow = ne.layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(vel)
    if bidir:
        # rev_z_sample = Lambda(lambda x: -x)(z_sample)
        neg_vel = layers.Negate()(vel)
        neg_flow = ne.layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(neg_vel)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')
    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # transform
    y = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_src')([src, flow])
    if bidir:
        y_tgt = ne.layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_tgt')([tgt, neg_flow])

    # prepare outputs and losses
    outputs = [y, vel]
    if bidir:
        outputs = [y, y_tgt, vel]

    model = Model(inputs=[src, tgt], outputs=outputs)

    if ret_flows:
        outputs += [model.get_layer('diffflow').output, model.get_layer('neg_diffflow').output]
        return Model(inputs=[src, tgt], outputs=outputs)
    else: 
        return model


def atl_img_model(vol_shape, mult=1.0, src=None, atl_layer_name='img_params'):
    """
    atlas model with flow representation
    idea: starting with some (probably rough) atlas (like a ball or average shape),
    the output atlas is this input ball plus a 
    """

    # get a new layer (std)
    if src is None:
        src = Input(shape=[*vol_shape, 1], name='input_atlas')

    # get the velocity field
    v_layer = layers.LocalParamWithInput(shape=[*vol_shape, 1], mult=mult, name=atl_layer_name,
                                         my_initializer=RandomNormal(mean=0.0, stddev=1e-7))
    v = v_layer(src)  # this is so memory-wasteful...
    return keras.models.Model(src, v)


def cond_img_atlas_diff_model(vol_shape, nf_enc, nf_dec,
                              atl_mult=1.0,
                              bidir=True,
                              smooth_pen_layer='diffflow',
                              vel_resize=1/2,
                              int_steps=5,
                              nb_conv_features=32,
                              cond_im_input_shape=[10,12,14,1],
                              cond_nb_levels=5,
                              cond_conv_size=[3,3,3],
                              use_stack=True,
                              do_mean_layer=True,
                              pheno_input_shape=[1],
                              atlas_feats=1,
                              name='cond_model',
                              mean_cap=100,
                              templcondsi=False,
                              templcondsi_init=None,
                              full_size=False,
                              ret_vm=False,
                              extra_conv_layers=0,
                              **kwargs):
            
    # conv layer class
    Conv = getattr(KL, 'Conv%dD' % len(vol_shape))

    # vm model. inputs: "atlas" (we will replace this) and 
    mn = diff_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, bidir=bidir, src_feats=atlas_feats,
                  full_size=full_size, vel_resize=vel_resize, ret_flows=(not use_stack), **kwargs)
    
    # pre-warp model (atlas model)
    pheno_input = KL.Input(pheno_input_shape, name='pheno_input')
    dense_tensor = KL.Dense(np.prod(cond_im_input_shape), activation='elu')(pheno_input)
    reshape_tensor = KL.Reshape(cond_im_input_shape)(dense_tensor)
    pheno_init_model = keras.models.Model(pheno_input, reshape_tensor)
    pheno_tmp_model = ne.models.conv_dec(nb_conv_features, cond_im_input_shape, cond_nb_levels, cond_conv_size,
                             nb_labels=nb_conv_features, final_pred_activation='linear',
                             input_model=pheno_init_model, name='atlasmodel')
    last_tensor = pheno_tmp_model.output
    for i in range(extra_conv_layers):
        last_tensor = Conv(nb_conv_features, kernel_size=cond_conv_size, padding='same', name='atlas_ec_%d' % i)(last_tensor)
    pout = Conv(atlas_feats, kernel_size=3, padding='same', name='atlasmodel_c',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-7),
                 bias_initializer=RandomNormal(mean=0.0, stddev=1e-7))(last_tensor)
    atlas_input = KL.Input([*vol_shape, atlas_feats], name='atlas_input')
    if not templcondsi:
        atlas_tensor = KL.Add(name='atlas')([atlas_input, pout])
    else:
        atlas_tensor = KL.Add(name='atlas_tmp')([atlas_input, pout])

        # change first channel to be result from seg with another add layer
        tmp_layer = KL.Lambda(lambda x: K.softmax(x[...,1:]))(atlas_tensor)  # this is just tmp. Do not use me.
        cl = Conv(1, kernel_size=1, padding='same', use_bias=False, name='atlas_gen', kernel_initializer=RandomNormal(mean=0, stddev=1e-5))
        ximg = cl(tmp_layer)
        if templcondsi_init is not None:
            w = cl.get_weights()
            w[0] = templcondsi_init.reshape(w[0].shape)
            cl.set_weights(w)
        atlas_tensor = KL.Lambda(lambda x: K.concatenate([x[0], x[1][...,1:]]), name='atlas')([ximg, atlas_tensor]) 

    pheno_model = keras.models.Model([pheno_tmp_model.input, atlas_input], atlas_tensor)

    # stack models
    inputs = pheno_model.inputs + [mn.inputs[1]]

    if use_stack:
        sm = ne.utils.stack_models([pheno_model, mn], [[0]])
        neg_diffflow_out = sm.get_layer('neg_diffflow').get_output_at(-1)
        diffflow_out = mn.get_layer(smooth_pen_layer).get_output_at(-1)
        warped_src = sm.get_layer('warped_src').get_output_at(-1)
        warped_tgt = sm.get_layer('warped_tgt').get_output_at(-1)

    else:
        assert bidir
        assert smooth_pen_layer == 'diffflow'
        warped_src, warped_tgt, _, diffflow_out, neg_diffflow_out = mn(pheno_model.outputs + [mn.inputs[1]])
        sm = keras.models.Model(inputs, [warped_src, warped_tgt])
        
    if do_mean_layer:
        mean_layer = ne.layers.MeanStream(name='mean_stream', cap=mean_cap)(neg_diffflow_out)
        outputs = [warped_src, warped_tgt, mean_layer, diffflow_out]
    else:
        outputs = [warped_src, warped_tgt, diffflow_out]


    model = keras.models.Model(inputs, outputs, name=name)
    if ret_vm:
        return model, mn
    else:
        return model


def img_atlas_diff_model(vol_shape, nf_enc, nf_dec,
                        atl_mult=1.0,
                        bidir=True,
                        smooth_pen_layer='diffflow',
                        atl_int_steps=3,
                        vel_resize=1/2,
                        int_steps=3,
                        mean_cap=100,
                        atl_layer_name='atlas',
                        **kwargs):
    # vm model
    mn = diff_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, bidir=bidir, 
                        vel_resize=vel_resize, **kwargs)
    
    # pre-warp model (atlas model)
    pw = atl_img_model(vol_shape, mult=atl_mult, src=mn.inputs[0], atl_layer_name=atl_layer_name) # Wait I'm confused....

    # stack models
    sm = ne.utils.stack_models([pw, mn], [[0]])
    # note: sm.outputs might be out of order now

    # TODO: I'm not sure the mean layer is the right direction
    mean_layer = ne.layers.MeanStream(name='mean_stream', cap=mean_cap)(sm.get_layer('neg_diffflow').get_output_at(-1))

    outputs = [
        sm.get_layer('warped_src').get_output_at(-1),
        sm.get_layer('warped_tgt').get_output_at(-1),
        mean_layer,
        mn.get_layer(smooth_pen_layer).get_output_at(-1)
    ]

    model = keras.models.Model(mn.inputs, outputs)
    return model


########################################################
# Helper functions
########################################################


def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def trf_resize(trf, vel_resize, name='flow'):
    if vel_resize > 1:
        trf = ne.layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return layers.Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = layers.Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return  ne.layers.Resize(1/vel_resize, name=name)(trf)
