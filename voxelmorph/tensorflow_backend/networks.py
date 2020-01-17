import numpy as np
import neuron as ne
import tensorflow as tf

import keras
import keras.backend as K
import keras.layers as KL
from keras.models import Model, Sequential
from keras.layers import Layer, Conv3D, Activation, Input, UpSampling3D
from keras.layers import concatenate, LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal, Constant

from .. import utils
from . import layers


def transform(
        inshape,
        interp_method='linear',
        indexing='ij',
        nb_feats=1,
        int_steps=0,
        int_method='ss',
        vel_resize=1,
        **kwargs  # kwargs are for VecInt
    ):
    """
    Simple transform model.

    NOTE: This is essentially a wrapper for neuron.utils.transform.
    TODO: Have a new 'Transform' layer that is specific to VoxelMorph that can be a deformation or something else.
    """
    ndims = len(inshape)

    # nn warp model
    scan_input = Input((*inshape, nb_feats), name='scan_input')
    trf_input = Input((*[int(f*vel_resize) for f in inshape], ndims) , name='trf_input')

    if int_steps > 0:
        trf = ne.layers.VecInt(method=int_method, name='trf-int', int_steps=int_steps, **kwargs)(trf_input)
        trf = trf_resize(trf, vel_resize, name='flow')  # TODO change to ResizeTransform
    else:
        trf = trf_input

    # note the nearest neighbour interpolation method
    # use xy indexing when Guha's original code switched x and y dimensions
    nn_output = ne.layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)
    nn_spatial_output = nn_output([scan_input, trf])
    return Model([scan_input, trf_input], nn_spatial_output)


def transform_nn(inshape, **kwargs):
    """
    Simple transform model for nearest-neighbor based transformation.
    """
    return transform(inshape, interp_method='nearest', **kwargs)


def unet(inshape, enc_nf, dec_nf, src_feats=1, trg_feats=1):
    """ 
    Unet architecture for the voxelmorph models.

    Parameters:
        inshape: Input shape. e.g. (256, 256, 256)
        enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
        dec_nf: List of decoder filters. e.g. [32, 32, 32, 32, 8, 8]
        src_feats: Number of source image features. Default is 1.
        trg_feats: Number of target image features. Default is 1.
    """

    # configure inputs
    source = Input(shape=(*inshape, src_feats))
    target = Input(shape=(*inshape, trg_feats))

    # configure encoder (down-sampling path)
    enc_layers = [concatenate([source, target])]
    for nf in enc_nf:
        enc_layers.append(conv_block(enc_layers[-1], nf, strides=2))

    # configure decoder (up-sampling path)
    x = enc_layers.pop()
    for nf in dec_nf[:len(enc_nf)]:
        x = conv_block(x, nf, strides=1)
        x = upsample_block(x, enc_layers.pop())

    # now we take care of the remaining convolutions
    for i, nf in enumerate(dec_nf[len(enc_nf):]):
        x = conv_block(x, nf, strides=1)

    return Model(inputs=[source, target], outputs=[x])


def vxm_net(inshape, enc_nf, dec_nf, int_steps=7, int_downsize=2, bidir=False, use_probs=False, src_feats=1, trg_feats=1):
    """ 
    VoxelMorph model that registers two images.

    Parameters:
        inshape: Input shape. e.g. (256, 256, 256)
        enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
        dec_nf: List of decoder filters. e.g. [32, 32, 32, 32, 8, 8]
        int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
        int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
            is not downsampled when this value is 1.
        bidir: Enable bidirectional cost function. Default is False.
        use_probs: Use probabilities in flow field. Default is False.
        src_feats: Number of source image features. Default is 1.
        trg_feats: Number of target image features. Default is 1.
    """

    # ensure correct dimensionality
    ndims = len(inshape)
    assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    # build core unet model and grab inputs
    unet_model = unet(inshape, enc_nf, dec_nf, src_feats=src_feats, trg_feats=trg_feats)
    source, target = unet_model.inputs

    # transform unet output into a flow field
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(unet_model.output)

    # optionally include probabilities
    if use_probs:
        # initialize the velocity variance very low, to start stable
        flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                        bias_initializer=Constant(value=-10),
                        name='log_sigma')(unet_model.output)
        flow_params = concatenate([flow_mean, flow_logsigma])
        flow = ne.layers.SampleNormalLogVar(name="z_sample")([flow_mean, flow_logsigma])
    else:
        flow_params = flow_mean
        flow = flow_mean

    # optionally resize for integration
    if int_steps > 0 and int_downsize > 1:
        flow = trf_resize(flow, int_downsize, name='resize')

    # optionally negate flow for bidirectional model
    pos_flow = flow
    if bidir:
        neg_flow = ne.layers.Negate()(flow)

    # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
    if int_steps > 0:
        pos_flow = ne.layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(pos_flow)
        if bidir:
            neg_flow = ne.layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(neg_flow)

        # resize to final resolution
        if int_downsize > 1:
            pos_flow = trf_resize(pos_flow, 1 / int_downsize, name='diffflow')
            if bidir:
                neg_flow = trf_resize(neg_flow, 1 / int_downsize, name='neg_diffflow')

    # warp image with flow field
    y_source = ne.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='transformer')([source, pos_flow])
    if bidir:
        y_target = ne.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='neg_transformer')([target, neg_flow])

    # build the model
    outputs = [y_source, y_target, flow_params] if bidir else [y_source, flow_params]
    return Model(inputs=[source, target], outputs=outputs)


def affine_net(inshape, enc_nf, blurs=[1], return_affines=False):
    """
    Affine VoxelMorph network to align two images.

    Parameters:
        inshape: Input shape. e.g. (256, 256, 256)
        enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
        blurs: List of gaussian blur kernel levels for inputs. Default is [1].
        return_affines: Returns affines in addition to model.  Default is False.
    """

    # ensure correct dimensionality
    ndims = len(inshape)
    assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    # configure base encoder CNN
    Conv = getattr(KL, 'Conv%dD' % ndims)   
    basenet = Sequential()
    for nf in enc_nf:
        basenet.add(Conv(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=2))
        basenet.add(LeakyReLU(0.2))
    
    # dense layer to affine matrix
    basenet.add(KL.Flatten())
    basenet.add(KL.Dense(ndims * (ndims + 1)))

    # inputs
    source = Input(shape=[*inshape, 1])
    target = Input(shape=[*inshape, 1])

    # build net with multi-scales
    affines = []
    scale_source = source
    for blur in blurs:
        # set input and blur using gaussian kernel  
        source_blur = gaussian_blur(scale_source, blur, ndims)
        target_blur = gaussian_blur(target, blur, ndims)
        x_in = concatenate([source_blur, target_blur])
            
        # apply base net to affine
        affine = basenet(x_in)
        affines.append(affine)
        
        # spatial transform using affine matrix
        y_source = ne.layers.SpatialTransformer()([source_blur, affine])
        
        # provide new input for next scale
        if len(blurs) > 1:
            scale_source = ne.layers.SpatialTransformer()([scale_source, affine])

    if return_affines:
        return Model(inputs=[source, target], outputs=[y_source]), affines
    else:
        return Model(inputs=[source, target], outputs=[y_source])


def affine_transformer(inshape):
    """
    Transformer network that applies an affine registration matrix to an image.
    """
    source = Input((*inshape, 1))
    affine = Input((12,))
    aligned = ne.layers.SpatialTransformer()([source, affine])
    return Model([source, affine], aligned)


def cvpr2018_net_probatlas(inshape, enc_nf, dec_nf, nb_labels,
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
    ndims = len(inshape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    weaknorm = RandomNormal(mean=0.0, stddev=1e-5)

    # get the core model
    unet_model = unet(inshape, enc_nf, dec_nf, tgt_feats=nb_labels)
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

    # combine mu with initialization
    if init_mu is not None: 
        init_mu = np.array(init_mu)
        stat_mu = Lambda(lambda x: network_stat_weight * x + init_mu, name='comb_mu')(stat_mu)
    
    # combine sigma with initialization
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

    # voxelmorph model
    downsize = 1 / vel_resize
    mn = vxm_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, int_downsize=downsize, bidir=bidir, src_feats=atlas_feats, **kwargs)

    if not use_stack:
        # return warp in model output
        warp = vxm_net.get_layer('transformer').input[1]
        neg_warp = vxm_net.get_layer('transformer').input[1]
        outputs = mn.outputs + [warp, neg_warp]
        mn = Model(inputs=mn.inputs, outputs=outputs)

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
    downsize = 1 / vel_resize
    mn = vxm_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, int_downsize=downsize, bidir=bidir, **kwargs)
    
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


def conv_block(x, nfeat, strides=1):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), "ndims should be one of 1, 2, or 3. found: %d" % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides)(x)
    return LeakyReLU(0.2)(convolved)


def upsample_block(x, connection):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), "ndims should be one of 1, 2, or 3. found: %d" % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    upsampled = UpSampling()(x)
    return concatenate([upsampled, connection])


def trf_resize(trf, vel_resize, name='flow'):
    """
    Resizes a transform by a given factor.
    """
    if vel_resize > 1:
        trf = ne.layers.Resize(1 / vel_resize, name=name+'_tmp')(trf)
        return ne.layers.RescaleValues(1 / vel_resize, name=name)(trf)
    else:
        # multiply first to save memory (multiply in smaller space)
        trf = ne.layers.RescaleValues(1 / vel_resize, name=name+'_tmp')(trf)
        return  ne.layers.Resize(1 / vel_resize, name=name)(trf)


def gaussian_blur(tensor, level, ndims):
    """
    Blurs a tensor using a gaussian kernel (if level=1, then do nothing).
    """
    if level > 1:
        sigma = (level-1) ** 2
        blur_kernel = ne.utils.gaussian_kernel([sigma] * ndims)
        blur_kernel = tf.reshape(blur_kernel, blur_kernel.shape.as_list() + [1, 1])
        conv = lambda x: tf.nn.conv3d(x, blur_kernel, [1, 1, 1, 1, 1], 'SAME')
        return KL.Lambda(conv)(tensor)
    elif level == 1:
        return tensor
    else:
        raise ValueError('Gaussian blur level must not be less than 1')


def build_warpnet(vxm_net):
    """
    Builds a model from a vxm_net that returns the warped image
    and diffeomorphic warp instead of the non-integrated flow field.
    """
    warp = vxm_net.get_layer('transformer').input[1]
    return keras.models.Model(vxm_net.inputs, [vxm_net.outputs[0], warp])


# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel
