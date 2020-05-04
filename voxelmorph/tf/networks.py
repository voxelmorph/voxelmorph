import numpy as np
from collections.abc import Iterable

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import concatenate, LeakyReLU, Reshape, Lambda
from tensorflow.keras.initializers import RandomNormal, Constant

from .. import default_unet_features
from . import layers
from . import neuron as ne
from .modelio import LoadableModel, store_config_args
from .utils import gaussian_blur, value_at_location, point_spatial_transformer


# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            input_model=None):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level
        )

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
            flow_params = concatenate([flow_mean, flow_logsigma], name='prob_concat')
            flow = ne.layers.SampleNormalLogVar(name="z_sample")([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        # optionally resize for integration
        if int_steps > 0 and int_downsize > 1:
            flow = layers.RescaleTransform(1 / int_downsize, name='flow_resize')(flow)

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='neg_flow')(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = ne.layers.VecInt(method='ss', name='flow_int', int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = ne.layers.VecInt(method='ss', name='neg_flow_int', int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='diffflow')(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='neg_diffflow')(neg_flow)

        # warp image with flow field
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='transformer')([source, pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='neg_transformer')([target, neg_flow])

        # initialize the keras model
        outputs = [y_source, y_target, flow_params] if bidir else [y_source, flow_params]
        super().__init__(name='vxm_dense', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffine(LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images.
    """

    @store_config_args
    def __init__(self, inshape, enc_nf, bidir=False, transform_type='affine', blurs=[1], rescale_affine=1.0):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            bidir: Enable bidirectional cost function. Default is False.
            transform_type: 'affine' (default), 'rigid' or 'rigid+scale' currently
            blurs: List of gaussian blur kernel levels for inputs. Default is [1].
            rescale_affine: a scalar (or ndims*(ndims+1) array) to rescale the output of the dense layer
                this improves stability by enabling different gradient flow to affect the affine parameters
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure base encoder CNN
        Conv = getattr(KL, 'Conv%dD' % ndims)   
        basenet = tf.keras.Sequential(name='core_model')
        for nf in enc_nf:
            basenet.add(Conv(nf, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=2))
            basenet.add(LeakyReLU(0.2))
        
        # dense layer to affine matrix
        basenet.add(KL.Flatten())

        if transform_type == 'rigid':
            print('Warning: rigid registration has not been fully tested')
            basenet.add(KL.Dense(ndims * 2, name='dense'))
            basenet.add(layers.AffineTransformationsToMatrix(ndims, name='matrix_conversion'))
        elif transform_type == 'rigid+scale':
            print('Warning: rigid registration has not been fully tested')
            basenet.add(KL.Dense(ndims * 2+1, name='dense'))
            basenet.add(layers.AffineTransformationsToMatrix(ndims, scale=True, name='matrix_conversion'))
        else:
            basenet.add(KL.Dense(ndims * (ndims + 1), name='dense'))

        # inputs
        source = tf.keras.Input(shape=[*inshape, 1], name='source_input')
        target = tf.keras.Input(shape=[*inshape, 1], name='target_input')

        scale_affines = []
        full_affine = None
        y_source = source

        # build net with multi-scales
        for blur_num, blur in enumerate(blurs):
            # get layer name prefix
            prefix = 'blur_%d_' % blur_num

            # set input and blur using gaussian kernel  
            source_blur = gaussian_blur(y_source, blur, ndims)
            target_blur = gaussian_blur(target, blur, ndims)

            # per-scale affine encoder
            curr_affine_scaled = basenet(concatenate([source_blur, target_blur], name=prefix+'concat'))
            curr_affine = ne.layers.RescaleValues(rescale_affine, name=prefix+'rescale')(curr_affine_scaled)
            scale_affines.append(curr_affine)

            # compose affine at this scale
            if full_affine is None:
                full_affine = curr_affine
            else:
                full_affine = layers.ComposeTransform(name=prefix+'compose')([full_affine, curr_affine])

            # provide new input for next scale
            y_source = layers.SpatialTransformer(name=prefix+'transformer')([source, full_affine])

        # invert affine for bidirectional training
        if bidir:
            inv_affine = layers.InvertAffine(name='invert_affine')(full_affine)
            y_target = layers.SpatialTransformer(name='neg_transformer')([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name='affine_net', inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = LoadableModel.ReferenceContainer()
        self.references.affine = full_affine
        self.references.scale_affines = scale_affines
        self.references.transform_type = transform_type

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.affine)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineDense(LoadableModel):
    """
    VoxelMorph network to perform combined affine and nonlinear registration.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        enc_nf_affine=None,
        transform_type='affine',
        affine_bidir=False,
        affine_blurs=[1],
        **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            enc_nf_affine: List of affine encoder filters. Default is None (uses unet encoder features).
            transform_type:  See VxmAffine for types. Default is 'affine'.
            affine_bidir: Enable bidirectional affine training. Default is False.
            affine_blurs: List of blurring levels for affine transform. Default is [1].
            kwargs: Forwarded to the internal VxmDense model.
        """

        # default encoder and decoder layer features if nothing provided
        if nb_unet_features is None:
            nb_unet_features = default_unet_features()

        # use dense-net encoder features if affine-net features aren't provided
        if enc_nf_affine is None:
            if isinstance(nb_unet_features, int):
                raise ValueError('enc_nf_affine list must be provided when nb_unet_features is an integer')
            enc_nf_affine = nb_unet_features[0]

        # affine component
        affine_model = VxmAffine(inshape, enc_nf_affine, transform_type=transform_type, bidir=affine_bidir, blurs=affine_blurs)
        source = affine_model.inputs[0]
        affine = affine_model.references.affine

        # build a dense model that takes the affine transformed src as input
        dense_input_model = tf.keras.Model(affine_model.inputs, (affine_model.outputs[0], affine_model.inputs[1]))
        dense_model = VxmDense(inshape, nb_unet_features=nb_unet_features, input_model=dense_input_model, **kwargs)
        flow_params = dense_model.outputs[1]
        pos_flow = dense_model.references.pos_flow

        # build a single transform that applies both affine and dense to src
        # and apply it to the input (src) volume so that there is only 1 interpolation
        # and output it as the combined model output (plus the dense warp)
        composed = layers.ComposeTransform()([affine, pos_flow])
        y_source = layers.SpatialTransformer()([source, composed])

        # initialize the keras model
        super().__init__(inputs=affine_model.inputs, outputs=[y_source, flow_params])

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.affine = affine
        self.references.pos_flow = pos_flow
        self.references.composed = composed

    def get_split_registration_model(self):
        """
        Returns a reconfigured model to predict only the final affine and dense transforms.
        """
        return tf.keras.Model(self.inputs, [self.references.affine, self.references.pos_flow])

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final composed transform.
        """
        return tf.keras.Model(self.inputs, self.references.composed)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class InstanceTrainer(LoadableModel):
    """
    VoxelMorph network to perform instance-specific optimization.
    """

    @store_config_args
    def __init__(self, inshape, warp):
        source = tf.keras.Input(shape=inshape)
        target = tf.keras.Input(shape=inshape)
        nullwarp = tf.keras.Input(shape=warp.shape[1:])  # this is basically ignored by LocalParamWithInput
        flow_layer = vxm.layers.LocalParamWithInput(shape=warp.shape[1:])
        flow = flow_layer(nullwarp)
        y = vxm.layers.SpatialTransformer()([source, flow])

        # initialize the keras model
        super().__init__(name='instance_net', inputs=[source, target, nullwarp], outputs=[y, flow])

        # initialize weights with original predicted warp
        flow_layer.set_weights(warp)


class ProbAtlasSegmentation(LoadableModel):
    """
    VoxelMorph network to segment images by warping a probabilistic atlas.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_labels,
        nb_unet_features=None,
        init_mu=None,
        init_sigma=None,
        warp_atlas=True,
        stat_post_warp=True,
        stat_nb_feats=16,
        network_stat_weight=0.001,
        **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_labels: Number of labels in probabilistic atlas.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            init_mu: Optional initialization for gaussian means. Default is None.
            init_sigma: Optional initialization for gaussian sigmas. Default is None.
            stat_post_warp: Computes gaussian stats using the warped atlas. Default is True.
            stat_nb_feats: Number of features in the stats convolutional layer. Default is 16.
            network_stat_weight: Relative weight of the stats learned by the network. Default is 0.001.
            kwargs: Forwarded to the internal VxmDense model.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build warp network
        vxm_model = VxmDense(inshape, nb_unet_features=nb_unet_features, src_feats=nb_labels, **kwargs)

        # extract necessary layers from the network
        # important to note that we're warping the atlas to the image in this case and
        # we'll swap the input order later
        atlas, image = vxm_model.inputs
        warped_atlas = vxm_model.references.y_source if warp_atlas else atlas
        flow = vxm_model.references.pos_flow

        # compute stat using the warped atlas (or not)
        if stat_post_warp:
            assert warp_atlas, 'must enable warp_atlas if computing stat post warp'
            combined = concatenate([warped_atlas, image], name='post_warp_concat')
        else:
            # use last convolution in the unet before the flow convolution
            combined = vxm_model.references.unet_model.layers[-2].output

        # convolve into nlabel-stat volume
        conv = conv_block(combined, stat_nb_feats)
        conv = conv_block(conv, nb_labels)

        Conv = getattr(KL, 'Conv%dD' % ndims)
        weaknorm = RandomNormal(mean=0.0, stddev=1e-5)

        # convolve into mu and sigma volumes
        stat_mu_vol = Conv(nb_labels, kernel_size=3, name='mu_vol', kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv)
        stat_logssq_vol = Conv(nb_labels, kernel_size=3, name='logsigmasq_vol', kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv)
        
        # pool to get 'final' stat
        stat_mu = tf.keras.layers.GlobalMaxPooling3D(name='mu_pooling')(stat_mu_vol)
        stat_logssq = tf.keras.layers.GlobalMaxPooling3D(name='logssq_pooling')(stat_logssq_vol)

        # combine mu with initialization
        if init_mu is not None: 
            init_mu = np.array(init_mu)
            stat_mu = Lambda(lambda x: network_stat_weight * x + init_mu, name='comb_mu')(stat_mu)
        
        # combine sigma with initialization
        if init_sigma is not None: 
            init_logsigmasq = np.array([2 * np.log(f) for f in init_sigma])
            stat_logssq = Lambda(lambda x: network_stat_weight * x + init_logsigmasq, name='comb_sigma')(stat_logssq)

        # unnorm loglike
        def unnorm_loglike(I, mu, logsigmasq, use_log=True):
            P = tf.distributions.Normal(mu, K.exp(logsigmasq/2))
            return P.log_prob(I) if use_log else P.prob(I)
        uloglhood = KL.Lambda(lambda x:unnorm_loglike(*x), name='unsup_likelihood')([image, stat_mu, stat_logssq])

        # compute data loss as a layer, because it's a bit easier than outputting a ton of things
        def logsum(prob_ll, atl):
            # safe computation using the log sum exp trick (NOTE: this does not normalize p)
            # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning
            logpdf = prob_ll + K.log(atl + K.epsilon())
            alpha = tf.reduce_max(logpdf, -1, keepdims=True)
            return alpha + tf.log(tf.reduce_sum(K.exp(logpdf-alpha), -1, keepdims=True) + K.epsilon())
        loss_vol = Lambda(lambda x: logsum(*x))([uloglhood, warped_atlas])

        # initialize the keras model
        super().__init__(inputs=[image, atlas], outputs=[loss_vol, flow])

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.vxm_model = vxm_model
        self.references.uloglhood = uloglhood
        self.references.stat_mu = stat_mu
        self.references.stat_logssq = stat_logssq

    def get_gaussian_warp_model(self):
        """
        Extracts a predictor model from the ProbAtlasSegmentation model that directly
        outputs the gaussian stats and warp field.
        """
        outputs = [
            self.references.uloglhood,
            self.references.stat_mu,
            self.references.stat_logssq,
            self.outputs[-1]
        ]
        return tf.keras.Model(self.inputs, outputs)


class TemplateCreation(LoadableModel):
    """
    VoxelMorph network to generate an unconditional template image.
    """

    @store_config_args
    def __init__(self, inshape, nb_unet_features=None, mean_cap=100, atlas_feats=1, src_feats=1, **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            mean_cap: Cap for mean stream. Default is 100.
            atlas_feats: Number of atlas/template features. Default is 1.
            src_feats: Number of source image features. Default is 1.
            kwargs: Forwarded to the internal VxmDense model.
        """

        # configure inputs
        atlas_input = tf.keras.Input(shape=[*inshape, atlas_feats], name='atlas_input')
        source_input = tf.keras.Input(shape=[*inshape, src_feats], name='source_input')

        # pre-warp (atlas) model
        atlas_layer = layers.LocalParamWithInput(name='atlas', shape=(*inshape, 1), mult=1.0, initializer=RandomNormal(mean=0.0, stddev=1e-7))
        atlas_tensor = atlas_layer(atlas_input)
        warp_input_model = tf.keras.Model([atlas_input, source_input], outputs=[atlas_tensor, source_input])

        # warp model
        vxm_model = VxmDense(inshape, nb_unet_features=nb_unet_features, bidir=True, input_model=warp_input_model, **kwargs)

        # extract tensors from stacked model
        y_source = vxm_model.references.y_source
        y_target = vxm_model.references.y_target
        pos_flow = vxm_model.references.pos_flow
        neg_flow = vxm_model.references.neg_flow

        # get mean stream of negative flow
        mean_stream = ne.layers.MeanStream(name='mean_stream', cap=mean_cap)(neg_flow)

        # initialize the keras model
        super().__init__(inputs=[atlas_input, source_input], outputs=[y_source, y_target, mean_stream, pos_flow])

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.atlas_layer = atlas_layer
        self.references.atlas_tensor = atlas_tensor


class ConditionalTemplateCreation(LoadableModel):
    """
    VoxelMorph network to generate an conditional template image.
    """

    @store_config_args
    def __init__(self,
        inshape,
        pheno_input_shape,
        nb_unet_features=None,
        src_feats=1,
        conv_image_shape=None,
        conv_size=3,
        conv_nb_levels=0,
        conv_nb_features=32,
        extra_conv_layers=3,
        use_mean_stream=True,
        mean_cap=100,
        templcondsi=False,
        templcondsi_init=None,
        **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            pheno_input_shape: Pheno data input shape. e.g. (2)
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            src_feats: Number of source (atlas) features. Default is 1.
            conv_image_shape: Intermediate phenotype image shape. Default is inshape with conv_nb_features.
            conv_size: Atlas generator convolutional kernel size. Default is 3.
            conv_nb_levels: Number of levels in atlas generator unet. Default is 0.
            conv_nb_features: Number of features in atlas generator convolutions. Default is 32.
            extra_conv_layers: Number of extra convolutions after unet in atlas generator. Default is 3.
            use_mean_stream: Return mean stream layer for training. Default is True.
            mean_cap: Cap for mean stream. Default is 100.
            templcondsi: Default is False.
            templcondsi_init: Default is None.
            kwargs: Forwarded to the internal VxmDense model.
        """

        if conv_image_shape is None:
            conv_image_shape = (*inshape, conv_nb_features)

        # build initial dense pheno to image shape model
        pheno_input = KL.Input(pheno_input_shape, name='pheno_input')
        pheno_dense = KL.Dense(np.prod(conv_image_shape), activation='elu')(pheno_input)
        pheno_reshaped = KL.Reshape(conv_image_shape, name='pheno_reshape')(pheno_dense)
        pheno_init_model = tf.keras.models.Model(pheno_input, pheno_reshaped)

        # build model to decode reshaped pheno
        pheno_decoder_model = ne.models.conv_dec(conv_nb_features, conv_image_shape, conv_nb_levels, conv_size,
                                                 nb_labels=conv_nb_features, final_pred_activation='linear',
                                                 input_model=pheno_init_model, name='atlas_decoder')

        # add extra convolutions
        Conv = getattr(KL, 'Conv%dD' % len(inshape))
        last = pheno_decoder_model.output
        for n in range(extra_conv_layers):
            last = Conv(conv_nb_features, kernel_size=conv_size, padding='same', name='atlas_extra_conv_%d' % n)(last)

        # final convolution to get atlas features
        atlas_gen = Conv(src_feats, kernel_size=3, padding='same', name='atlas_gen',
                         kernel_initializer=RandomNormal(mean=0.0, stddev=1e-7),
                         bias_initializer=RandomNormal(mean=0.0, stddev=1e-7))(last)

        # image input layers
        atlas_input = tf.keras.Input((*inshape, src_feats), name='atlas_input')
        source_input = tf.keras.Input((*inshape, src_feats), name='source_input')

        if templcondsi:
            atlas_tensor = KL.Add(name='atlas_tmp')([atlas_input, pout])
            # change first channel to be result from seg with another add layer
            tmp_layer = KL.Lambda(lambda x: K.softmax(x[..., 1:]))(atlas_tensor)
            conv_layer = Conv(1, kernel_size=1, padding='same', use_bias=False, name='atlas_gen', kernel_initializer=RandomNormal(mean=0, stddev=1e-5))
            x_img = conv_layer(tmp_layer)
            if templcondsi_init is not None:
                weights = conv_layer.get_weights()
                weights[0] = templcondsi_init.reshape(weights[0].shape)
                conv_layer.set_weights(weights)
            atlas_tensor = KL.Lambda(lambda x: K.concatenate([x[0], x[1][...,1:]]), name='atlas')([x_img, atlas_tensor])
        else:
            atlas_tensor = KL.Add(name='atlas')([atlas_input, atlas_gen])

        # build complete pheno to atlas model
        pheno_model = tf.keras.models.Model([pheno_decoder_model.input, atlas_input], atlas_tensor)

        inputs = [pheno_decoder_model.input, atlas_input, source_input]
        warp_input_model = tf.keras.Model(inputs=inputs, outputs=[atlas_tensor, source_input])

        # warp model
        vxm_model = VxmDense(inshape, nb_unet_features=nb_unet_features, bidir=True, input_model=warp_input_model, **kwargs)

        # extract tensors from stacked model
        y_source = vxm_model.references.y_source
        pos_flow = vxm_model.references.pos_flow
        neg_flow = vxm_model.references.neg_flow

        if use_mean_stream:
            # get mean stream from negative flow
            mean_stream = ne.layers.MeanStream(name='mean_stream', cap=mean_cap)(neg_flow)
            outputs = [y_source, mean_stream, pos_flow, pos_flow]
        else:
            outputs = [y_source, pos_flow, pos_flow]

        # initialize the keras model
        super().__init__(inputs=inputs, outputs=outputs)


class VxmDenseSegSemiSupervised(LoadableModel):
    """
    VoxelMorph network for (semi-supervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self, inshape, nb_labels, nb_unet_features=None, int_steps=7, int_downsize=2, seg_downsize=2):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_labels: Number of labels used for ground truth segmentations.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            seg_downsize: Interger specifying the downsampled factor of the segmentations. Default is 2.
        """

        # configure base voxelmorph network
        vxm_model = VxmDense(inshape, nb_unet_features=nb_unet_features, int_steps=int_steps, int_downsize=int_downsize)

        # configure downsampled seg input layer
        inshape_downsized = (np.array(inshape) / seg_downsize).astype(int)
        seg_src = tf.keras.Input(shape=(*inshape_downsized, nb_labels))

        # configure warped seg output layer
        seg_flow = layers.RescaleTransform(1 / seg_downsize, name='seg_resize')(vxm_model.references.pos_flow)
        y_seg = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='seg_transformer')([seg_src, seg_flow])

        # initialize the keras model
        inputs = vxm_model.inputs + [seg_src]
        outputs = vxm_model.outputs + [y_seg]
        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.pos_flow = vxm_model.references.pos_flow

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineSegSemiSupervised(LoadableModel):
    """
    VoxelMorph network for (semi-supervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self, inshape, enc_nf, nb_labels, int_downsize=2, seg_downsize=2, **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            nb_labels: Number of labels used for ground truth segmentations.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            seg_downsize: Interger specifying the downsampled factor of the segmentations. Default is 2.
            kwargs: Forwarded to the internal VxmAffine model.
        """

        # configure base voxelmorph network
        vxm_model = VxmAffine(inshape, enc_nf, **kwargs)

        # configure downsampled seg input layer
        inshape_downsized = (np.array(inshape) / seg_downsize).astype(int)
        seg_src = tf.keras.Input(shape=(*inshape_downsized, nb_labels))

        # configure warped seg output layer
        seg_transformer_layer = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='seg_transformer')
        if seg_downsize > 1:
            # TODO: this fails, not sure why (BRF)
            seg_flow = layers.RescaleTransform(1 / seg_downsize, name='seg_resize')(vxm_model.references.affine)
            y_seg = seg_transformer_layer([seg_src, seg_flow])
        else:
            y_seg = seg_transformer_layer([seg_src, vxm_model.references.affine])

        # initialize the keras model
        inputs = vxm_model.inputs + [seg_src]
        outputs = vxm_model.outputs + [y_seg]
        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.affine = vxm_model.references.affine

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.affine)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmDenseSurfaceSemiSupervised(LoadableModel):
    """
    VoxelMorph network for semi-supervised nonlinear registration aided by surface point registration.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_surface_points,
        nb_labels_sample,
        nb_unet_features=None,
        sdt_vol_resize=1,
        surf_bidir=True,
        **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_surface_points: Number of surface points to warp.
            nb_labels_sample: Number of labels to sample.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            sdt_vol_resize: Resize factor of distance transform. Default is 1.
            surf_bidir: Train with bidirectional surface warping. Default is True.
            kwargs: Forwarded to the internal VxmDense model.
        """

        sdt_shape = [int(f * sdt_vol_resize) for f in inshape]
        surface_points_shape = [nb_surface_points, len(inshape) + 1]
        single_pt_trf = lambda x: point_spatial_transformer(x, sdt_vol_resize=sdt_vol_resize)

        # vxm model
        vxm_model = VxmDense(inshape, nb_unet_features=nb_unet_features, bidir=True, **kwargs)
        pos_flow = vxm_model.references.pos_flow
        neg_flow = vxm_model.references.neg_flow

        # surface inputs and invert atlas_v for inverse transform to get final 'atlas surface'
        atl_surf_input = tf.keras.Input(surface_points_shape, name='atl_surface_input')

        # warp atlas surface
        # NOTE: pos diffflow is used to define an image moving x --> A, but when moving points, it moves A --> x
        warped_atl_surf_pts = Lambda(single_pt_trf, name='warped_atl_surface')([atl_surf_input, pos_flow])

        # get value of dt_input *at* warped_atlas_surface
        subj_dt_input = tf.keras.Input([*sdt_shape, nb_labels_sample], name='subj_dt_input')
        subj_dt_value = Lambda(value_at_location, name='hausdorff_subj_dt')([subj_dt_input, warped_atl_surf_pts])

        if surf_bidir:
            # go the other way and warp subject to atlas
            subj_surf_input = tf.keras.Input(surface_points_shape, name='subj_surface_input')
            warped_subj_surf_pts = Lambda(single_pt_trf, name='warped_subj_surface')([subj_surf_input, neg_flow])

            atl_dt_input = tf.keras.Input([*sdt_shape, nb_labels_sample], name='atl_dt_input')
            atl_dt_value = Lambda(value_at_location, name='hausdorff_atl_dt')([atl_dt_input, warped_subj_surf_pts])

            inputs  = [*vxm_model.inputs, subj_dt_input, atl_dt_input, subj_surf_input, atl_surf_input]
            outputs = [*vxm_model.outputs, subj_dt_value, atl_dt_value]

        else:
            inputs  = [*vxm_model.inputs, subj_dt_input, atl_surf_input]
            outputs = [*vxm_model.outputs, subj_dt_value]

        # initialize the keras model
        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.pos_flow = pos_flow

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineSurfaceSemiSupervised(LoadableModel):
    """
    VoxelMorph network for semi-supervised nonlinear registration aided by surface point registration.
    """

    @store_config_args
    def __init__(self,
        inshape,
        enc_nf,
        nb_surface_points,
        nb_labels_sample,
        sdt_vol_resize=1,
        surf_bidir=True,
        **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            nb_surface_points: Number of surface points to warp.
            nb_labels_sample: Number of labels to sample.
            sdt_vol_resize: Resize factor of distance transform. Default is 1.
            surf_bidir: Train with bidirectional surface warping. Default is True.
            kwargs: Forwarded to the internal VxmAffine model.
        """

        sdt_shape = [int(f * sdt_vol_resize) for f in inshape]
        surface_points_shape = [nb_surface_points, len(inshape) + 1]
        single_pt_trf = lambda x: point_spatial_transformer(x, sdt_vol_resize=sdt_vol_resize)

        # vm model
        affine_model = VxmAffine(inshape, enc_nf, **kwargs)
        affine_tensor = affine_model.references.affine
        pos_flow = layers.AffineToDense(inshape, name='affine_to_flow')(affine_tensor)
        
        # invert
        inverse_affine = layers.InvertAffine(name='affine_invert')(affine_tensor)
        neg_flow = layers.AffineToDense(inshape, name='neg_affine_to_flow')(inverse_affine)

        # surface inputs and invert atlas_v for inverse transform to get final 'atlas surface'
        atl_surf_input = tf.keras.Input(surface_points_shape, name='atl_surface_input')

        # warp atlas surface
        # NOTE: pos diffflow is used to define an image moving x --> A, but when moving points, it moves A --> x
        warped_atl_surf_pts = Lambda(single_pt_trf, name='warped_atl_surface')([atl_surf_input, pos_flow])

        # get value of dt_input *at* warped_atlas_surface
        subj_dt_input = tf.keras.Input([*sdt_shape, nb_labels_sample], name='subj_dt_input')
        subj_dt_value = Lambda(value_at_location, name='hausdorff_subj_dt')([subj_dt_input, warped_atl_surf_pts])

        if surf_bidir:
            # go the other way and warp subject to atlas
            subj_surf_input = tf.keras.Input(surface_points_shape, name='subj_surface_input')
            warped_subj_surf_pts = Lambda(single_pt_trf, name='warped_subj_surface')([subj_surf_input, neg_flow])

            atl_dt_input = tf.keras.Input([*sdt_shape, nb_labels_sample], name='atl_dt_input')
            atl_dt_value = Lambda(value_at_location, name='hausdorff_atl_dt')([atl_dt_input, warped_subj_surf_pts])

            inputs  = [*affine_model.inputs, subj_dt_input, atl_dt_input, subj_surf_input, atl_surf_input]
            outputs = [*affine_model.outputs, subj_dt_value, atl_dt_value]

        else:
            inputs  = [*affine_model.inputs, subj_dt_input, atl_surf_input]
            outputs = [*affine_model.outputs, subj_dt_value]

        # initialize the keras model
        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.affine = affine_tensor
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.affine)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmSynthetic(LoadableModel):
    """
    VoxelMorph network for registering segmentations.
    """

    @store_config_args
    def __init__(self, inshape, all_labels, hot_labels, nb_unet_features=None, int_steps=5, **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            all_labels: List of all labels included in training segmentations.
            hot_labels: List of labels to output as one-hot maps.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            kwargs: Forwarded to the internal VxmAffine model.
        """
        from SynthSeg.labels_to_image_model import labels_to_image_model

        # brain generation
        make_im_model = lambda id: labels_to_image_model(inshape, inshape, all_labels, hot_labels, id=id,
            apply_affine_trans=False, apply_nonlin_trans=True, nonlin_shape_factor=0.0625, bias_shape_factor=0.025
        )
        bg_model_1, self.warp_shape, self.bias_shape = make_im_model(0)
        bg_model_2 = make_im_model(1)[0]
        image_1, labels_1 = bg_model_1.outputs[:2]
        image_2, labels_2 = bg_model_2.outputs[:2]

        # build brain generation input model
        inputs = bg_model_1.inputs + bg_model_2.inputs
        unet_input_model = tf.keras.Model(inputs=inputs, outputs=[image_1, image_2])

        # attach dense voxelmorph network and extract flow field layer
        dense_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            int_steps=int_steps,
            input_model=unet_input_model,
            **kwargs
        )
        flow = dense_model.references.pos_flow

        # one-hot encoding
        one_hot_func = lambda x: tf.one_hot(x[..., 0], len(hot_labels), dtype='float32')
        one_hot_1 = KL.Lambda(one_hot_func)(labels_1)
        one_hot_2 = KL.Lambda(one_hot_func)(labels_2)

        # transformation
        pred = layers.SpatialTransformer(interp_method='linear', name='transformer')([one_hot_1, flow])
        concat = KL.Concatenate(axis=-1, name='concat')([one_hot_2, pred])

        # initialize the keras model
        super().__init__(name='vxm_synth', inputs=inputs, outputs=[concat, flow])


class Transform(tf.keras.Model):
    """
    Simple transform model to apply dense or affine transforms.
    """

    def __init__(self, inshape, affine=False, interp_method='linear', nb_feats=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            affine: Enable affine transform. Default is False.
            interp_method: Interpolation method. Can be 'linear' or 'nearest'. Default is 'linear'.
            nb_feats: Number of source image features. Default is 1.
        """

        # configure inputs
        ndims = len(inshape)
        scan_input = tf.keras.Input((*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = tf.keras.Input((ndims * (ndims + 1),), name='trf_input')
        else:
            trf_input = tf.keras.Input((*inshape, ndims), name='trf_input')

        # transform and initialize the keras model
        y_source = layers.SpatialTransformer(interp_method=interp_method, name='transformer')([scan_input, trf_input])
        super().__init__(inputs=[scan_input, trf_input], outputs=y_source)


def conv_block(x, nfeat, strides=1, name=None):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(x)
    name = name + '_activation' if name else None
    return LeakyReLU(0.2, name=name)(convolved)


def upsample_block(x, connection, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)
    
    upsampled = UpSampling(name=name)(x)
    name = name + '_concat' if name else None
    return concatenate([upsampled, connection], name=name)


class Unet(tf.keras.Model):
    """
    A unet architecture that builds off of an input keras model. Layer features can be specified directly
    as a list of encoder and decoder features or as a single integer along with a number of unet levels.
    The default network features per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self, input_model, nb_features=None, nb_levels=None, feat_mult=1, nb_conv_per_level=1):
        """
        Parameters:
            input_model: Input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
        """

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        # configure encoder (down-sampling path)
        enc_layers = [tf.keras.layers.concatenate(input_model.outputs, name='unet_input_concat')]
        last = enc_layers[0]
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                strides = 2 if conv == (nb_conv_per_level - 1) else 1
                name = 'unet_enc_conv_%d_%d' % (level, conv)
                last = conv_block(last, nf, strides=strides, name=name)
            enc_layers.append(last)

        # configure decoder (up-sampling path)
        last = enc_layers.pop()
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                name = 'unet_dec_conv_%d_%d' % (real_level, conv)
                last = conv_block(last, nf, name=name)
            name = 'unet_dec_upsample_' + str(real_level)
            last = upsample_block(last, enc_layers.pop(), name=name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            name = 'unet_dec_final_conv_' + str(num)
            last = conv_block(last, nf, name=name)

        return super().__init__(inputs=input_model.inputs, outputs=last)
