import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import OrderedDict
# import neurite as ne

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
import neurite.torch.layers as ne_layers


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 hyp_num_outputs=None,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

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
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf,
                                       hyp_num_outputs=hyp_num_outputs))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf,
                                       hyp_num_outputs=hyp_num_outputs))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf,
                                            hyp_num_outputs=hyp_num_outputs))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x, *args):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x, *args)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x, *args)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x, *args)

        return x


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
                 unet_half_res=False,
                 hyp_model=None):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        hyp_num_outputs = None
        if hyp_model is not None:
            hyp_num_outputs = list(hyp_model.modules())[-2].out_features

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            hyp_num_outputs=hyp_num_outputs,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, hyp_output=None, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            hyp_output: Output tensor from hypernetwork (Hypermorph models only)
            registration: Return transformed image and flow. Default is False.
        '''

        hyp_outputs_args = [hyp_output] if hyp_output is not None else []

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x, *hyp_outputs_args)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class HyperVxmDense(LoadableModel):
    """
    Dense HyperMorph network for amortized hyperparameter learning.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_hyp_params=1,
                 nb_hyp_layers=6,
                 nb_hyp_units=128,
                 name='hyper_vxm_dense',

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
                 unet_half_res=False,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_hyp_params: Number of input hyperparameters.
            nb_hyp_layers: Number of dense layers in the hypernetwork.
            nb_hyp_units: Number of units in each dense layer of the hypernetwork.
            # name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
            # kwargs: Forwarded to the internal VxmDense model.
        """

        super().__init__()

        hyp_layers = OrderedDict()
        n_last = nb_hyp_params
        for n in range(nb_hyp_layers):
            hyp_layers[f'{name}_hyp_dense_{n + 1}_lin'] = nn.Linear(n_last,
                                                                    nb_hyp_units)
            hyp_layers[f'{name}_hyp_dense_{n + 1}_relu'] = nn.ReLU()
            n_last = nb_hyp_units
        self.hyp_model = nn.Sequential(hyp_layers)

        self.int_downsize = int_downsize
        self.bidir = bidir

        # VxmDense_model = VxmDense_legacy if legacy else VxmDense
        # if legacy:
        #     include_masks_in_input = False


        vxm_model = VxmDense(inshape=inshape,
                             nb_unet_features=nb_unet_features,
                             nb_unet_levels=nb_unet_levels,
                             unet_feat_mult=unet_feat_mult,
                             nb_unet_conv_per_level=nb_unet_conv_per_level,
                             int_steps=int_steps,
                             int_downsize=int_downsize,
                             bidir=bidir,
                             use_probs=use_probs,
                             src_feats=src_feats,
                             trg_feats=trg_feats,
                             unet_half_res=unet_half_res,
                             hyp_model=self.hyp_model)

        self.vxm_model = vxm_model


    def forward(self, source, target, hyp_input: torch.Tensor, registration=False):

        hyp_output = self.hyp_model(hyp_input)

        vxm_outputs = self.vxm_model.forward(source, target,
                                             hyp_output, registration=registration)

        return vxm_outputs





class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by LeakyReLU for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1,
                 hyp_num_outputs=None):
        super().__init__()

        if hyp_num_outputs is not None:
            # Use HyperLayers
            Conv = getattr(ne_layers, 'HyperConv%ddFromDense' % ndims)
            n_hyp_inputs_kwargs = {'hyp_inputs': hyp_num_outputs}
        else:
            # Use Standard conv layers
            Conv = getattr(nn, 'Conv%dd' % ndims)
            n_hyp_inputs_kwargs = {}

        self.main = Conv(**n_hyp_inputs_kwargs,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         stride=stride,
                         padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, *args):
        # for hypernetworks, *args contains the output tensor from the HyperNetwork
        out = self.main(x, *args)
        out = self.activation(out)
        return out
