import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
from .utils import *
from .. import py


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
                 outfeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
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
        assert ndims in [
            1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    'must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult **
                             np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError(
                'cannot use nb_levels if nb_features is not an integer')

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
        self.upsampling = [nn.Upsample(
            scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
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
                convs.append(ConvBlock(ndims, prev_nf, nf, normalize=True))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf, normalize=False))
            prev_nf = nf

        # cache final number of features
        self.final_nf = outfeats

    def forward(self, x):

        # encoder forward pass

        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            # print(f"Mona-13: original x shape {x.shape}")
            x = self.pooling[level](x)
        # print(f"Mona-12: original x shape {x.shape} and length {len(x_history)}")
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                # print(f"Mona-9: {level} - original x shape {x.shape} and x_history shape {(x_history[0]).shape}")
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        if self.remaining:
            for conv in self.remaining:
                x = conv(x)

        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel_size=3, stride=1, normalize=True):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, 1)
        self.normalize = normalize
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        if self.normalize:
            out = self.norm(out)
        out = self.activation(out)
        return out


def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation

    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f"Conv{ndim}d")(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      )
    return y


class GroupVxmDenseBspline(LoadableModel):
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
                 cps=4,
                 svf=True,
                 svf_steps=7,
                 svf_scale=1,
                 resize_channels=[32, 20],
                 method='avg'
                 ):
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
            cps: Number of control points for dense bspline transform. Default is 4.
            svf: Enable smooth vector field (SVF) regularization. Default is True.
            svf_steps: Number of SVF integration steps. Default is 7.
            svf_scale: SVF regularization scale. Default is 1.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        self.method = method

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [
            1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        Conv = getattr(nn, 'Conv%dd' % ndims)

        # determine and set output control point size from image size and control point spacing
        img_size = param_ndim_setup(inshape, ndims)
        cps = param_ndim_setup(cps, ndims)
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(
                    f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])

        # configure core unet model
        self.n = src_feats
        self.dims = inshape

        self.unet_model = Unet(
            inshape,
            infeats=src_feats,
            outfeats=trg_feats,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
                # Network:
        # encoder: same u-net encoder
        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        enc_channels = nb_unet_features[0]
        dec_channels = nb_unet_features[1]
        n_dec = min(len(self.unet_model.decoder), len(self.unet_model.encoder))
        num_dec_layers = n_dec - int(math.ceil(math.log2(min(cps))))


        self.unet_model.decoder = self.unet_model.decoder[:num_dec_layers]
        # delete the u-net final remaining cov layers
        self.unet_model.remaining = None
        # self.unet_model.remaining = nn.ModuleList()
        # prev_nf = enc_channels[num_dec_layers] + dec_channels[num_dec_layers]
        # self.unet_model.remaining.append(ConvBlock(ndims, prev_nf, trg_feats, kernel_size=1, normalize=False))

        # conv layers following resizing
        
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    in_ch = dec_channels[num_dec_layers -
                                         1] + enc_channels[-num_dec_layers]
                else:
                    in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndims, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # configure unet to flow field layer

        self.flow = Conv(resize_channels[-1], self.unet_model.final_nf,
                         kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
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
        self.integrate = layers.VecInt(
            down_shape, int_steps) if int_steps > 0 else None
        self.integrate = convNd(ndims, resize_channels[-1], ndims)

        # configure transformer
        self.transformer = layers.SpatialTransformerBspline(
            ndims, inshape, cps=cps, svf=svf, svf_steps=svf_steps, svf_scale=svf_scale)
    
    
    def forward(self, source, registration=False):
        """

        Args:
            source (_type_): (#batch, N, *size)
            registration (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        x = self.unet_model(source)
        

        # resize output of encoder-decoder
        x = interpolate_(x, size=self.output_size)
        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        dims = pos_flow.shape[2:]
        pos_flow = torch.reshape(pos_flow, (2, self.n, *dims)).transpose(0, 1)
        # warp image with flow field
        warp_img, pos_flow_svf = self.transformer(source.transpose(0, 1), pos_flow)

        template = update_atlas(warp_img, self.method)
        # return non-integrated flow field if training
        if not registration:
            return (warp_img, template, pos_flow_svf)
        else:
            return warp_img, pos_flow_svf
