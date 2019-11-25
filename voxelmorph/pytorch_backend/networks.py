import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import utils
from . import layers


class unet(nn.Module):
    """ 
    Unet architecture for the voxelmorph models.

    Parameters:
        inshape: Input shape. e.g. (256, 256, 256)
        enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
        dec_nf: List of decoder filters. e.g. [32, 32, 32, 32, 8, 8]
    """

    def __init__(self, inshape, enc_nf, dec_nf):
        super(unet, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in enc_nf:
            self.downarm.append(conv_block(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(dec_nf[:len(enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(conv_block(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(conv_block(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class vxm_net(nn.Module):
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
    """

    def __init__(self, inshape, enc_nf, dec_nf, int_steps=7, int_downsize=2, bidir=False, use_probs=False):
        super(vxm_net, self).__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = unet(inshape, enc_nf, dec_nf)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target):
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

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
        if self.training:
            return (y_source, y_target, flow_field) if self.bidir else (y_source, flow_field)
        else:
            return y_source, pos_flow

    def warp(self, source, target):
        """
        Function to run the model and return the warped image and final diffeomorphic warp.
        """
        self.training = False
        moved, warp = self(source, target)
        self.training = True
        return moved, warp


class conv_block(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super(conv_block, self).__init__()

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise ValueError('stride must be 1 or 2')

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
