#!/usr/bin/env python

"""
Example script to apply a deformation to an image. Usage is:

    warp.py --moving moving.nii.gz --warp warp.nii.gz --moved moved.nii.gz

Interpolation method can be specified with the --interp flag.
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import keras


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image filename')
parser.add_argument('--warp', required=True, help='warp image filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--interp', default='linear', help='interpolation method linear/nearest (default: linear)')
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# load moving image and deformation field
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
deform, deform_affine = vxm.py.utils.load_volfile(args.warp, add_batch_axis=True, ret_affine=add_feat_axis)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# build transfer model and warp
with tf.device(device):
    moved = vxm.networks.Transform(moving.shape[1:-1], interp_method=args.interp, nb_feats=moving.shape[-1]).predict([moving, deform])

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, deform_affine)
