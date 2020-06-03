"""
Example script to apply a deformation to an image. Usage is:

python register.py moving.nii.gz warp.nii.gz moved.nii.gz

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
parser.add_argument('moving', help='moving image filename')
parser.add_argument('warp', help='warp image filename')
parser.add_argument('moved', help='registered image output filename')
parser.add_argument('--interp', default='linear', help='interpolation method linear/nearest (default: linear)')
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# load moving image and deformation field
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
deform, deform_affine = vxm.py.utils.load_volfile(args.warp, add_batch_axis=True, ret_affine=add_feat_axis)

# device handling
if args.gpu and (args.gpu != '-1'):
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with tf.device(device):

    # build transfer model and warp
    transform_model = vxm.networks.Transform(moving.shape[1:-1], interp_method=args.interp, nb_feats=moving.shape[-1])
    moved = transform_model.predict([moving, deform])

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, deform_affine)
