"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    python register.py moving.nii.gz fixed.nii.gz moved.nii.gz --model model.h5 --save-warp warp.nii.gz

The source and target input images are expected to be affinely registered, but if not, an initial linear registration
can be run by specifing an affine model with the --affine-model flag, which expects an affine model file as input.
If an affine model is provided, it's assumed that it was trained with images padded to (256, 256, 256) and resized
by a factor of 0.25 (the default affine model configuration).
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('moving', help='moving image (source) filename')
parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('moved', help='registered image output filename')
parser.add_argument('--model', help='run nonlinear registration - must specify keras model file')
parser.add_argument('--affine-model', help='run intitial affine registration - must specify keras model file')
parser.add_argument('--save-warp', help='output warp filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# sanity check on the input
assert (args.model or args.affine_model), 'must provide at least a warp or affine model'

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

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_feat_axis=add_feat_axis, ret_affine=True)

if args.affine_model:

    # pad inputs to a standard size
    padshape = [256 for _ in moving.shape[:-1]] + [moving.shape[-1]]
    moving_padded, _ = vxm.py.utils.pad(moving, padshape)
    fixed_padded, cropping = vxm.py.utils.pad(fixed, padshape)

    # scale image sizes by some factor
    resize = 0.25
    moving_resized = vxm.py.utils.resize(moving_padded, resize)[np.newaxis]
    fixed_resized = vxm.py.utils.resize(fixed_padded, resize)[np.newaxis]

    with tf.device(device):
        # load the affine model, predict the transform(s), and merge
        affine = vxm.networks.VxmAffine.load(args.affine_model).register(moving_resized, fixed_resized)

        # apply the transform and crop back to the target space
        moving = moving_padded[np.newaxis]
        aligned = vxm.tf.utils.transform(moving, affine, rescale=(1 / resize))[0, ...]
        moved = aligned[cropping]

        # set as 'moving' for the following nonlinear registration
        moving = moved

if args.model:

    moving = moving[np.newaxis]
    fixed = fixed[np.newaxis]

    with tf.device(device):
        # load model and predict
        warp = vxm.networks.VxmDense.load(args.model).register(moving, fixed)
        moved = vxm.tf.utils.transform(moving, warp)

    # save warp
    if args.save_warp:
        vxm.py.utils.save_volfile(warp.squeeze(), args.save_warp, fixed_affine)

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)
