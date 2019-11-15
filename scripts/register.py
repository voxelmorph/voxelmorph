"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
subject (moving) to an atlas (fixed). To register a subject to the atlas and save the warp field, run:

    python register.py <moving> <atlas> <model> -o <warped-image> -w <warp>

For example, our test volume can be warped to the atlas with the cvpr2018-trained model by running:

    python register.py data/test_vol.nii.gz data/atlas_norm.nii.gz models/cvpr2018_vm2_cc.h5 -o data/test_warped.nii.gz

The warped input will be saved to data/test_warped.nii.gz
"""

import os
import argparse
import numpy as np
import nibabel as nib
import voxelmorph as vxm
import tensorflow as tf
import keras


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('moving', help='moving image (source) filename')
parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('model', help='keras model filename')
parser.add_argument('-g', '--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('-o', '--out-image', help='warped output image filename')
parser.add_argument('-w', '--warp', help='output warp filename')
args = parser.parse_args()

# device handling
if args.gpu:
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'

# load moving image
moving_image = nib.load(args.moving)
moving = moving_image.get_data()[np.newaxis, ..., np.newaxis]
affine = moving_image.affine

# load fixed image
fixed_image = nib.load(args.fixed)
fixed = fixed_image.get_data()[np.newaxis, ..., np.newaxis]

# load model and predict
with tf.device(device):
    # load voxelmorph model and compose flow output model
    net = vxm.networks.load_model(args.model)
    flownet = vxm.networks.compose_flownet(net)

    # predict warp
    moved, warp = flownet.predict([moving, fixed])

# save warped image
if args.out_image:
    img = nib.Nifti1Image(moved.squeeze(), moving_image.affine)
    nib.save(img, args.out_image)

# save warp
if args.warp:
    img = nib.Nifti1Image(warp.squeeze(), moving_image.affine)
    nib.save(img, args.warp)
