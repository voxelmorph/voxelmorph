"""
Example testing script for trained VoxelMorph models.

This script iterates over a list of images and corresponding segmentations, registers them to
an atlas, propagates segmentations to the atlas, and computes the dice overlap.
"""

import os
import argparse
import scipy
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import keras


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('config', help='model configuration')
parser.add_argument('weights', help='keras model weights')
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
args = parser.parse_args()

# list of test subject volumes and segmentations to evaluate
test_subjects = [('data/test_vol.npz', 'data/test_seg.npz')]

# corresponding seg labels
labels = scipy.io.loadmat('data/labels.mat')['labels'][0]

# load atlas volume and seg
atlas_vol = vxm.utils.load_volfile('data/atlas_norm.npz', np_var='vol', add_axes=True)
atlas_seg = vxm.utils.load_volfile('data/atlas_norm.npz', np_var='seg')
inshape = atlas_seg.shape

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

with tf.device(device):
    # load voxelmorph model and compose flow output model
    vxmnet = vxm.utils.NetConfig.read(args.config).build_model(args.weights)
    flownet = vxm.networks.build_warpnet(vxmnet)

    # build nearest-neighbor transfer model
    transform_model = vxm.networks.transform(inshape, interp_method='nearest')

for i, (vol_name, seg_name) in enumerate(test_subjects):

    # load subject
    moving_vol = vxm.utils.load_volfile(vol_name, add_axes=True)
    moving_seg = vxm.utils.load_volfile(seg_name, add_axes=True)

    # predict transform
    with tf.device(device):
        _, warp = warp_net.predict([moving_vol, atlas_vol])

    # warp segments with flow
    warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
    
    # compute volume overlap (dice)
    overlap = vxm.utils.dice(warped_seg, atlas_seg, labels=labels)
    print('subject %3d:   dice mean = %5.3f  std = %5.3f' % (i, np.mean(overlap), np.std(overlap)))
