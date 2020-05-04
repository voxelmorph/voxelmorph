"""
Example testing script for trained VoxelMorph models. This script iterates over a list of
images and corresponding segmentations, registers them to an atlas, propagates segmentations
to the atlas, and computes the dice overlap. Usage is:

    python test.py --model model.h5 --atlas data/atlas.npz --scans data/test_scan.npz --labels data/labels.npz

Where each atlas and scan npz file is assumed to contain the array variables 'vol' and 'seg'.
This script will most likely need to be customized to fit your data.
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='keras warp model file')
parser.add_argument('--atlas', required=True, help='atlas npz file')
parser.add_argument('--scans', nargs='+', required=True, help='test scan npz files')
parser.add_argument('--labels', required=True, help='label lookup file in npz format')
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# corresponding seg labels
labels = np.load(args.labels)['labels']

# load atlas volume and seg
add_feat_axis = not args.multichannel
atlas_vol = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
atlas_seg = vxm.py.utils.load_volfile(args.atlas, np_var='seg')
inshape = atlas_seg.shape

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
    # load warp model and build nearest-neighbor transfer model
    warp_predictor = vxm.networks.VxmDense.load(args.model).get_predictor_model()
    transform_model = vxm.networks.Transform(inshape, interp_method='nearest')

for i, scan in enumerate(args.scans):

    # load scan
    moving_vol = vxm.py.utils.load_volfile(scan, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = vxm.py.utils.load_volfile(scan, np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)

    # predict transform
    with tf.device(device):
        _, warp = warp_predictor.predict([moving_vol, atlas_vol])

    # warp segments with flow
    warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
    
    # compute volume overlap (dice)
    overlap = vxm.py.utils.dice(warped_seg, atlas_seg, labels=labels)
    print('scan %3d:   dice mean = %5.3f  std = %5.3f' % (i, np.mean(overlap), np.std(overlap)))
