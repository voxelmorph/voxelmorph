#!/usr/bin/env python

"""
Example script for testing quality of trained VxmDense models. This script iterates over a list of
images pairs, registers them, propagates segmentations via the deformation, and computes the dice
overlap. Example usage is:

    test.py  \
        --model model.h5  \
        --pairs pairs.txt  \
        --img-suffix /img.nii.gz  \
        --seg-suffix /seg.nii.gz

Where pairs.txt is a text file with line-by-line space-seperated registration pairs.
This script will most likely need to be customized to fit your data.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import time
import numpy as np
import voxelmorph as vxm
import tensorflow as tf


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--model', required=True, help='VxmDense model file')
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# sanity check on input pairs
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load seg labels if provided
labels = np.load(args.labels) if args.labels else None

# check if multi-channel data
add_feat_axis = not args.multichannel

# keep track of all dice scores
reg_times = []
dice_means = []

with tf.device(device):

    # load model and build nearest-neighbor transfer model
    model = vxm.networks.VxmDense.load(args.model, input_model=None)
    registration_model = model.get_registration_model()
    inshape = registration_model.inputs[0].shape[1:-1]
    transform_model = vxm.networks.Transform(inshape, interp_method='nearest')

    for i in range(len(img_pairs)):

        # load moving image and seg
        moving_vol = vxm.py.utils.load_volfile(
            img_pairs[i][0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][0], np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)

        # load fixed image and seg
        fixed_vol = vxm.py.utils.load_volfile(
            img_pairs[i][1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][1], np_var='seg')

        # predict warp and time
        start = time.time()
        warp = registration_model.predict([moving_vol, fixed_vol])
        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)

        # apply transform
        warped_seg = transform_model.predict([moving_seg, warp]).squeeze()

        # compute volume overlap (dice)
        overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=labels)
        dice_means.append(np.mean(overlap))
        print('Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (i + 1, reg_time,
                                                                    np.mean(overlap),
                                                                    np.std(overlap)))

print()
print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                    np.std(reg_times)))
print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))
