#!/usr/bin/env python

"""
Instance-specific optimization

If you use this code, please cite the following 
    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces 
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

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
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='registered image output filename')
parser.add_argument('--model', help='initialize with prediction from pretrained vxm model')
parser.add_argument('--warp', help='output warp filename')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--steps', type=int, default=200, help='num training steps (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

# network architecture parameters
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--multiplier', type=float, default=1000,
                    help='local weight multiplier (default: 1000)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                    help='weight of gradient loss (default: 0.01)')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# initialize instance network
inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]
model = vxm.networks.InstanceDense(
    inshape,
    nb_feats=nb_feats,
    mult=args.multiplier,
    int_steps=args.int_steps,
    int_resolution=args.int_downsize
)

# load model and predict
if args.model is not None:
    initialization = vxm.networks.VxmDense.load(args.model).register(moving, fixed)
    model.set_flow(initialization)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights = [1, args.lambda_weight]

# train
zeros = np.zeros((1, *inshape, len(inshape)), dtype='float32')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)
model.fit(
    [moving],
    [fixed, zeros],
    batch_size=None,
    epochs=args.steps,
    steps_per_epoch=1,
    verbose=1
)

# get warped image and deformation field
warp = model.register(moving)
moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)
