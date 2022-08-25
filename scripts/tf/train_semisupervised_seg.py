#!/usr/bin/env python

"""
Example script to train a VoxelMorph model in a semi-supervised
fashion by providing ground-truth segmentation data for training images.

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
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph_custom as vxm


# disable eager execution
tf.compat.v1.disable_eager_execution()


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', required=True, help='label list (npy format) to use in dice loss')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--atlas', help='optional atlas to perform scan-to-atlas training')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--grad-loss-weight', type=float, default=0.01,
                    help='weight of gradient loss (lamba) (default: 0.01)')
parser.add_argument('--dice-loss-weight', type=float, default=0.01,
                    help='weight of dice loss (gamma) (default: 0.01)')
args = parser.parse_args()


# sanity check on inputs
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)
train_imgs = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                         suffix=args.img_suffix)
train_segs = vxm.py.utils.read_file_list(args.img_list, prefix=args.seg_prefix,
                                         suffix=args.seg_suffix)
assert len(train_imgs) > 0, 'Could not find any training data.'

# load labels file
train_labels = np.load(args.labels)

# generator (scan-to-scan unless the atlas cmd argument was provided)

generator = vxm.generators.semisupervised(
    train_imgs, train_segs, labels=train_labels, atlas_file=args.atlas)


#       invols  = [src_vol, trg_vol, src_seg]
#       outvols = [trg_vol, zeros  , trg_seg]
#       yield (invols, outvols)
#       next(generator)[0][0] := src_vol

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1] # (256, 512, 64)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# build the model
model = vxm.networks.VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    nb_labels=len(train_labels),
    int_steps=args.int_steps,
    int_resolution=args.int_downsize
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# losses
losses = [image_loss_func, vxm.losses.Grad(
    'l2', loss_mult=args.int_downsize).loss, vxm.losses.Dice().loss]
weights = [1, args.grad_loss_weight, args.dice_loss_weight]

# multi-gpu support
nb_devices = len(args.gpu.split(','))
if nb_devices > 1:
    save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
    model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
else:
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=20)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

# save starting weights
model.save(save_filename.format(epoch=args.initial_epoch))

model.fit_generator(generator,
                    initial_epoch=args.initial_epoch,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    callbacks=[save_callback],
                    verbose=1
                    )
