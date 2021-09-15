#!/usr/bin/env python

"""
Example script for training a HyperMorph model to tune the
regularization weight hyperparameter.

If you use this code, please cite the following:

    A Hoopes, M Hoffmann, B Fischl, J Guttag, AV Dalca. 
    HyperMorph: Amortized Hyperparameter Learning for Image Registration
    IPMI: Information Processing in Medical Imaging. 2021. https://arxiv.org/abs/2101.01035

Copyright 2020 Andrew Hoopes

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from tensorflow.keras import backend as K


# disable eager execution
tf.compat.v1.disable_eager_execution()


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--test-reg', nargs=3,
                    help='example registration pair and result (moving fixed moved) to test')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=6000,
                    help='number of training epochs (default: 6000)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='steps per epoch (default: 100)')
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
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=0.2,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
args = parser.parse_args()

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    base_generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                                  batch_size=args.batch_size,
                                                  add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    base_generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, add_feat_axis=add_feat_axis)


# random hyperparameter generator
def random_hyperparam():
    if np.random.rand() < args.oversample_rate:
        return np.random.choice([0, 1])
    else:
        return np.random.rand()


# hyperparameter generator extension
def hyp_generator():
    while True:
        hyp = np.expand_dims([random_hyperparam() for _ in range(args.batch_size)], -1)
        inputs, outputs = next(base_generator)
        inputs = (*inputs, hyp)
        yield (inputs, outputs)


generator = hyp_generator()

# extract shape and number of features from sampled input
sample_shape = next(generator)[0][0].shape
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# build the model
model = vxm.networks.HyperVxmDense(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    int_steps=args.int_steps,
    int_resolution=args.int_downsize,
    src_feats=nfeats,
    trg_feats=nfeats,
    svf_resolution=2,
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    scaling = 1.0 / (args.image_sigma ** 2)
    image_loss_func = lambda x1, x2: scaling * K.mean(K.batch_flatten(K.square(x1 - x2)), -1)
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


# prepare loss functions and compile model
def image_loss(y_true, y_pred):
    hyp = (1 - tf.squeeze(model.references.hyper_val))
    return hyp * image_loss_func(y_true, y_pred)


def grad_loss(y_true, y_pred):
    hyp = tf.squeeze(model.references.hyper_val)
    return hyp * vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss(y_true, y_pred)


model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=[image_loss, grad_loss])

save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=100)

model.fit_generator(generator,
                    initial_epoch=args.initial_epoch,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    callbacks=[save_callback],
                    verbose=1)

# save final weights
model.save(save_filename.format(epoch=args.epochs))

# save an example registration across lambda values
if args.test_reg:
    moving = vxm.py.utils.load_volfile(args.test_reg[0], add_batch_axis=True,
                                       add_feat_axis=add_feat_axis)
    fixed = vxm.py.utils.load_volfile(args.test_reg[1], add_batch_axis=True,
                                      add_feat_axis=add_feat_axis)
    moved = []

    # sweep across 20 values of lambda
    for i, hyp in enumerate(np.linspace(0, 1, 20)):
        hyp = np.array([[hyp]], dtype='float32')  # reformat hyperparam
        img = model.predict([moving, fixed, hyp])[0].squeeze()
        moved.append(img)

    moved = np.stack(moved, axis=-1)
    if moved.ndim == 3:
        moved = np.expand_dims(moved, axis=-2)  # if 2D, ensure multi-frame nifti
    vxm.py.utils.save_volfile(moved, args.test_reg[2])
