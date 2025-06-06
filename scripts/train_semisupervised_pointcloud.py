#!/usr/bin/env python

"""
Example script for training semi-supervised nonlinear registration aided by surface point clouds
generated from segmentations.

If you use this code, please cite the following: Unsupervised Learning for Probabilistic
    Diffeomorphic Registration for Images and Surfaces
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
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# disable eager execution
tf.compat.v1.disable_eager_execution()


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', required=True, help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--smooth-seg', type=float, default=0.1,
                    help='segmentation smoothness sigma (default: 0.1)')
parser.add_argument('--labels', type=int, nargs='+', default=None,
                    help='labels to include - by default all labels in the atlas seg are used')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
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
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--surf-points', type=int, default=5000,
                    help='number of surface points to warp (default: 5000)')
parser.add_argument('--surf-bidir', action='store_true',
                    help='enable surface-based bidirectional cost function')
parser.add_argument('--sdt-resize', type=float, default=1.0,
                    help='resize factor for distance transform (default: 1.0)')
parser.add_argument('--num-labels', type=float, help='number of labels to sample (default: all)')
parser.add_argument('--align-segs', action='store_true', help='only align segmentations')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                    help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--dt-sigma', type=float, default=1.0,
                    help='surface noise parameter (default: 1.0)')
parser.add_argument('--kl-lambda', type=float, default=10,
                    help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')  # nopep8
args = parser.parse_args()

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

atlas_vol = vxm.py.utils.load_volfile(args.atlas, np_var='vol')
atlas_seg = vxm.py.utils.load_volfile(args.atlas, np_var='seg')

# get labels and number of labels to sample
labels = args.labels if args.labels is not None else np.sort(np.unique(atlas_seg))[1:]
num_labels = args.num_labels if args.num_labels is not None else len(labels)

# scan-to-atlas sdt generator
generator = vxm.generators.surf_semisupervised(
    train_files,
    atlas_vol,
    atlas_seg,
    nb_surface_pts=args.surf_points,
    labels=labels,
    batch_size=args.batch_size,
    surf_bidir=args.surf_bidir,
    smooth_seg_std=args.smooth_seg,
    nb_labels_sample=num_labels,
    sdt_vol_resize=args.sdt_resize,
    align_segs=args.align_segs,
    add_feat_axis=add_feat_axis
)

# extract shape and number of features from atlas
inshape = atlas_seg.shape
nfeats = 1 if not args.multichannel else atlas_vol[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
assert np.mod(args.batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# build the model
model = vxm.networks.VxmDenseSemiSupervisedPointCloud(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    nb_surface_points=args.surf_points,
    nb_labels_sample=num_labels,
    sdt_vol_resize=args.sdt_resize,
    surf_bidir=args.surf_bidir,
    use_probs=args.use_probs,
    int_steps=args.int_steps,
    int_resolution=args.int_downsize,
    src_feats=nfeats,
    trg_feats=nfeats
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE(args.image_sigma).loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# base dense network is bidirectional
losses = [image_loss_func, image_loss_func]
weights = [0.5, 0.5]

# prepare deformation loss
if args.use_probs:
    flow_shape = model.outputs[-1].shape[1:-1]
    losses += [vxm.losses.KL(args.kl_lambda, flow_shape).loss]
else:
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.lambda_weight]

# prepare sdt loss
nb_dst_outputs = 2 if args.surf_bidir else 1
losses += [vxm.losses.MSE().loss] * nb_dst_outputs
weights += [0.25 / (args.dt_sigma**2)] * nb_dst_outputs

# multi-gpu support
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
