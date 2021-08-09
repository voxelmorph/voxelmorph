#!/usr/bin/env python

"""
Example script to train conditional template creation. This code is still experimental based on the
experiments run in the preprint.

If you use this code, please cite the following 
    Learning Conditional Deformable Templates with Convolutional Networks 
    Adrian V. Dalca, Marianne Rakic, John Guttag, Mert R. Sabuncu 
    NeurIPS 2019. https://arxiv.org/abs/1908.02738

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
parser.add_argument('--pheno-csv', required=True, help='cvs file defining training data attributes')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

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

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-loss-weight', type=float, default=1.0,
                    help='relative weight of transformed atlas loss (default: 1.0)')
parser.add_argument('--mean-loss-weight', type=float, default=1.0,
                    help='weight of mean stream loss (default: 1.0)')
parser.add_argument('--grad-loss-weight', type=float, default=0.01,
                    help='weight of gradient loss (lamba) (default: 0.01)')
parser.add_argument('--deform-loss-weight', type=float, default=0.01,
                    help='weight of deformation MS loss (default: 0.01)')

args = parser.parse_args()


# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# load pheno attributes for the training data
train_vol_attributes, train_files = vxm.py.utils.load_pheno_csv(args.pheno_csv, train_files)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# prepare the initial weights for the template
if args.init_template:
    # load template from file
    template = vxm.py.utils.load_volfile(args.init_template,
                                         add_batch_axis=True, add_feat_axis=add_feat_axis)
else:
    # generate rough atlas by averaging inputs
    navgs = min((100, len(train_files)))
    print('Creating starting template by averaging first %d scans.' % navgs)
    template = 0
    for scan in train_files[:navgs]:
        template += vxm.py.utils.load_volfile(scan, add_batch_axis=True,
                                              add_feat_axis=add_feat_axis)
    template /= navgs

# save input template for the record
vxm.py.utils.save_volfile(template.squeeze(), os.path.join(model_dir, 'input_atlas.npz'))

# get template shape
inshape = template.shape[1:-1]
nfeats = template.shape[-1]
pheno_shape = list(train_vol_attributes.values())[0].shape

# configure generator
generator = vxm.generators.conditional_template_creation(train_files, template,
                                                         train_vol_attributes,
                                                         batch_size=args.batch_size,
                                                         add_feat_axis=add_feat_axis)

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
model = vxm.networks.ConditionalTemplateCreation(
    inshape,
    pheno_input_shape=pheno_shape,
    nb_unet_features=[enc_nf, dec_nf],
    conv_nb_features=4,
    conv_nb_levels=0,
    extra_conv_layers=3,
    src_feats=nfeats,
    trg_feats=nfeats
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights, by_name=True)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

losses = [image_loss_func, vxm.losses.MSE().loss, vxm.losses.Grad(
    'l2', loss_mult=2).loss, vxm.losses.MSE().loss]
weights = [args.image_loss_weight, args.mean_loss_weight,
           args.grad_loss_weight, args.deform_loss_weight]

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
                    callbacks=[save_callback],
                    steps_per_epoch=args.steps_per_epoch,
                    verbose=1
                    )
