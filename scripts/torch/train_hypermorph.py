#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019.

    or

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
import time
import numpy as np
import torch

import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--test-reg', nargs=3,
                    help='example registration pair and result (moving fixed moved) to test')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=6000,
                    help='number of training epochs (default: 6000)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--scale', type=float, default=1., help='rescale the input images/volume')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=0.2,
                    help='hyperparameter end-point over-sample rate (default 0.2)')

args = parser.parse_args()

bidir = args.bidir

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
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    base_generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis,
        resize_factor=args.scale)

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
a = next(generator)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
device, nb_gpus = vxm.torch.utils.setup_device(args.gpu)
# gpus = args.gpu.split(',')
# nb_gpus = len(gpus)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.HyperVxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.HyperVxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        # svf_resolution=2,
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    # image_loss_func = vxm.losses.MSE().loss
    scaling = 1.0 / (args.image_sigma ** 2)
    image_loss_func = lambda x1, x2: scaling * torch.mean(torch.flatten(torch.square(x1 - x2)), -1)

else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# prepare loss functions and compile model
def image_loss(y_true, y_pred, hyper_val):
    # hyp = (1 - torch.squeeze(model.references.hyper_val))
    hyp = (1 - hyper_val)
    return hyp * image_loss_func(y_true, y_pred)

# grad_loss = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
def grad_loss(y_pred, hyper_val):
    # hyp = torch.squeeze(model.references.hyper_val)
    hyp = hyper_val
    return hyp * vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss(None, y_pred)

# prepare deformation loss
# weights = [1]
# losses += [grad_loss]
# weights += [args.weight]
losses = [image_loss, grad_loss]

if len(inshape) == 3:
    permute_dims = (0, 4, 1, 2, 3)  # vol-concat-channel last -> vol-concat-channel first
else:
    assert len(inshape) == 2
    permute_dims = (0, 3, 1, 2)  # channel-last -> channel-first

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, 'model_%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors

        inputs, y_true = next(generator)
        hyper_val = torch.from_numpy(inputs[-1]).to(device).float()
        inputs = [torch.from_numpy(d).to(device).float().permute(*permute_dims) for d in inputs[:2]] + [hyper_val]
        y_true = [torch.from_numpy(d).to(device).float().permute(*permute_dims) for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs, registration=True)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            if loss_function == grad_loss:
                curr_loss = loss_function(y_pred[n], hyper_val)  # * weights[n]
            else:
                curr_loss = loss_function(y_true[n], y_pred[n], hyper_val)  # * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4f' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4f  (%s)' % (float(np.mean(epoch_total_loss)), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))


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


