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
from monai.losses.dice import DiceLoss
from pathlib import Path

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
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

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
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

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
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
    generator = vxm.py.generators.scan_to_atlas(train_files, atlas,
                                                batch_size=args.batch_size, bidir=args.bidir,
                                                add_feat_axis=add_feat_axis)
elif args.image_loss == "dice":
    # Load the mov,fixe, mov_label -> fixed, blank, fixed_label datagenerator
    seg_files = [str(Path(entry).parent / "labels/synthseg_flair.nii") for entry in train_files]
    fs_labels = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 26, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72, 77, 80, 85, 251, 252, 253, 254, 255]
    print(f"train_files: {train_files}")
    print(f"seg_files: {seg_files}")
    generator = vxm.py.generators.semisupervised(vol_names = train_files, seg_names = seg_files, labels = fs_labels, atlas_file=None, downsize=1)
    invols, outvols = next(generator)
else:
    # scan-to-scan generator
    print(f"Generator inputs:")
    print(f"train_files: {train_files}")
    print(f"batch_size: {args.batch_size}")
    print(f"bidir: {args.bidir}")
    print(f"add_feat_axis: {add_feat_axis}")
    generator = vxm.py.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)


# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

print(f"This is inshape: {inshape}")

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture | WIP, sticking to defaults for now
# enc_nf = args.enc if args.enc else [16, 32, 32, 32]
# dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
# combined_nf = [enc_nf, dec_nf]
combined_nf = [16, 32, 32, 32, 32]
print(f"combined_nf: {combined_nf}")

if args.load_model:
    # load initial model (if specified)
    model = torch.load(args.load_model, map_location=device)
else:
    print(f"Configuring new model")
    print(f"spatial_shape: {inshape}")
    print(f"ndim: {len(inshape)}")
    print(f"nb_features: {combined_nf}")
    print(f"integration_steps: {args.int_steps}")
    print(f"bidrectional_cost: {bidir}")
    print(f"device: {device}")
    print(f"image loss: {args.image_loss}")
    # otherwise configure new model
    model = vxm.nn.models.VxmPairwise(
        spatial_shape=inshape,
        order='cna',
        ndim=len(inshape),
        source_channels=1,  # Assuming single channel
        target_channels=1,  # Assuming single channel
        nb_features=combined_nf,
        integration_steps=args.int_steps,
        bidirectional_cost=bidir,
        device=device
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
    image_loss_func = vxm.nn.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.nn.losses.MSE().loss
elif args.image_loss == "dice":
    image_loss_func = DiceLoss(
        sigmoid=False,
        squared_pred=True,
    )
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]


# prepare deformation loss
losses += [vxm.nn.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

if args.image_loss == "dice":
    weights = [0, args.weight, 1]

# Create a spatial transformer for segmentations (nearest-neighbor interpolation)
seg_transformer = vxm.nn.modules.SpatialTransformer(
    size=model.spatial_shape,
    # interpolation_mode='nearest'  # Critical for segmentations to preserve label values
).to(model.device)

def apply_displacement_field_to_seg(vxm_model, displacement, seg_volume):
    if vxm_model.integration_steps > 0:
        # Provide negative velocity only when bidirectional cost is desired
        neg_velocity = -displacement if vxm_model.bidirectional_cost else None
        displacement, _ = vxm_model._integrate_velocity_fields(displacement, neg_velocity)

    # Warp the segmentation using the displacement field from registration
    warped_seg = seg_transformer(seg_volume, displacement)
    return warped_seg

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        # model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        # Save the torch module since VxmPairwise object has no attribute save
        torch.save(model, os.path.join(model_dir, '%04d.pt' % epoch))


    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        if args.image_loss == "dice":
            invols, outvols = next(generator)
            inputs = [invols[0], invols[1]]
        else:
            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

        # run inputs through the model to produce a warped image and displacement field
        displacement, y_pred = model(*inputs, return_warped=True)

        # calculate total loss
        loss = 0
        loss_list = []
        # Append the dice loss and gradient loss:
        # Warp the moving labels
        warped_seg = apply_displacement_field_to_seg(model, displacement, torch.from_numpy(invols[2]).to(device).float().permute(0, 4, 1, 2, 3))
        outvols[2] = torch.from_numpy(outvols[2]).to(device).float().permute(0, 4, 1, 2, 3)
        dice_loss = image_loss_func(outvols[2], warped_seg)
        grad_loss = vxm.nn.losses.Grad('l2', loss_mult=args.int_downsize).loss(displacement) * args.weight
        loss_list = [dice_loss.item(), grad_loss.item()]
        loss = dice_loss + grad_loss

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
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

# final model save
# model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
# Save the torch module since VxmPairwise object has no attribute save
torch.save(model, os.path.join(model_dir, '%04d.pt' % args.epochs))