"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
subject (moving) to an atlas (fixed). To register a subject to the atlas and save the warp field, run:

    python register.py <moving> <fixed> <config> <weights> -o <warped-image> -w <warp>

"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('moving', help='moving image (source) filename')
parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('config', help='model configuration')
parser.add_argument('weights', help='keras model weights')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('-o', '--out-image', help='warped output image filename')
parser.add_argument('-w', '--warp', help='output warp filename')
args = parser.parse_args()

# device handling
if args.gpu:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving image
moving_image = nib.load(args.moving)
moving = moving_image.get_data()[np.newaxis, ..., np.newaxis]
affine = moving_image.affine

# load fixed image
fixed_image = nib.load(args.fixed)
fixed = fixed_image.get_data()[np.newaxis, ..., np.newaxis]

# set up model and load weights
model = vxm.utils.NetConfig.read(args.config).build_model(args.weights)
model.to(device)

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

# predict
moved, warp = model.warp(input_moving, input_fixed)

# save warped image
if args.out_image:
    array = moved.detach().cpu().numpy().squeeze()
    img = nib.Nifti1Image(array, moving_image.affine)
    nib.save(img, args.out_image)

# save warp
if args.warp:
    array = warp.detach().cpu().numpy().squeeze()
    img = nib.Nifti1Image(array, moving_image.affine)
    nib.save(img, args.warp)
