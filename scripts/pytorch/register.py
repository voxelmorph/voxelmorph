"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    python register.py moving.nii.gz fixed.nii.gz moved.nii.gz --model model.pt --save-warp warp.nii.gz

The source and target input images are expected to be affinely registered.
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
parser.add_argument('moved', help='registered image output filename')
parser.add_argument('--model', required=True, help='run nonlinear registration - must specify torch model file')
parser.add_argument('--save-warp', help='output warp filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=True)
fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=True, ret_affine=True)

# load and set up model
model = vxm.networks.VxmDense.load(args.model)
model.eval()
model.to(device)

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

# predict
moved, warp = model.warp(input_moving, input_fixed)

# save moved image
moved = array = moved.detach().cpu().numpy().squeeze()
vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.save_warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.save_warp, fixed_affine)
