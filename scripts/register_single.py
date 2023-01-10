#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
from skimage import exposure
import seaborn as sns
import matplotlib.pyplot as plt
from sewar.full_ref import mse
from utils import *


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_group as vxm   # nopep8

# parse commandline conf

def register_single(conf, subject, logger=None):
    # device handling
    if conf.gpu and (conf.gpu != '-1'):
        device = 'cuda'
        # os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    name = (subject).split(".")[0]

    # load and set up model
    print(f'Loading bspline model - {conf.model_path} and {conf.transformation}')
    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense.load(conf.model_path, device)
    elif conf.transformation == 'bspline':
        
        model = vxm.networks.GroupVxmDenseBspline.load(conf.model_path, device)
    else:
        raise ValueError('transformation must be dense or bspline')

    model.to(device)
    model.eval()
    # load moving and fixed images
    add_feat_axis = not conf.multichannel
    vols, fixed_affine = vxm.py.utils.load_volfile(os.path.join(conf.moving, subject), add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    [_, slices, x, y, _] = vols.shape
    fixed = torch.from_numpy(vols[0, :, :, :, :]).float().permute(3, 0, 1, 2).to(device)

    
    predvols, warp = model(fixed, registration=True)

    invols = np.squeeze(fixed.detach().cpu().numpy())
    outvols = np.squeeze(predvols.detach().cpu().numpy())
    warp = warp.detach().cpu().numpy()

    orig = invols.transpose(1, 2, 0)
    moved = outvols.transpose(1, 2, 0)

    if conf.moved:
        vxm.py.utils.save_volfile(outvols.transpose(1, 2, 0), os.path.join(conf.moved, f"{name}.nii"), fixed_affine)

    if conf.warp:
        warp = warp.transpose(2, 3, 0, 1)
        vxm.py.utils.save_volfile(warp, os.path.join(conf.warp, f"{name}.nii"), fixed_affine)


    # output the metrics
    # save the gif
    
    warp = warp.transpose(3, 2, 0, 1)
    # warp = np.flip(warp, axis=2)
    # print(f"Shape of orig {orig.shape} and moved {moved.shape} and warp {warp.shape}")
    original_gif_path = save_gif(orig, name, conf.result, "original")
    moved_gif_path = save_gif(moved, name, conf.result, "registered")
    # quiver_path = save_quiver(warp, name, conf.result)
    morph_field_path = save_morphField(warp, name, conf.result)

    org_mse, rig_mse = 0, 0
    print(f"Shape of orig {orig.shape} and moved {moved.shape}")
    for j in range(1, moved.shape[-1]):
        rig_mse += mse(moved[:, :, j-1], moved[:, :, j])
        org_mse += mse(orig[:, :, j-1], orig[:, :, j])

    eig_org, org_K, org_dis = pca(
        orig, name, conf.result, "original", n_components=2)
    eig_rig, rig_K, rig_dis = pca(
        moved, name, conf.result, "registered", n_components=2)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    sns.barplot(x=np.arange(len(eig_org)),
                y=np.around(eig_org, 2), palette="rocket", ax=ax1)
    sns.barplot(x=np.arange(len(eig_rig)),
                y=np.around(eig_rig, 2), palette="rocket", ax=ax2)
    # ax1.bar_label(ax1.containers[0])
    # ax2.bar_label(ax2.containers[0])
    ax1.set_title(f"Eigenvalues of original image {name}")
    ax2.set_title(f"Eigenvalues of registered image {name}")
    plt.savefig(os.path.join(conf.result, f"{name[:-4]}_pca_barplot.png"))
    if logger:
        logger.log_gifs(original_gif_path, label="original Gif")
        logger.log_gifs(moved_gif_path, label="registered Gif")
        logger.log_gifs(morph_field_path, label="Quiver Gif")
        # logger.log_img(plt, "PCA change")

    
    print(f"File {name}, original MSE - {org_mse:.5f} PCA - {org_dis:.5f}, registered MSE - {rig_mse:5f} PCA - {rig_dis:.5f}")
    return name, org_mse, org_dis, rig_mse, rig_dis
    