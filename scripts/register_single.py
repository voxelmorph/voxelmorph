#!/usr/bin/env python
import logging
import os

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import torch
from sewar.full_ref import mse
from utils import *

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_group as vxm  # nopep8


hydralog = logging.getLogger(__name__)
def register_single(conf, subject, logger=None):
    if conf.gpu and (conf.gpu != '-1'):
        device = 'cuda'
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    name = (subject).split(".")[0]

    # load and set up model
    hydralog.debug(f'Loading bspline model - {conf.model_path} and {conf.transformation}')
    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense.load(conf.model_path, device)
    elif conf.transformation == 'bspline':
        
        model = vxm.networks.GroupVxmDenseBspline.load(conf.model_path, device)
    else:
        raise ValueError('transformation must be dense or bspline')

    model.to(device)
    model.eval()
    add_feat_axis = not conf.multichannel
    vols, fixed_affine = vxm.py.utils.load_volfile(os.path.join(conf.moving, subject), add_feat_axis=add_feat_axis, ret_affine=True)
    fixed = torch.from_numpy(vols).float().permute(3, 0, 1, 2).to(device)
    predvols, warp = model(fixed, registration=True)

    # Apply the warp to the original image
    original_name = os.path.join(conf.orig_folder, f"{name}.nii.gz")
    fbMOLLI_vols = sitk.GetArrayFromImage(sitk.ReadImage(original_name))
    fbMOLLI_size = fbMOLLI_vols.shape[1:]
    fbMOLLI_vols = torch.from_numpy(fbMOLLI_vols[:, None, ...]).float().to(device)

    resized_warp = vxm.networks.interpolate_(warp, size=fbMOLLI_size, mode='bilinear')
    hydralog.info(f"Type of resized_warp {type(resized_warp)} and type of fbMOLLI_vols {type(fbMOLLI_vols)}")
    fbMOLLI_vols_pred = vxm.layers.SpatialTransformer(size=fbMOLLI_size)(fbMOLLI_vols, resized_warp.to(device))

    # # Save the results of resize image
    # invols = np.squeeze(fixed.detach().cpu().numpy())
    # outvols = np.squeeze(predvols.detach().cpu().numpy())
    # warp = warp.detach().cpu().numpy()
    # name, org_mse, org_dis, rig_mse, rig_dis = saveEval(invols, outvols, warp, conf, name, fixed_affine, logger=None)

    # Save the results of original image
    fbMOLLI_invols = np.squeeze(fbMOLLI_vols.detach().cpu().numpy())
    fbMOLLI_outvols = np.squeeze(fbMOLLI_vols_pred.detach().cpu().numpy())
    fbMOLLI_warp = resized_warp.detach().cpu().numpy()
    name, org_mse, org_dis, rig_mse, rig_dis = saveEval(fbMOLLI_invols, fbMOLLI_outvols, fbMOLLI_warp, conf, name, fixed_affine, logger)
    return name, org_mse, org_dis, rig_mse, rig_dis
    

def saveEval(invols, outvols, warp, conf, name, fixed_affine, logger=None):
    orig = invols.transpose(1, 2, 0)
    moved = outvols.transpose(1, 2, 0)

    if conf.moved:
        vxm.py.utils.save_volfile(outvols.transpose(1, 2, 0), os.path.join(conf.moved, f"{name}.nii"), fixed_affine)

    if conf.warp:
        warp = warp.transpose(2, 3, 0, 1)
        vxm.py.utils.save_volfile(warp, os.path.join(conf.warp, f"{name}.nii"), fixed_affine)

    warp = warp.transpose(3, 2, 0, 1)
    original_gif_path = save_gif(orig, name, conf.result, "original")
    moved_gif_path = save_gif(moved, name, conf.result, "registered")
    morph_field_path = save_morphField(warp, name, conf.result)

    org_mse, rig_mse = 0, 0
    hydralog.debug(f"Shape of orig {orig.shape} and moved {moved.shape}")
    for j in range(1, moved.shape[-1]):
        rig_mse += mse(moved[:, :, j-1], moved[:, :, j])
        org_mse += mse(orig[:, :, j-1], orig[:, :, j])

    eig_org, org_K, org_dis = pca(orig, topk=1)
    eig_rig, rig_K, rig_dis = pca(moved, topk=1)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    sns.barplot(x=np.arange(len(eig_org)),
                y=np.around(eig_org, 2), palette="rocket", ax=ax1)
    sns.barplot(x=np.arange(len(eig_rig)),
                y=np.around(eig_rig, 2), palette="rocket", ax=ax2)
    ax1.set_title(f"Eigenvalues of original image {name}")
    ax2.set_title(f"Eigenvalues of registered image {name}")
    plt.savefig(os.path.join(conf.result, f"{name[:-4]}_pca_barplot.png"))
    if logger:
        logger.log_gifs(original_gif_path, label="original Gif")
        logger.log_gifs(moved_gif_path, label="registered Gif")
        logger.log_gifs(morph_field_path, label="Quiver Gif")
    plt.close('all')
    
    hydralog.info(f"File {name}, original MSE - {org_mse:.5f} PCA - {org_dis:.5f}, registered MSE - {rig_mse:5f} PCA - {rig_dis:.5f}")
    return name, org_mse, org_dis, rig_mse, rig_dis
    