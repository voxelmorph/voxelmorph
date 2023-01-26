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
def register_single(conf, subject, tvec, logger=None):
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
    normalized_vols = normalize(vols)
    low_matrix, sparse_matrix = rpca(np.squeeze(normalized_vols).transpose(1, 2, 0), rank=conf.rank) # (H, W, N)

    fixed = torch.from_numpy(low_matrix[None, ...]).float().permute(0, 3, 1, 2).to(device)
    predvols, warp = model(fixed, registration=True)

    if conf.final:
    # Apply the warp to the original image
        original_name = os.path.join(conf.orig_folder, f"{name}.nii.gz")
        fbMOLLI_vols = sitk.GetArrayFromImage(sitk.ReadImage(original_name))
        fbMOLLI_size = fbMOLLI_vols.shape[1:]
        fbMOLLI_vols = torch.from_numpy(fbMOLLI_vols[:, None, ...].astype(np.int16)).float().to(device)

        resized_warp = vxm.networks.interpolate_(warp, size=fbMOLLI_size, mode='bilinear')
        hydralog.debug(f"Type of resized_warp {resized_warp.get_device()} and type of fbMOLLI_vols {fbMOLLI_vols.get_device()}")
        # hydralog.debug(f"Type fbmolli {type(fbMOLLI_vols)} and warp {type(resized_warp.to(device))}")
        MOLLI_vols_pred = vxm.layers.SpatialTransformer(size=fbMOLLI_size)(fbMOLLI_vols.to('cpu'), resized_warp.to('cpu'))
        orig_vols = fbMOLLI_vols
        rigs_vols = MOLLI_vols_pred
        rigs_warp = resized_warp
    else:    
        vol_size = vols.shape[1:-1]
        vols = torch.from_numpy(vols).float().permute(0, 3, 1, 2).to(device)
        # print(f"Mona: vols shape {vols.shape}")
        predvols = vxm.layers.SpatialTransformer(size=vol_size)(vols.to('cpu'), warp.to('cpu'))
        orig_vols = vols
        rigs_vols = predvols
        rigs_warp = warp
    orig_T1err = vxm.groupwise.utils.update_atlas(orig_vols, 't1map', tvec=tvec)
    rigs_T1err = vxm.groupwise.utils.update_atlas(rigs_vols, 't1map', tvec=tvec)
    hydralog.info(f"The T1 error orig {np.mean(orig_T1err)} and rigs {np.mean(rigs_T1err)}")
    # Save the results of original image
    orig_vols = np.squeeze(orig_vols.detach().cpu().numpy())
    rigs_vols = np.squeeze(rigs_vols.detach().cpu().numpy())
    rigs_warp = rigs_warp.detach().cpu().numpy()
    
    name, org_mse, org_dis, rig_mse, rig_dis = saveEval(orig_vols, rigs_vols, rigs_warp, conf, name, fixed_affine, logger)
    saveT1err(orig_T1err, rigs_T1err, conf, name, logger)
    return name, org_mse, org_dis, np.mean(orig_T1err), rig_mse, rig_dis, np.mean(rigs_T1err)


def saveT1err(orig, rigs, conf, name, logger=None, size=(4, 2), title_font_size=8, title_pad = 10):
    fig, ax = plt.subplots(figsize=size)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(np.squeeze(orig), cmap='gray')
    plt.axis('off')
    ax1.set_title('T1err_orig', fontsize=title_font_size, pad=title_pad)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(np.squeeze(rigs), cmap='gray')
    plt.axis('off')
    ax2.set_title('T1err_rigs', fontsize=title_font_size, pad=title_pad)
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9,
                        bottom=0.1, wspace=0.001, hspace=0.2)
    plt.savefig(os.path.join(conf.result, f"{name}_T1err_compare.png"))
    plt.close()

    fig, ax = plt.subplots()
    sns.distplot(orig, label='Original', ax=ax)
    sns.distplot(rigs, label='Registered', ax=ax)
    ax.set_title("T1 Error Distribution")
    ax.set_xlabel("T1 Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.savefig(os.path.join(conf.result, f"{name}_T1err_distribution.png"))
    plt.close()

def saveEval(invols, outvols, warp, conf, name, fixed_affine, logger=None):
    orig = invols.transpose(1, 2, 0)
    moved = outvols.transpose(1, 2, 0)

    if conf.moved:
        vxm.py.utils.save_volfile(outvols.transpose(1, 2, 0), os.path.join(conf.moved, f"{name}.nii"), fixed_affine)
        np.save(os.path.join(conf.moved, f"{name}.npy"), outvols)

    if conf.warp:
        warp = warp.transpose(2, 3, 0, 1)
        vxm.py.utils.save_volfile(warp, os.path.join(conf.warp, f"{name}.nii"), fixed_affine)

    warp = warp.transpose(3, 2, 0, 1)

    org_mse, rig_mse = 0, 0
    hydralog.debug(f"Shape of orig {orig.shape} and moved {moved.shape}")
    for j in range(1, moved.shape[-1]):
        rig_mse += mse(moved[:, :, j-1], moved[:, :, j])
        org_mse += mse(orig[:, :, j-1], orig[:, :, j])

    eig_org, org_K, org_dis = pca(orig, topk=1)
    eig_rig, rig_K, rig_dis = pca(moved, topk=1)
    
    if conf.final:
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        sns.barplot(x=np.arange(len(eig_org)),
                    y=np.around(eig_org, 2), palette="rocket", ax=ax1)
        sns.barplot(x=np.arange(len(eig_rig)),
                    y=np.around(eig_rig, 2), palette="rocket", ax=ax2)
        ax1.set_title(f"Eigenvalues of original image {name}")
        ax2.set_title(f"Eigenvalues of registered image {name}")
        plt.savefig(os.path.join(conf.result, f"{name[:-4]}_pca_barplot.png"))
        original_gif_path = save_gif(orig, name, conf.result, "original")
        moved_gif_path = save_gif(moved, name, conf.result, "registered")
        morph_field_path = save_morphField(warp, name, conf.result)
        if logger:
            logger.log_gifs(original_gif_path, label="original Gif")
            logger.log_gifs(moved_gif_path, label="registered Gif")
            logger.log_gifs(morph_field_path, label="Quiver Gif")
        plt.close('all')
    
    hydralog.debug(f"File {name}, original MSE - {org_mse:.5f} PCA - {org_dis:.5f}, registered MSE - {rig_mse:5f} PCA - {rig_dis:.5f}")
    return name, org_mse, org_dis, rig_mse, rig_dis
    