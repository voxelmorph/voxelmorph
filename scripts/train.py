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
import time
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from NeptuneLogger import NeptuneLogger
from torchsummary import summary
import matplotlib.pyplot as plt
import voxelmorph_group as vxm  # nopep8

# parse the commandline


def train(conf, logger=None):

    bidir = conf.bidir

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(conf.img_list, prefix=conf.img_prefix,
                                              suffix=conf.img_suffix)
    print(f"Mona-1: the number of input files {len(train_files)}")
    assert len(train_files) > 0, 'Could not find any training data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not conf.multichannel

    if conf.atlas:
        # group-to-atlas generator
        print("Mona: use the group to atlas generator")
        # group wise batch size is always 1
        generator = vxm.generators.group_to_atlas(train_files, conf.atlas,
                                                  batch_size=1, bidir=conf.bidir,
                                                  add_feat_axis=add_feat_axis,
                                                  method=conf.atlas_methods)
    else:
        # scan-to-scan generator
        print("Mona: use the scan to scan generator")
        generator = vxm.generators.scan_to_scan(
            train_files, in_order=conf.in_order, batch_size=conf.batch_size, bidir=conf.bidir, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    shapes = next(generator)[0].shape
    inshape = shapes[1:-1]
    n_inputs = shapes[0]
    print(f"Mona-2: inshape {inshape}")

    # prepare model folder
    model_dir = conf.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    nb_gpus = conf.gpu
    device = 'cuda' if nb_gpus > 0 else 'cpu'
    if device == 'cuda':
        assert np.mod(conf.batch_size, nb_gpus) == 0, \
            'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (
                conf.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not conf.cudnn_nondet

    # unet architecture
    enc_nf = conf.enc if conf.enc else [16, 32, 32, 32]
    dec_nf = conf.dec if conf.dec else [32, 32, 32, 32, 32, 16, 16]

    if conf.load_model:
        # load initial model (if specified)
        if conf.transformation == 'Dense':
            model = vxm.networks.VxmDense.load(conf.model_path, device)
            print("Mona: load the bspline model")
        elif conf.transformation == 'bspline':
            model = vxm.networks.VxmDenseBspline.load(conf.model_path, device)
            print("Mona: load the bspline model")
    else:
        # otherwise configure new model
        if conf.transformation == 'Dense':
            model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=conf.int_steps,
                int_downsize=conf.int_downsize
            )
        elif conf.transformation == 'bspline':
            model = vxm.networks.GroupVxmDenseBspline(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=bidir,
                int_steps=conf.int_steps,
                int_downsize=conf.int_downsize,
                src_feats=20,
                trg_feats=40,
                cps=conf.bspline_config.cps,
                svf=conf.bspline_config.svf,
                svf_steps=conf.bspline_config.svf_steps,
                svf_scale=conf.bspline_config.svf_scale,
                resize_channels=conf.bspline_config.resize_channels
            )
            print("Mona: use the bspline model")
            # summary(model, input_size=(20, 224, 224), batch_size=1, device='cpu')

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    # prepare image loss
    if conf.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC(device=device).loss
    elif conf.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif conf.image_loss == 'nmi':
        # image_loss_func = vxm.losses.NMI().metric
        image_loss_func = vxm.losses.MILossGaussian(conf.mi_config)
    elif conf.image_loss == 'ngf':
        image_loss_func = vxm.losses.GradientCorrelation2d(device=device)
    elif conf.image_loss == 'jointcorrelation':
        image_loss_func = vxm.losses.JointCorrelation()
    else:
        raise ValueError(
            'Image loss should be "mse" or "ncc", but found "%s"' % conf.image_loss)
    
    mse = vxm.losses.MSE().loss
    ncc = vxm.losses.NCC(device=device).loss
    nmi = vxm.losses.MILossGaussian(conf.mi_config)
    jacobian = vxm.losses.Jacobian().loss
    ngf = vxm.losses.GradientCorrelation2d(device=device)

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [conf.image_loss_weight]

    # prepare deformation loss
    if conf.norm == 'l2' or conf.norm == 'l1':
        losses += [vxm.losses.Grad(conf.norm, loss_mult=conf.int_downsize).loss]
    elif conf.norm == 'd2':
        losses += [vxm.losses.BendingEnergy2d().grad]
    weights += [conf.weight]

    l1 = vxm.losses.Grad('l1', loss_mult=conf.int_downsize).loss
    l2 = vxm.losses.Grad('l2', loss_mult=conf.int_downsize).loss
    d2 = vxm.losses.BendingEnergy2d().grad

    global_step = 0
    # training loops
    for epoch in range(conf.initial_epoch, conf.epochs):

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(conf.steps_per_epoch):
            global_step += 1
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, atlas = next(generator)
            inputs = [torch.from_numpy(inputs).to(device).float().permute(3, 0, 1, 2)] # (C, n, H, W)
            atlas = torch.from_numpy(atlas).to(device).float().permute(3, 0, 1, 2) # (C, n, H, W)
            # run inputs through the model to produce a warped image and flow field
            y_pred, new_atlas, flow = model(*inputs) # y_pred: (n, C, H, W), new_atlas: (1, 1, H, W), flow: (n, 2, H, W)
            # calculate total loss
            loss_list = []
            metrics_list = []
            sim_loss = 0
            reg_loss = 0
            folding_ratio, mag_det_jac_det = jacobian(flow)
            if conf.image_loss == 'jointcorrelation':
                sim_loss += (losses[0](y_pred) * weights[0]).to(device)
                for slice in range(y_pred.shape[0]):
                    reg_loss += (losses[1](y_pred[slice, ...], flow) * weights[1]).to(device)
                reg_loss += torch.tensor(mag_det_jac_det * weights[1]).to(device)
            else:
                for slice in range(y_pred.shape[0]):
                    y = y_pred[slice, ...]
                    sim_loss += (losses[0](y[None, ...], new_atlas) * weights[0]).to(device)
                    reg_loss += (losses[1](y_pred[slice, ...], flow) * weights[1]).to(device)
            loss_list.append(sim_loss.item()/y_pred.shape[0])
            loss_list.append(reg_loss.item()/y_pred.shape[0])
            loss = (sim_loss + reg_loss)/y_pred.shape[0]

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            logger.log_metric(global_step, "Step/Folding Ratio", folding_ratio)
            logger.log_metric(global_step, "Step/Mag Det Jac Det", mag_det_jac_det)
            logger.log_metric(global_step, "Step/L1", l1(y_pred, flow))
            logger.log_metric(global_step, "Step/L2", l2(y_pred, flow))
            logger.log_metric(global_step, "Step/D2", d2(y_pred, flow))
            
        if epoch % 10 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, conf.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(
            ['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (
            np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

        logger.log_epoch_metric(epoch, np.mean(epoch_total_loss), np.mean(
            epoch_loss, axis=0))
        fig = logger.log_morph_field(global_step, y_pred, inputs[0], atlas, new_atlas, flow, "Validation Image")

        plt.savefig(os.path.join(conf.val, 'valimg_%04d.png' % conf.epochs))
        logger.log_img_frompath(fig, "Validation Image", os.path.join(conf.val, 'valimg_%04d.png' % conf.epochs))
        plt.close(fig)

        logger.log_metric(epoch, "Epoch/Folding Ratio", folding_ratio)
        logger.log_metric(epoch, "Epoch/Mag Det Jac Det", mag_det_jac_det)
        logger.log_metric(epoch, "Epoch/L1", l1(y_pred, flow))
        logger.log_metric(epoch, "Epoch/L2", l2(y_pred, flow))
        logger.log_metric(epoch, "Epoch/D2", d2(y_pred, flow))
    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % conf.epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/MOLLI_jointcorrelation_group.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    if conf.wandb:
        logger = NeptuneLogger(project_name=conf.wandb_project, cfg=conf)

    # run the training
    train(conf, logger)
