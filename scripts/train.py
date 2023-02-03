#!/usr/bin/env python

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *

import voxelmorph_group as vxm  # nopep8

hydralog = logging.getLogger(__name__)


def train(conf, logger=None):

    conf.model_path = os.path.join(
        conf.model_dir_round, '%04d.pt' % conf.epochs)
    if os.path.exists(conf.model_path):
        hydralog.info(f"Model {conf.model_path} already exists !!!")
        return

    bidir = conf.bidir

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(conf.img_list, prefix=conf.img_prefix,
                                              suffix=conf.img_suffix)
    hydralog.info(f"The number of input files {len(train_files)}")
    assert len(train_files) > 0, 'Could not find any training data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not conf.multichannel

    # load the TI for all subjects
    if conf.TI_csv:
        TI_dict = csv_to_dict(conf.TI_csv)

    if conf.atlas:
        # group-to-atlas generator
        hydralog.debug("Use the group to atlas generator")
        # group wise batch size is always 1
        generator = vxm.generators.group_to_atlas(train_files, conf.atlas,
                                                  batch_size=1, bidir=conf.bidir,
                                                  add_feat_axis=add_feat_axis,
                                                  method=conf.atlas_methods)
    else:
        # scan-to-scan generator
        hydralog.debug("Mona: use the scan to scan generator")
        generator = vxm.generators.scan_to_scan(
            train_files, in_order=conf.in_order, batch_size=conf.batch_size, bidir=conf.bidir, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    shapes = next(generator)[0].shape
    inshape = shapes[1:-1]
    sequences = shapes[0]
    hydralog.info(f"Inshape {shapes}")

    # prepare model folder
    model_dir = conf.model_dir_round
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
            hydralog.debug("Mona: load the Dense model")
        elif conf.transformation == 'bspline':
            model = vxm.networks.VxmDenseBspline.load(conf.model_path, device)
            hydralog.debug("Mona: load the Bspline model")
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
                src_feats=sequences,
                trg_feats=sequences*2,
                cps=conf.bspline_config.cps,
                svf=conf.bspline_config.svf,
                svf_steps=conf.bspline_config.svf_steps,
                svf_scale=conf.bspline_config.svf_scale,
                resize_channels=conf.bspline_config.resize_channels,
                method=conf.atlas_methods
            )
            hydralog.debug("Mona: use the bspline model")
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
    totalcorr = vxm.losses.JointCorrelation()
    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [conf.image_loss_weight]

    # prepare deformation loss
    if conf.norm == 'l2' or conf.norm == 'l1':
        losses += [vxm.losses.Grad(conf.norm,
                                   loss_mult=conf.int_downsize).loss]
    elif conf.norm == 'd2':
        losses += [vxm.losses.BendingEnergy2d().grad]
    weights += [conf.weight]

    l1 = vxm.losses.Grad('l1', loss_mult=conf.int_downsize).loss
    l2 = vxm.losses.Grad('l2', loss_mult=conf.int_downsize).loss
    d2 = vxm.losses.BendingEnergy2d().grad

    global_step = 0
    # training loops

    if logger._wandb.run.resumed:
        CHECKPOINT_PATH = os.path.join(
            model_dir, '%04d.pt' % conf.initial_epoch)
        model.load(CHECKPOINT_PATH, device)
        hydralog.info("Wandb resumed, load the model from the checkpoint")

    for epoch in range(conf.initial_epoch, conf.epochs):

        model.train()

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(conf.steps_per_epoch):
            global_step += 1
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, name = next(generator)
            hydralog.debug(
                f"The subject name is {name}, TI is {TI_dict[name]}")
            low_matrix, sparse_matrix = rpca(np.squeeze(
                inputs).transpose(1, 2, 0), rank=conf.rank)  # (H, W, N)
            inputs = [torch.from_numpy(low_matrix[None, ...]).to(
                device).float().permute(0, 3, 1, 2)]  # (C, n, H, W)

            atlas = vxm.groupwise.utils.update_atlas(inputs[0].permute(
                1, 0, 2, 3), conf.atlas_methods, tvec=TI_dict[name])
            # run inputs through the model to produce a warped image and flow field
            # y_pred: (n, C, H, W), new_atlas: (1, 1, H, W), flow: (n, 2, H, W)
            # hydralog.debug(f"Mona atlas {np.sum(atlas)}")
            y_pred, new_atlas, flow = model(*inputs, tvec=TI_dict[name])
            hydralog.debug(
                f"The atlas shape {atlas.shape} and new atlas shape {new_atlas.shape}, type {type(new_atlas)}")
            # calculate total loss
            loss_list = []
            sim_loss = 0
            reg_loss = 0
            mse_loss = 0
            nmi_loss = 0
            ncc_loss = 0
            ngf_loss = 0

            folding_ratio, mag_det_jac_det = jacobian(flow)
            # compute all metric
            tc = totalcorr(y_pred)
            for slice in range(y_pred.shape[0]):
                ncc_loss += ncc(y_pred[slice, ...][None, ...], new_atlas)
                mse_loss += mse(y_pred[slice, ...][None, ...], new_atlas)
                nmi_loss += nmi(y_pred[slice, ...][None, ...], new_atlas)
                ngf_loss += ngf(y_pred[slice, ...][None, ...], new_atlas)

            if conf.image_loss == 'jointcorrelation':
                sim_loss += (losses[0](y_pred) * weights[0]).to(device)
                for slice in range(y_pred.shape[0]):
                    reg_loss += (losses[1](y_pred[slice, ...],
                                 flow) * weights[1]).to(device)
                reg_loss += torch.tensor(mag_det_jac_det *
                                         weights[1]).to(device)
            else:
                for slice in range(y_pred.shape[0]):
                    y = y_pred[slice, ...]
                    sim_loss += (losses[0](y[None, ...],
                                 new_atlas) * weights[0]).to(device)
                    reg_loss += (losses[1](y_pred[slice, ...],
                                 flow) * weights[1]).to(device)
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
            logger.log_metric(
                global_step, "Step/Mag Det Jac Det", mag_det_jac_det)
            logger.log_metric(global_step, "Step/L1", l1(y_pred, flow))
            logger.log_metric(global_step, "Step/L2", l2(y_pred, flow))
            logger.log_metric(global_step, "Step/D2", d2(y_pred, flow))

            logger.log_metric(global_step, "Step/MSE",
                              mse_loss.item()/y_pred.shape[0])
            logger.log_metric(global_step, "Step/NMI",
                              nmi_loss.item()/y_pred.shape[0])
            logger.log_metric(global_step, "Step/NCC",
                              ncc_loss.item()/y_pred.shape[0])
            logger.log_metric(global_step, "Step/NGF",
                              ngf_loss.item()/y_pred.shape[0])
            logger.log_metric(global_step, "Step/TC",
                              tc.item()/y_pred.shape[0])

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
        fig = logger.log_morph_field(
            global_step, y_pred, inputs[0], atlas, new_atlas, flow, "Validation Image")

        plt.savefig(os.path.join(conf.val, 'valimg_%04d.png' % epoch))
        logger.log_img_frompath(fig, "Validation Image", os.path.join(
            conf.val, 'valimg_%04d.png' % epoch))
        plt.close(fig)

    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % conf.epochs))
