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

    if conf.register == 'Group':
        # group-to-atlas generator
        hydralog.debug("Use the group to atlas generator")
        # group wise batch size is always 1
        generator = vxm.generators.group_to_atlas(train_files, True,
                                                  batch_size=1, bidir=conf.bidir,
                                                  add_feat_axis=add_feat_axis,
                                                  method=conf.atlas_methods)
    elif conf.register == 'Pair':
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

    if conf.transformation == 'Dense':
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=conf.int_steps,
            int_downsize=conf.int_downsize
        )
    elif conf.transformation == 'bspline':
        if conf.register == 'Group':
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
            hydralog.debug("Mona: use the Group bspline model")
    else:
        hydralog.error("Mona: the register type is not supported")
        raise NotImplementedError   
    model.to(device)

    # Model already exists in this round
    model_path = os.path.join(conf.model_dir_round, '%04d.pt' % conf.epochs)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        hydralog.info(f"Model {model_path} already exists !!!")
        return model
    
    # set optimizer and load the model in the previous round
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    if conf.load_model and conf.round > 1:
        saved_model = os.path.join(conf.model_dir, f"round{conf.round-1}", '%04d.pt' % conf.epochs)
        try:
            checkpoint = torch.load(saved_model, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch']
            loss = checkpoint['loss'].to(device)
            hydralog.info(f"Load the model from {saved_model}")
        except RuntimeError:
            raise RuntimeError(
                'Could not load model. Please check that the model was trained with the same configuration.')
    else:
        initial_epoch = 0
        loss = 0

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.train()

    # prepare image loss
    mse = vxm.losses.MSE().loss
    ncc = vxm.losses.NCC(device=device).loss
    nmi = vxm.losses.MILossGaussian(conf.mi_config)
    jacobian = vxm.losses.Jacobian().loss
    ngf = vxm.losses.GradientCorrelation2d(device=device)
    totalcorr = vxm.losses.JointCorrelation()
    # need two image loss functions if bidirectional
    if bidir:
        weights = [0.5, 0.5]
    else:
        weights = [conf.image_loss_weight]

    # prepare deformation loss
    weights += [conf.weight]
    weights += [conf.cycle_loss_weight]

    l1 = vxm.losses.Grad('l1', loss_mult=conf.int_downsize).loss
    l2 = vxm.losses.Grad('l2', loss_mult=conf.int_downsize).loss
    d2 = vxm.losses.BendingEnergy2d().grad

    global_step = 0
    # training loops
    if initial_epoch == conf.epochs:
        initial_epoch = 0
    for epoch in range(initial_epoch, conf.epochs):
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
            if conf.register == 'Group' and conf.rank == sequences:
                low_matrix = np.squeeze(inputs).transpose(1, 2, 0)
            else:
                low_matrix, sparse_matrix = vxm.py.utils.rpca(np.squeeze(
                    inputs).transpose(1, 2, 0), rank=conf.rank, lambda1=conf.lambda1)  # (H, W, N)
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

            folding_ratio, mag_det_jac_det = jacobian(flow)
            # compute all metric
            tc = totalcorr(y_pred) / y_pred.shape[0]
            new_atlas_repeat = new_atlas.repeat(y_pred.shape[0], 1, 1, 1)
            ncc_loss = ncc(y_pred, new_atlas)
            mse_loss = mse(y_pred, new_atlas)
            nmi_loss = nmi(y_pred, new_atlas_repeat)
            ngf_loss = ngf(y_pred, new_atlas_repeat)

            cyclic_loss = (torch.mean((torch.sum(flow, 0))**2))**0.5
            smooth_loss = vxm.losses.smooth_loss(flow, new_atlas)
            l2_loss = l2(y_pred, flow)
            l1_loss = l1(y_pred, flow)
            d2_loss = d2(y_pred, flow)
            if conf.image_loss == 'tc':
                sim_loss = tc.to(device)
            elif conf.image_loss == 'ncc':
                sim_loss = ncc_loss
            elif conf.image_loss == 'mse':
                sim_loss = mse_loss
            elif conf.image_loss == 'nmi':
                sim_loss = nmi_loss
            elif conf.image_loss == 'ngf':
                sim_loss = ngf_loss
            if conf.norm == 'l2':
                reg_loss = l2_loss
            elif conf.norm == 'l1':
                reg_loss = l1_loss
            elif conf.norm == 'd2':
                reg_loss = d2_loss
            elif conf.norm == 'smooth':
                reg_loss = smooth_loss

            loss_list.append(sim_loss.item())
            loss_list.append(reg_loss.item())
            loss = sim_loss * weights[0] + reg_loss * weights[1] + cyclic_loss * weights[2]

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            logger.log_metric(global_step, "Step/lr", optimizer.param_groups[0]['lr'])
            logger.log_metric(global_step, "Step/Folding Ratio", folding_ratio)
            logger.log_metric(global_step, "Step/Mag Det Jac Det", mag_det_jac_det)
            logger.log_metric(global_step, "Step/L1", l1_loss.item())
            logger.log_metric(global_step, "Step/L2", l2_loss.item())
            logger.log_metric(global_step, "Step/D2", d2_loss.item())
            logger.log_metric(global_step, "Step/Smooth", smooth_loss.item())
            logger.log_metric(global_step, "Step/Cyclic", cyclic_loss.item())

            logger.log_metric(global_step, "Step/MSE", mse_loss.item())
            logger.log_metric(global_step, "Step/NMI", nmi_loss.item())
            logger.log_metric(global_step, "Step/NCC", ncc_loss.item())
            logger.log_metric(global_step, "Step/NGF", ngf_loss.item())
            logger.log_metric(global_step, "Step/TC", tc.item())

        if epoch % 100 == 0:
            torch.save({
                'epoch': conf.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': model.config
            }, os.path.join(model_dir, '%04d.pt' % epoch))

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
    # model.save(os.path.join(model_dir, '%04d.pt' % conf.epochs))
    torch.save({
        'epoch': conf.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config
    }, os.path.join(model_dir, '%04d.pt' % conf.epochs))
    return model
