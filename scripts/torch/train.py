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
from wandbLogger import WandbLogger

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# parse the commandline


def train(conf, wandb_logger=None):

    bidir = conf.bidir

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(conf.img_list, prefix=conf.img_prefix,
                                              suffix=conf.img_suffix)
    print(f"Mona-1: the number of input files {len(train_files)}")
    assert len(train_files) > 0, 'Could not find any training data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not conf.multichannel

    if conf.atlas:
        # scan-to-atlas generator
        atlas = vxm.py.utils.load_volfile(conf.atlas, np_var='vol',
                                          add_batch_axis=True, add_feat_axis=add_feat_axis)
        generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                                 batch_size=conf.batch_size, bidir=conf.bidir,
                                                 add_feat_axis=add_feat_axis)
    else:
        # scan-to-scan generator
        print("Mona: use the scan to scan generator")
        generator = vxm.generators.scan_to_scan(
            train_files, in_order=conf.in_order, batch_size=conf.batch_size, bidir=conf.bidir, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]
    print(f"Mona-2: inshape {inshape}")

    # prepare model folder
    model_dir = conf.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    nb_gpus = conf.gpu
    device = 'cuda' if nb_gpus > 0 else 'cpu'
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
        model = vxm.networks.VxmDense.load(conf.model_path, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=conf.int_steps,
            int_downsize=conf.int_downsize
        )

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
        image_loss_func = vxm.losses.NCC().loss
    elif conf.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif conf.image_loss == 'mu3':
        image_loss_func = vxm.losses.MutualInformation_v3().loss
    elif conf.image_loss == 'mu4':
        image_loss_func = vxm.losses.MutualInformation_v4().loss
    elif conf.image_loss == 'mi':
        image_loss_func = vxm.losses.NMI().metric
    else:
        raise ValueError(
            'Image loss should be "mse" or "ncc", but found "%s"' % conf.image_loss)

    mutual_info = vxm.losses.NMI().metric
    jacobian = vxm.losses.Jacobian().loss

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

    # wandb_logger.watchModel(model)

    global_step = 0
    # training loops
    for epoch in range(conf.initial_epoch, conf.epochs):

        if epoch % 100 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(conf.steps_per_epoch):
            global_step += 1
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            inputs = [torch.from_numpy(d).to(
                device).float().permute(0, 3, 1, 2) for d in inputs]
            # print(f"inputs shape {inputs[0].shape} and {inputs[1].shape}")
            y_true = [torch.from_numpy(d).to(
                device).float().permute(0, 3, 1, 2) for d in y_true]
            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            MI = mutual_info(y_true[0], y_pred[0]).item()
            # print(f"Mona: {type(y_pred[1])}")
            folding_ratio, mag_det_jac_det = jacobian(y_pred[1])

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
            wandb_logger.log_step_metric(global_step, loss, loss_list[0], loss_list[1], MI, folding_ratio, mag_det_jac_det)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, conf.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(
            ['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (
            np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

        wandb_logger.log_epoch_metric(epoch, np.mean(epoch_total_loss), np.mean(
            epoch_loss, axis=0)[0], np.mean(epoch_loss, axis=0)[1])
    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % conf.epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/ncc_b1.yaml', help='config file')
    args = parser.parse_args()

    # load the config file
    cfg = OmegaConf.load(args.config)
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    print(f"Mona debug - conf: {conf} and type: {type(conf)}")

    if conf.wandb:
        wandb_logger = WandbLogger(project_name=conf.wandb_project, cfg=conf)

    # run the training
    train(conf, wandb_logger)
