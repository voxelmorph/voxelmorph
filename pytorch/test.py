"""
*Preliminary* pytorch implementation.

VoxelMorph testing
"""


# python imports
import os
import glob
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
import scipy.io as sio

sys.path.append('../ext/medipy-lib')
from medipy.metrics import dice


def test(gpu, 
         atlas_file, 
         model, 
         init_model_file):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
   
    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
    atlas_seg = atlas['seg']
    vol_size = atlas_vol.shape[1:-1]

    # Test file and anatomical labels we want to evaluate
    test_file = open('../voxelmorph/data/val_examples.txt')
    test_strings = test_file.readlines()
    test_strings = [x.strip() for x in test_strings]
    good_labels = sio.loadmat('../voxelmorph/data/test_labels.mat')['labels'][0]

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)

    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf.to(device)

    for k in range(0, len(test_strings)):

        vol_name, seg_name = test_strings[k].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 4, 1, 2, 3)
        
        warp, flow = model(input_moving, input_fixed)

        # Warp segment using flow
        moving_seg = torch.from_numpy(X_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 4, 1, 2, 3)
        warp_seg   = trf(moving_seg, flow).detach().cpu().numpy()

        vals, labels = dice(warp_seg, atlas_seg, labels=good_labels, nargout=2)
        #dice_vals[:, k] = vals
        #print(np.mean(dice_vals[:, k]))
        print(np.mean(vals))

        #return
 

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        help="gpu id")

    parser.add_argument("--atlas_file",
                        type=str,
                        dest="atlas_file",
                        default='../data/atlas_norm.npz',
                        help="gpu id number")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--init_model_file", 
                        type=str,
                        dest="init_model_file", 
                        help="model weight file")


    test(**vars(parser.parse_args()))

