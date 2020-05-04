"""
Instance-specific optimization
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('moving', help='moving image (source) filename')
parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('moved', help='registered image output filename')
parser.add_argument('--model', required=True, help='pretrained nonlinear vxm model')
parser.add_argument('--save-warp', help='output warp filename')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

# training parameters
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--epochs', type=int, default=1, help='number of training epochs (default: 1)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or nccc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01, help='weight of gradient loss (default: 0.01)')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

with tf.device(device):

    # load model and predict
    warp_predictor = vxm.networks.VxmDense.load(args.model).get_predictor_model()
    _, orig_warp = warp_predictor.predict([moving, fixed])

    # initialize instance network
    model = vxm.networks.InstanceTrainer(moving.shape[1:], orig_warp)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    losses  = [image_loss_func, vxm.losses.Grad('l2').loss]
    weights = [1, args.lambda_weight]

    # train
    zeros = np.zeros(orig_warp.shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)
    model.fit(
        [moving, fixed, orig_warp],
        [fixed, zeros],
        batch_size=None,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        verbose=1
    )

    # get warped image and deformation field
    moved, warp = model.predict([moving, fixed, zeros])

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

# save warp
if args.save_warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.save_warp, fixed_affine)
