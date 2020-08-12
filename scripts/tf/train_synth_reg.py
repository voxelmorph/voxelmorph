#!/usr/bin/env python

"""
Example script to train a VoxelMorph model on images synthesized from segmentations.
"""

import sys
import os
import random
import argparse
import glob
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--data-dir', default='data', help='base data directory')
parser.add_argument('--model-dir', default='model', help='model output directory (default: models)')
parser.add_argument('--log-dir', default='log', help='TensorBoard log directory (default: None)')
parser.add_argument('--sub-dir', default=None, help='sub-directory for logging and saving model weights (default: None)')

# generation parameters
parser.add_argument('--labels', default=None, help='labels whose overlap to optimize (default: all)')
parser.add_argument('--same-subj', action='store_true', help='generate image and label-map pairs from the same segmentation')
parser.add_argument('--vel-std', type=float, default=6, help='standard deviation of velocity field (default: 6)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--verbose', type=int, default=1, help='verbosity level (default: 1)')
parser.add_argument('--save-freq', type=int, default=10, help='number of epochs between model saves (default: 10)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 32 64 64 64)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 64 64 64 64 64 32)')
parser.add_argument('--int-steps', type=int, default=5, help='number of integration steps (default: 5)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--reg-param', type=float, default=1, help='weight of gradient loss (default: 1)')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
assert np.mod(args.batch_size, nb_devices) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (args.batch_size, nb_devices)

# prepare model directory
model_dir = args.model_dir
if args.sub_dir:
    model_dir = os.path.join(model_dir, args.sub_dir)
os.makedirs(model_dir, exist_ok=True)

# prepare loggging directory
log_dir = args.log_dir
if log_dir:
    if args.sub_dir:
        log_dir = os.path.join(log_dir, args.sub_dir)
    os.makedirs(log_dir, exist_ok=True)

# extract shape and training labels
all_labels, dataset = vxm.tf.synthseg.utils.get_list_labels(labels_folder=args.data_dir)
hot_labels = np.load(args.labels) if args.labels else all_labels
inshape = dataset[0].shape

# unet architecture
enc_nf = args.enc if args.enc else [32, 64, 64, 64]
dec_nf = args.dec if args.dec else [64, 64, 64, 64, 32]

with tf.device(device):

    # build the model
    model = vxm.networks.VxmDenseSynth(
        inshape=inshape,
        all_labels=all_labels,
        hot_labels=hot_labels,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
    )
    warp_shape = model.warp_shape
    bias_shape = model.bias_shape


# define a generator
def synth_generator(files, all_labels, warp_shape, bias_shape, batch_size=1, same_subj=False, vel_std=6):
    """
    Generator for sythesizing images and label maps from segmentations.
    Parameters:
        files: List of segmentations or paths.
        all_labels: List of unique labels included.
        warp_shape: Shape of warps to sample internally.
        bias_shape: Shape of bias field to sample internally.
        batch_size: Batch size. Default is 1.
        same_subj: Whether to generate source and target from the same segmentation. Default is False.
        vel_std: Maximum sampling SD for velocity fields. Default is 6.
    """
    gen = vxm.tf.synthseg.build_model_input_generator(files, all_labels, warp_shape, bias_shape,
        nonlin_std_dev=vel_std, batch_size=2*batch_size, same_subj=same_subj)
    split = lambda x: (x[:batch_size,...], x[batch_size:,...])
    zeros = None
    while True:
        inputs = [split(i) for i in next(gen)]
        in1, in2 = map(lambda i: [x[i] for x in inputs], range(2))
        if zeros is None:
            zeros = np.zeros(in1[0].shape)
        yield (*in1, *in2), (zeros, zeros)


# construct a generator
generator = synth_generator(
    dataset,
    all_labels,
    warp_shape,
    bias_shape,
    same_subj=args.same_subj,
    vel_std=args.vel_std,
    batch_size=args.batch_size
)

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

with tf.device(device):

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # configure custom dice
    def data_loss(_, x):
        shape = x.shape.as_list()
        depth = int(shape[-1] / 2)
        true = x[..., :depth]
        pred = x[..., depth:]
        return 1 + vxm.losses.Dice().loss(true, pred)

    losses  = [data_loss, vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights = [1, args.reg_param]

    # multi-gpu support and callbacks
    if nb_devices > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename, period=args.save_freq)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=args.save_freq)

    # tensorboard callback
    callbacks = [save_callback]
    if log_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))

    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=args.verbose,
    )
