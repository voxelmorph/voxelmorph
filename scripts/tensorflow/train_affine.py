"""
Example script to train an affine VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

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
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')
parser.add_argument('--prob-same', type=float, default=0.3, help='likelihood that source/target training images will be the same (default: 0.3)')

# network architecture parameters
parser.add_argument('--rigid', action='store_true', help='force rigid registration')
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--blurs', type=float, nargs='+', default=[1], help='levels of gaussian blur kernel for each scale (default: 1)')
parser.add_argument('--padding', type=int, nargs='+', default=[256, 256, 256], help='padded image target shape (default: 256 256 256')
parser.add_argument('--resize', type=float, default=0.25, help='after-padding image resize factor (default: 0.25)')
args = parser.parse_args()

batch_size = args.batch_size

# load and prepare training data
train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
random.shuffle(train_vol_names)  # shuffle volume list
assert len(train_vol_names) > 0, 'Could not find any training data'

generator_args = dict(no_warp=True, batch_size=batch_size, pad_shape=args.padding, resize_factor=args.resize, bidir=args.bidir)

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=True, resize_factor=args.resize, pad_shape=args.padding)
    generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, **generator_args)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(train_vol_names, prob_same=args.prob_same, **generator_args)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow gpu handling
device = '/gpu:' + args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.keras.backend.set_session(tf.Session(config=config))

# ensure valid batch size given gpu count
nb_gpus = len(args.gpu.split(','))
assert np.mod(batch_size, nb_gpus) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (batch_size, nb_gpus)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# transform type
transform_type = 'rigid' if args.rigid else 'affine'

with tf.device(device):

    # build the model
    model = vxm.networks.VxmAffine(
        inshape,
        enc_nf=enc_nf,
        transform_type=transform_type,
        bidir=args.bidir,
        blurs=args.blurs
    )

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # multi-gpu support
    if nb_gpus > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename)

    # configure loss
    image_loss_func = vxm.losses.NCC().loss

    # need two image loss functions if bidirectional
    if args.bidir:
        losses  = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss_func]
        weights = [1]

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))

    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback],
        verbose=1
    )

    # save final model weights
    model.save(save_filename.format(epoch=args.epochs))
