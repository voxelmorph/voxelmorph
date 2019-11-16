"""
Example script to train a subject-to-atlas VoxelMorph model.

Note: For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are subfolders with normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this
script slightly to accomadate your own data.

To replicate cvpr2018 training:
    python train.py datadir --int-steps 0

To replicate miccai2018 training:
    python train.py datadir --int-steps 7 --half-size --resize 0.5 --use-probs --legacy-image-sigma 0.02
"""

import os
import random
import argparse
import glob
import numpy as np
import keras
import tensorflow as tf
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', help='model output directory (default: models)')
parser.add_argument('--load-model', help='optional model file to initialize with')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')

# network architecture parameters
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--resize', type=float, default=1, help='resize (default: 1)')
parser.add_argument('--half-size', action='store_true', help='???')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# model hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or nccc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='sigma', type=float, help='image noise parameter for miccai2018 (recommended: 0.02)')
args = parser.parse_args()

batch_size = args.batch_size
bidir = args.bidir
use_probs = args.use_probs
int_steps = args.int_steps
image_loss = args.image_loss
full_size = not args.half_size

# get base voxemorph directory so this script can be called from anywhere
basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# load atlas volume - the atlas we used is 160 x 192 x 224
atlas_file = args.atlas
if atlas_file is None:
    atlas_file = os.path.join(basedir, 'data/atlas_norm.npz')
atlas = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
vol_size = atlas.shape[1:-1]

# load and prepare training data
train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
random.shuffle(train_vol_names)  # shuffle volume list
assert len(train_vol_names) > 0, 'Could not find any training data'

# prepare model folder
model_dir = os.path.join(basedir, 'models') if not args.model_dir else args.model_dir
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

# TODO extend this as an argument
# unet architecture used in CVPR 2018 paper
nf_enc = [16, 32, 32, 32]
nf_dec = [32, 32, 32, 32, 32, 16, 16]

with tf.device(device):

    # prepare the model
    model = vxm.networks.vxmnet(vol_size, nf_enc, nf_dec, bidir=bidir, full_size=full_size,
                                vel_resize=args.resize, use_probs=use_probs, int_steps=int_steps)

    # load initial weights (if provided)
    if args.load_model:
        model.load_weights(args.load_model)

    # prepare image loss
    assert image_loss in ('mse', 'ncc'), 'Loss should be mse or ncc, but found "%s"' % image_loss
    if image_loss == 'ncc':
        image_loss = losses.NCC().loss
    elif image_loss == 'mse' and args.sigma:
        image_loss = losses.ReconMSE(args.sigma)

    if bidir:
        losses  = [image_loss, image_loss]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss]
        weights = [1]

    # prepare deformation loss TODO for KL
    deformation_loss = vxm.losses.KL(args.kl_lambda).loss if use_probs else vxm.losses.Grad('l2').loss
    losses  += [deformation_loss]
    weights += [args.weight]

# subject to atlas data generator
generator = vxm.generators.subj2atlas(train_vol_names, atlas, batch_size=batch_size, bidir=bidir)

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# fit model
with tf.device(device):

    # multi-gpu support
    # TODO fix this
    if nb_gpus > 1:
        save_callback = nrn_gen.ModelCheckpointParallel(save_filename)
        model = multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = keras.callbacks.ModelCheckpoint(save_filename)

    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))

    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback],
        verbose=1
    )
