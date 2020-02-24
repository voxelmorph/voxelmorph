"""
Example script to train a VoxelMorph model.

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
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.00001)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or nccc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

# load and prepare training data
train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
random.shuffle(train_vol_names)  # shuffle volume list
assert len(train_vol_names) > 0, 'Could not find any training data'

if args.atlas:
    # scan-to-atlas generator
    atlas = np.load(args.atlas)['vol'][np.newaxis, ..., np.newaxis]
    generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, batch_size=args.batch_size, bidir=bidir)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=args.batch_size, bidir=bidir)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = f'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (args.batch_size, nb_gpus)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        enc_nf=enc_nf,
        dec_nf=dec_nf,
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.train()
model.to(device)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses  = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses  = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses  += [vxm.losses.Grad('l2').loss]
weights += [args.weight]

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    for step in range(args.steps_per_epoch):

        # generate inputs (and predicted outputs) and convert them to tensors
        inputs, predicted = next(generator)
        inputs    = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        predicted = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in predicted]

        # run inputs through the model to produce a warped image and flow field
        outputs = model(*inputs)

        # calculate total loss
        loss = 0
        for n, loss_function in enumerate(losses):
            loss += loss_function(outputs[n], predicted[n]) * weights[n]

        print('epoch %d step %d/%d - loss: %f' % (epoch + 1, step + 1, args.steps_per_epoch, loss.item()), flush=True)

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
