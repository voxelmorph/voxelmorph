"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
subject (moving) to an atlas (fixed). To register a subject to the atlas and save the warp field, run:

    python register.py moving.nii.gz fixed.nii.gz model.yaml model.h5 -o warped.nii.gz -w warp.nii.gz

Where model.yaml and model.h5 represent the model configuration file and weights, respectively. The source
and target input images are expected to affinely-registered, but if not, an initial linear registration
can be run by specifing an affine model with the --affine flag, which expects two arguments - the model
config file and weights file.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import voxelmorph as vxm
import tensorflow as tf
import keras


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('moving', help='moving image (source) filename')
parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('config', help='model configuration')
parser.add_argument('weights', help='keras model weights')
parser.add_argument('--affine', nargs=2, help='run initial affine registration - must specify config file and weights')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('-o', '--out-image', help='warped output image filename')
parser.add_argument('-w', '--warp', help='output warp filename')
args = parser.parse_args()

# device handling
if args.gpu:
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'

# load moving image
moving_image = nib.load(args.moving)
moving = moving_image.get_data()

# load fixed image
fixed_image = nib.load(args.fixed)
fixed = fixed_image.get_data()

if args.affine:
    # read the affine model config
    config_file, affine_weights = args.affine
    config = vxm.utils.NetConfig.read(config_file)

    # pad inputs to a standard size
    padshape = config['padding']
    moving_padded, _ = vxm.utils.pad(moving.squeeze(), padshape)
    fixed_padded, cropping = vxm.utils.pad(fixed.squeeze(), padshape)

    # scale image sizes by some factor
    resize = config['resize']
    moving_resized = vxm.utils.resize(moving_padded, resize)[np.newaxis, ..., np.newaxis]
    fixed_resized = vxm.utils.resize(fixed_padded, resize)[np.newaxis, ..., np.newaxis]

    with tf.device(device):
        # load the affine model and build a net that returns the affine transforms
        config['return_affines'] = True
        affine_net, transforms = config.build_model(affine_weights)
        affine_matrix_model = keras.Model(affine_net.input, transforms)

        # predict the transform(s) and merge
        affines = affine_matrix_model.predict([moving_resized, fixed_resized])
        affine = vxm.utils.merge_affines(affines, resize)

        # apply the transform and crop back to the target space
        moving = moving_padded[np.newaxis, ..., np.newaxis]
        affine_transformer = vxm.networks.affine_transformer(moving_padded.shape)
        aligned = affine_transformer.predict([moving, affine])[0, ..., 0]
        moving = aligned[cropping]

moving = moving[np.newaxis, ..., np.newaxis]
fixed = fixed[np.newaxis, ..., np.newaxis]

with tf.device(device):
    # load flow model and convert to warp output model
    vxm_net = vxm.utils.NetConfig.read(args.config).build_model(args.weights)
    flownet = vxm.networks.build_warpnet(vxm_net)

    # predict warp and warped image
    moved, warp = flownet.predict([moving, fixed])

# save warped image
if args.out_image:
    img = nib.Nifti1Image(moved.squeeze(), fixed_image.affine)
    nib.save(img, args.out_image)

# save warp
if args.warp:
    img = nib.Nifti1Image(warp.squeeze(), fixed_image.affine)
    nib.save(img, args.warp)
