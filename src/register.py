"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately. 
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
"""

# py imports
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
import nibabel as nib

# project
import networks
sys.path.append('../ext/neuron')
import neuron.layers as nrn_layers

def register(gpu_id, moving, fixed, model_file, out_img, out_warp):
    """
    register moving and fixed. 
    """  
    assert model_file, "A model file is necessary"
    assert out_img or out_warp, "output image or warp file needs to be specified"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data
    mov_nii = nib.load(moving)
    mov = mov_nii.get_data()[np.newaxis, ..., np.newaxis]
    fix_nii = nib.load(fixed)
    fix = fix_nii.get_data()[np.newaxis, ..., np.newaxis]

    with tf.device(gpu):
        # load model
        custom_layers = {'SpatialTransformer':nrn_layers.SpatialTransformer,
                 'VecInt':nrn_layers.VecInt,
                 'Sample':networks.Sample,
                 'Rescale':networks.RescaleDouble,
                 'Resize':networks.ResizeDouble,
                 'Negate':networks.Negate}

        net = keras.models.load_model(model_file, custom_objects=custom_layers)

        # register
        [moved, warp] = net.predict([mov, fix])

    # output image
    if out_img is not None:
        img = nib.Nifti1Image(moved[0,...,0], mov_nii.affine)
        nib.save(img, out_img)

    # output warp
    if out_warp is not None:
        img = nib.Nifti1Image(warp[0,...], mov_nii.affine)
        nib.save(img, out_warp)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # positional arguments
    parser.add_argument("moving", type=str, default=None,
                        help="moving file name")
    parser.add_argument("fixed", type=str, default=None,
                        help="fixed file name")

    # optional arguments
    parser.add_argument("--model_file", type=str,
                        dest="model_file", default='../models/cvpr2018_vm1_cc.h5',
                        help="models h5 file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--out_img", type=str, default=None,
                        dest="out_img", help="output image file name")
    parser.add_argument("--out_warp", type=str, default=None,
                        dest="out_warp", help="output warp file name")

    args = parser.parse_args()
    register(**vars(args))