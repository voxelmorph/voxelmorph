# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators


def test(model_name, gpu_id, 
         compute_type = 'GPU',  # GPU or CPU
         nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

    # Anatomical labels we want to evaluate
    labels = sio.loadmat('../data/labels.mat')['labels'][0]

    atlas = np.load('../data/atlas_norm.npz')
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
    atlas_seg = atlas['seg']
    vol_size = atlas_vol.shape[1:-1]

    # gpu handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)
        net.load_weights(model_name)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)

    # load subject test
    X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')

    with tf.device(gpu):
        pred = net.predict([X_vol, atlas_vol])

        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[1][0, :, :, :, :]
            warp_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            warp_seg = nn_trf_model.predict([X_seg, pred[1]])[0,...,0]

    vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
    dice_mean = np.mean(vals)
    dice_std = np.std(vals)
    print('Dice mean over structures: {:.2f} ({:.2f})'.format(dice_mean, dice_std))


if __name__ == "__main__":
    # test(sys.argv[1], sys.argv[2], sys.argv[3])
    test(sys.argv[1], sys.argv[2])
