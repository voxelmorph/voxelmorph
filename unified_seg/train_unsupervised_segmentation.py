"""
Trains a segmentation network in an unsupervised fashion, 
using a probabilistic atlas and a bunch of unlabeled scans

Unsupervised deep learning for Bayesian brain MRI segmentation
A.V. Dalca, E. Yu, P. Golland, B. Fischl, M.R. Sabuncu, J.E. Iglesias
Under Review. arXiv https://arxiv.org/abs/1904.11319
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
import keras.backend as K

# project imports
sys.path.append('../src/')
import datagenerators
import networks
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen


def train_unsupervised_segmentation(data_dir,
          atlas_file,
          mapping_file,
          model,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          init_stats,
          reg_param,
          steps_per_epoch,
          batch_size,
          stat_post_warp,
          warp_method,
          load_model_file,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files (coregistered, intensity normalized)
    :param atlas_file: file with probabilistic atlas (coregistered to images)
    :param mapping_file: file with mapping from labels to tissue types
    :param model: registration (voxelmorph) model: vm1, vm2, or vm2double
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param nb_epochs: number of epochs
    :param init_stats: file with guesses for means and log-variances (vectors init_mu, init_sigma)
    :param reg_param: smoothness/reconstruction tradeoff parameter (lambda in the paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: default of 1. can be larger, depends on GPU memory and volume size
    :param stat_post_warp: set to 1  to use warped atlas to estimate Gaussian parameters
    :param warp_method: set to 'WARP' if you want to warp the atlas
    :param load_model_file: optional h5 model file to initialize with
    :param initial_epoch: initial epoch
    """

    # load reference soft edge and corresponding mask from provided files
    # (we used size 160x192x224).
    # Also: group labels in tissue types, if necessary
    if mapping_file is None:
        atlas_vol = np.load(atlas_file)['vol_data'][np.newaxis, ...]
        nb_labels = atlas_vol.shape[-1]

    else:
        atlas_full = np.load(atlas_file)['vol_data'][np.newaxis, ...]
        
        mapping = np.load(mapping_file)['mapping'].astype('int').flatten()
        assert len(mapping) == atlas_full.shape[-1], \
            'mapping shape %d is inconsistent with atlas shape %d' % (len(mapping), atlas_full.shape[-1])

        nb_labels = 1 + np.max(mapping)
        atlas_vol = np.zeros([1, *atlas_full.shape[1:-1], nb_labels.astype('int')])
        for j in range(np.max(mapping.shape)):
            atlas_vol[0, ..., mapping[j]] = atlas_vol[0, ..., mapping[j]] + atlas_full[0, ..., j]

    vol_size = atlas_vol.shape[1:-1]


    # load guesses for means and variances
    init_mu = np.load(init_stats)['init_mu']
    init_sigma = np.load(init_stats)['init_std']

    # prepare data files
    train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    random.shuffle(train_vol_names)
    assert len(train_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else: # 'vm2double': 
        nf_enc = [f*2 for f in nf_enc]
        nf_dec = [f*2 for f in [32, 32, 32, 32, 32, 16, 16]]

    # prepare model and log folders
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    log_dir=os.path.join(model_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # GPU handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))


    # prepare the model
    with tf.device(gpu):
        # prepare the model
        model = networks.cvpr2018_net_probatlas(vol_size, nf_enc, nf_dec, nb_labels,
                                                diffeomorphic=True,
                                                full_size=False,
                                                stat_post_warp=stat_post_warp,
                                                warp_method=warp_method,
                                                init_mu=init_mu,
                                                init_sigma=init_sigma)

        # load initial weights
        if load_model_file is not None:
            print('loading', load_model_file)
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size)
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    cvpr2018_gen = datagenerators.cvpr2018_gen(train_example_gen, atlas_vol_bs, batch_size=batch_size)
    
    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single-gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model
            
        # tensorBoard callback
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                                  write_graph=True, write_images=False)

        # compile loss and parameters
        def data_loss(_, yp):
            m = tf.cast(model.inputs[0] > 0, tf.float32)
            return -K.sum(yp * m) / K.sum(m)

        if warp_method != 'WARP':
            reg_param = 0

        # compile
        mg_model.compile(optimizer=Adam(lr=lr),
                        loss=[data_loss, losses.Grad('l2').loss],
                        loss_weights=[1.0, reg_param])

        # fit
        mg_model.fit_generator(cvpr2018_gen,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback ,tensorboard],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="data folder with npz files (coregistered, intensity normalized)"),
    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default='../data/prob_atlas_41_class.npz',
                        help="file with probabilistic atlas (coregistered to images)")
    parser.add_argument("--mapping_file", type=str,
                        dest="mapping_file", default='../data/atlas_class_mapping.npz',
                        help="file with mapping between labels and tissue types")
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2', 'vm2double'], default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/scratch/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--init_stats",
                        dest="init_stats", default='../data/meanstats_T1_WARP.npz',
                        help="npz with guesses for initial stats (vectors init_mu and init_sigma)")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=10.0,
                        help="regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--stat_post_warp", type=int,
                        dest="stat_post_warp", default=True,
                        help="compute Gaussian stats pre- (0) or post-warp (1)")
    parser.add_argument("--warp_method", type=str,
                        dest="warp_method", default='WARP',
                        help="warp_method: WARP or NONE")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default=None,
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()


    train_unsupervised_segmentation(**vars(args))
