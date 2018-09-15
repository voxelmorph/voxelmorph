"""
train atlas-based alignment with MICCAI2018 version of VoxelMorph, 
specifically adding uncertainty estimation and diffeomorphic transforms.
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
import datagenerators
import networks
import losses


## some data prep
# Volume size used in our experiments. Please change to suit your data.
vol_size = (160, 192, 224)  

# prepare the data
# for the CVPR paper, we have data arranged in train/validate/test folders
# inside each folder is a /vols/ and a /asegs/ folder with the volumes
# and segmentations
base_data_dir = '/your/path/here'
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
random.shuffle(train_vol_names)  # shuffle volume list

# load atlas from provided files. This atlas is 160x192x224.
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol'][np.newaxis,...,np.newaxis]


def train(model_dir, gpu_id, lr, n_iterations, alpha, image_sigma, model_save_iter, batch_size=1):
    """
    model training function
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param alpha: the alpha, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param model_save_iter: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    """
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print(model_dir)

    # gpu handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # Diffeomorphic network architecture used in MICCAI 2018 paper
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3] 

    # prepare the model
    # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, velocity_stats]
    # in the experiments, we use image_2 as atlas
    with tf.device(gpu):
        # miccai 2018 used xy indexing. 
        model = networks.miccai2018_net(vol_size,nf_enc,nf_dec, use_miccai_int=True, indexing='xy')

        # compile
        model_losses = [losses.kl_l2loss(image_sigma), losses.kl_loss(alpha)]
        model.compile(optimizer=Adam(lr=lr), loss=model_losses)

        # save first iteration
        model.save(os.path.join(model_dir,  str(0) + '.h5'))
    
    train_example_gen = datagenerators.example_gen(train_vol_names)
    zeros = np.zeros((1, *vol_size, 3))

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    for step in range(1, n_iterations):
        
        # get_data
        X = next(train_example_gen)[0]        

        # train
        with tf.device(gpu):
            train_loss = model.train_on_batch([X,atlas_vol], [atlas_vol, zeros])

        if not isinstance(train_loss,list):
            train_loss = [train_loss]

        # print
        print_loss(step, 0, train_loss)

        # save model
        with tf.device(gpu):
            if (step % model_save_iter == 0) or step < 10:
                model.save(os.path.join(model_dir,  str(step) + '.h5'))


def print_loss(step, training, train_loss):
    """
    Prints training progress to std. out
    :param step: iteration number
    :param training: a 0/1 indicating training/testing
    :param train_loss: model loss at current iteration
    """
    s = str(step) + "," + str(training)

    if isinstance(train_loss, list) or isinstance(train_loss, np.ndarray):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--alpha", type=float,
                        dest="alpha", default=70000/128,
                        help="alpha regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.05,
                        help="image noise parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args))
