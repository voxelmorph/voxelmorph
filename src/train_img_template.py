"""
train (unconditional) template creation
"""

# python imports
import os
import glob
import sys
import itertools
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K

# project imports
import datagenerators
import networks
import losses

sys.path.append('../ext/pynd-lib')
from pynd import ndutils as nd

def train(data_dir,
          atlas_file,
          model_dir,
          model,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          image_sigma,
          mean_lambda,
          steps_per_epoch,
          batch_size,
          load_model_file,
          bidir,
          atlas_wt,
          bias_mult,
          smooth_pen_layer,
          data_loss,
          reg_param,
          ncc_win,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param nb_epochs: number of training iterations
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param bidir: logical whether to use bidirectional cost function
    """
    
     
    # prepare data files
    # we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    train_vol_names = glob.glob(data_dir)
    train_vol_names = [f for f in train_vol_names if 'ADNI' not in f]
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size)

    # prepare the initial weights for the atlas "layer"
    if atlas_file is None or atlas_file == "":
        nb_atl_creation = 100
        print('creating "atlas" by averaging %d subjects' % nb_atl_creation)
        x_avg = 0
        for _ in range(nb_atl_creation):
            x_avg += next(train_example_gen)[0][0,...,0]
        x_avg /= nb_atl_creation

        x_avg = x_avg[np.newaxis,...,np.newaxis]
        atlas_vol = x_avg
    else:
        atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1]

    # Diffeomorphic network architecture used in MICCAI 2018 paper
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3] 
    if model == 'm1':
        pass
    elif model == 'm1double':
        nf_enc = [f*2 for f in nf_enc]
        nf_dec = [f*2 for f in nf_dec]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC(win=[ncc_win]*3).loss      
    else:
        data_loss = lambda y_t, y_p: K.mean(K.square(y_t-y_p))

    # gpu handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # the MICCAI201 model takes in [image_1, image_2] and outputs [warped_image_1, velocity_stats]
        # in these experiments, we use image_2 as atlas
        model = networks.img_atlas_diff_model(vol_size, nf_enc, nf_dec, 
                                            atl_mult=1, bidir=bidir,
                                            smooth_pen_layer=smooth_pen_layer)



        # compile
        mean_layer_loss = lambda _, y_pred: mean_lambda * K.mean(K.square(y_pred))

        flow_vol_shape = model.outputs[-2].shape[1:-1]
        loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)
        if bidir:
            model_losses = [data_loss,
                            lambda _,y_p: data_loss(model.get_layer('atlas').output, y_p),
                            mean_layer_loss,
                            losses.Grad('l2').loss]
            loss_weights = [atlas_wt, 1-atlas_wt, 1, reg_param]
        else:
            model_losses = [loss_class.recon_loss, loss_class.kl_loss, mean_layer_loss]
            loss_weights = [1, 1, 1]
        model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    
        # set initial weights in model
        model.get_layer('atlas').set_weights([atlas_vol[0,...]])

        # load initial weights. # note this overloads the img_param weights
        if load_model_file is not None and len(load_model_file) > 0:
            model.load_weights(load_model_file, by_name=True)



    # save first iteration
    model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # atlas_generator specific to this model. Once we're convinced of this, move to datagenerators
    def atl_gen(gen):  
        zero_flow = np.zeros([batch_size, *vol_size, len(vol_size)])
        zero_flow_half = np.zeros([batch_size] + [f//2 for f in vol_size] + [len(vol_size)])
        while 1:
            x2 = next(train_example_gen)[0]
            # TODO: note this is the opposite of train_miccai and it might be confusing.
            yield ([atlas_vol, x2], [x2, atlas_vol, zero_flow, zero_flow])

    atlas_gen = atl_gen(train_example_gen)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
    save_callback = ModelCheckpoint(save_file_name)

    # fit generator
    with tf.device(gpu):
        model.fit_generator(atlas_gen, 
                            initial_epoch=initial_epoch,
                            epochs=nb_epochs,
                            callbacks=[save_callback],
                            steps_per_epoch=steps_per_epoch,
                            verbose=1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="data folder")

    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default=None,
                        help="atlas file")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--model", type=str,
                        dest="model", default='m1',
                        help="models name")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=10,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.02,
                        help="image noise parameter")
    parser.add_argument("--mean_lambda", type=float,
                        dest="mean_lambda", default=1,
                        help="mean_lambda regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='../models/miccai2018_10_02_init1.h5',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--bidir", type=int,
                        dest="bidir", default=1,
                        help="whether to use bidirectional cost function")
    parser.add_argument("--atlas_wt", type=float,
                        dest="atlas_wt", default=1.0,
                        help="atlas_wt")
    parser.add_argument("--bias_mult", type=float,
                        dest="bias_mult", default=1,
                        help="bias_mult")
    parser.add_argument("--smooth_pen_layer", type=str,
                        dest="smooth_pen_layer", default='diffflow',
                        help="smooth_pen_layer")
    parser.add_argument("--data_loss", type=str,
                        dest="data_loss", default='mse',
                        help="data_loss: mse of ncc")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")
    parser.add_argument("--ncc_win", type=float,
                        dest="ncc_win", default=9, 
                        help="ncc window")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="first epoch")

    args = parser.parse_args()
    train(**vars(args))


"""
# running example:
TRAIN_VOLS=/path/to/your/training/vols/
MODELS_DIR=/path/to/models/
LOSSES_DIR=/path/to/losses/

# with atlas file
gpu=0; data_loss=ncc; reg_param=1.0; atlas_wt=1; name=miccai2019_atlas_expt_${data_loss}_reg${reg_param}_atlwt${atlas_wt}_releasetest; python train_img_template.py "$TRAIN_VOLS/*.npz" --model_dir ${MODELS_DIR}/${name} --gpu ${gpu}  --load_model_file ${MODELS_DIR}/miccai2018_s2s_lambda10_sigma0.02_bidir1_double/5000.h5 --model m1double --lambda ${reg_param} --data_loss ${data_loss} --atlas_wt ${atlas_wt} --atlas_file "/home/gid-dalcaav/projects/voxelmorph/code/voxelmorph-sandbox/voxelmorph-pd/data/x_med_similar_stochastic.npz" > ${LOSSES_DIR}/${name}.out 2> ${LOSSES_DIR}/${name}.err &
"""