"""
train conditional template creation. Note this code is still experimental based on the experiments run in the preprint.

Learning Conditional Deformable Templates with Convolutional Networks
Adrian V. Dalca, Marianne Rakic, John Guttag, Mert R. Sabuncu
https://arxiv.org/abs/1908.02738 arXiv 2019
"""

# python imports
import os
import glob
import sys
import itertools
import random
from argparse import ArgumentParser
import csv

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.models
import keras.backend as K
from tqdm import tqdm

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

    # prepare data generation specific to conditional templates
    train_vols_dir = os.path.join(data_dir, 'vols')
    train_vol_basenames = [f for f in os.listdir(train_vols_dir) if ('ADNI' in f or 'ABIDE' in f)]
    train_vol_names = [os.path.join(train_vols_dir, f) for f in train_vol_basenames]

    # csv pruning
    # our csv is a file of the form:
    #   file,age,sex
    #   ADNI_ADNI-1.5T-FS-5.3-Long_63196.long.094_S_1314_base_mri_talairach_norm.npz,81.0,2
    csv_file_path = os.path.join(data_dir, 'combined_pheno.csv')
    train_atr_dct = load_pheno_csv(csv_file_path)
    train_vol_basenames = [f for f in train_vol_basenames if f in list(train_atr_dct.keys())]
    train_vol_names = [os.path.join(train_vols_dir, f) for f in train_vol_basenames]
    
    # replace keys with full path
    for key in list(train_atr_dct.keys()):
        if key in train_vol_basenames:
            train_atr_dct[os.path.join(data_dir, 'vols', key)] = train_atr_dct[key]
        train_atr_dct.pop(key, None)

    

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


    genobj = Generator(train_vol_names, atlas_vol, y_k=train_atr_dct)

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

        # parameters for atlas construction.
        nb_conv_features = 4
        cond_im_input_shape = [160,192,224,4]  # note the 8 features
        cond_nb_levels = 0
        cond_conv_size = [3,3,3]
        extra_conv_layers = 3
        pheno_input_shape = [2]

        bidir = True
        smooth_pen_layer = 'diffflow'

        model, mn = networks.cond_img_atlas_diff_model(vol_size, nf_enc, nf_dec,
                                                atl_mult=1, bidir=bidir, smooth_pen_layer=smooth_pen_layer,
                                                nb_conv_features=nb_conv_features,
                                                cond_im_input_shape=cond_im_input_shape,
                                                cond_nb_levels=cond_nb_levels,
                                                cond_conv_size=cond_conv_size,
                                                pheno_input_shape=pheno_input_shape,
                                                    extra_conv_layers=extra_conv_layers,
                                                ret_vm = True,
                                                )
        outputs = [model.outputs[f] for f in [0, 2, 3, 3]] # latest model used in paper
        model = keras.models.Model(model.inputs, outputs)


        # compile
        mean_layer_loss = lambda _, y_pred: mean_lambda * K.mean(K.square(y_pred))

        model_losses = [data_loss,  # could be mse or ncc or a combination, etc
                        mean_layer_loss,
                        lambda _, yp: losses.Grad('l2').loss(_, yp),
                        lambda _, yp: K.mean(K.square(yp))]


        # parameters used in paper
        msmag_param = 0.01
        reg_param = 1 
        centroid_reg = 1   
        loss_weights = [atlas_wt, centroid_reg, reg_param, msmag_param]
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    
        # load initial weights. # note this overloads the img_param weights
        if load_model_file is not None and len(load_model_file) > 0:
            model.load_weights(load_model_file, by_name=True)

    # save first iteration
    model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
    save_callback = ModelCheckpoint(save_file_name)



    


    # fit generator
    with tf.device(gpu):
        model.fit_generator(genobj.cond_mean_flow_x2(batch_size=batch_size), 
                            initial_epoch=initial_epoch,
                            epochs=nb_epochs,
                            callbacks=[save_callback],
                            steps_per_epoch=steps_per_epoch,
                            verbose=1)





class Generator():
    """
    Generators
    """

    def __init__(self, x_k=None, atlas_vol_k=None, y_k=None,
                 npz_varname=None, randomize=True, rand_seed=None,
                 onehot_classes=None, idxprob=None):
        self.x_k = x_k  # assumed k-size [nb_items, *vol_shape, 1]
        self.vol_shape = None
        if type(self.x_k) is np.ndarray:
            self.vol_shape = self.x_k.shape[1:-1]
            self.numel = self.x_k.shape[0]
        else:
            assert isinstance(self.x_k, (list, tuple)), "x_k should be numpy array or list/tuple"
            self.numel = len(self.x_k)
        self.atlas_k = atlas_vol_k
        self.y_k = y_k
        self.npz_varname = npz_varname
        self.randomize = randomize
        self.rand_seed = rand_seed
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed) # not sure how consistent this is in generators
        self.idx = [-1]
        self.onehot_classes = onehot_classes
        self.idxprob = idxprob  # prob to allot
        if self.idxprob is not None:
            assert len(self.idxprob) == self.numel

        if self.vol_shape is None:
            self._get_data([0])

    def atl_data(self, batch_size=32):
        """
        yield batches of (atlas, data)
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            x_sel = self._get_data(idx)

            yield (atlas_k_bs, x_sel)


    def mean_flow_x2(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, zeros, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a,x], [x, zero_flow, zero_flow, zero_flow])


    def mean_flow(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a,x], [x, zero_flow, zero_flow])

    def bidir_mean_flow(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a,x], [x, a, zero_flow, zero_flow])

    def cond(self, batch_size=32):
        """
        yield batches of (cond, atlas)
        
        where cond represent other data about the subject
        """

        while 1:
            idx = self._get_next_idx(batch_size)
            y_sel = self._get_att(idx)

            yield y_sel

    def cond_atl(self, batch_size=32):
        """
        yield batches of (cond, atlas)
        
        where cond represent other data about the subject
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            y_sel = self._get_att(idx)

            yield (y_sel, atlas_k_bs)

    def cond_atl_data(self, batch_size=32):
        """
        yield batches of (cond, atlas, data)
        
        where cond represent other data about the subject
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            x_sel = self._get_data(idx)
            y_sel = self._get_att(idx)

            yield (y_sel, atlas_k_bs, x_sel)

    def cond_mean_flow(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            yield ([c, a, x], [x, zero_flow, zero_flow])


    def cond_mean_flow_x2(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            yield ([c, a, x], [x, zero_flow, zero_flow, zero_flow])

    def cond_bidir_mean_flow(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            if self.vol_shape is not None:  # need to build zero flow after first call in case vol_shape was None
                zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])

            yield ([c, a, x], [x, a, zero_flow, zero_flow])

    def data(self, batch_size):
        while 1:
            idx = self._get_next_idx(batch_size)
            yield self._get_data(idx)

    def _get_next_idx(self, batch_size):
        if self.randomize:
            if self.idxprob is None:
                idx = np.random.randint(self.numel, size=batch_size)
            else:
                idx = np.random.choice(range(self.numel), size=batch_size, p=self.idxprob)
        else:
            idx = np.arange(self.idx[-1]+1, self.idx[-1] + batch_size + 1)
            idx = np.mod(idx, self.numel)
        self.idx = idx
        return idx

    def _get_att(self, idx):
        if isinstance(self.y_k, (list, tuple, np.ndarray)):
            y_sel = self.y_k[idx]

            if self.onehot_classes is not None:
                y_sel = init_onehot(y_sel, self.onehot_classes)
        else:
            # assume it's dict
            if isinstance(self.x_k[0], (list, tuple)):
                y_sel = np.stack([self.y_k[self.x_k[i][0]] for i in idx], 0)
            else:
                y_sel = np.stack([self.y_k[self.x_k[i]] for i in idx], 0)
        return y_sel


    def _get_data(self, idx):
        if type(self.x_k) is np.ndarray:
            x = self.x_k[idx, ...]
        else: # assume list of names or list of lists (e.g. if you have [vol, seg])
            if isinstance(self.x_k[0], (list, tuple)):
                batch_data = [[load_image(f, npz_varname=self.npz_varname) for f in self.x_k[i]] for i in idx]
                batch_data = list(map(list, zip(*batch_data)))  # transpose list
                x = [np.stack(f, 0)[..., np.newaxis] for f in batch_data]
                if self.vol_shape is None:
                    self.vol_shape = x[0].shape[1:-1]
            else:
                batch_data = [load_image(self.x_k[i], npz_varname=self.npz_varname) for i in idx]
                x = np.stack(batch_data, 0)[..., np.newaxis]
            if self.vol_shape is None:
                self.vol_shape = x.shape[1:-1]
        return x
            

def load_image(filename, npz_varname=None):
    """
    load and return image

    Available formats:
        image (jpg/jpeg/png)
        npy
        npz (provide npz_varname)
    """

    # get extension
    base, ext = os.path.splitext(filename)

    if ext == '.gz':
        ext = os.path.splitext(base)[-1]
        full_ext = ext + '.gz'

    elif ext in ['.jpg', '.png', '.jpeg']:
        im = plt.imread(filename)

        if len(im.shape) > 2 and im.shape[2] > 1:
            assert im.shape[2] == 3, 'expecting RGB image'
            im = _rgb2gray(im.astype(float))

    elif ext in ['.npz']:
        loaded_file = np.load(filename)
        if npz_varname is None:
            assert len(loaded_file.keys()) == 1, \
                "need to provide npz_varname for file {} since several found".format(filename)
            npz_varname = loaded_file.keys()[0]
        im = loaded_file[npz_varname]
    
    elif ext in ['.npy']:
        im = np.load(filename)

    else:
        raise Exception('extension not understood')

    return im



def load_pheno_csv(filename, verbose=True, tqdm=tqdm):
    """
    load a phenotype csv formatted as 
    filename, pheno1, pheno2, etc
    where each pheno is a float
    """

    dct = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        # print header
        header = next(csv_reader)
        if verbose:
            print(header)
        
        for row in tqdm(csv_reader):
            filename = row[0]
            assert filename not in dct.keys()
            dct[filename] = [float(f) for f in row[1:]]
    
    return dct


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
                        dest="data_loss", default='ncc',
                        help="data_loss: mse of ncc")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.00,  # recommend 1.0 for ncc, 0.01 for mse
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
TRAIN=/path/to/your/training/
MODELS_DIR=/path/to/models/
LOSSES_DIR=/path/to/losses/

# with atlas file
gpu=0; data_loss=ncc; reg_param=1.0; atlas_wt=1; name=cond_atlas_expt_${data_loss}_reg${reg_param}_atlwt${atlas_wt}_releasetest; python train_cond_template.py "$TRAIN/" --model_dir ${MODELS_DIR}/${name} --gpu ${gpu}  --load_model_file ${MODELS_DIR}/miccai2018_s2s_lambda10_sigma0.02_bidir1_double/5000.h5 --model m1double --lambda ${reg_param} --data_loss ${data_loss} --atlas_wt ${atlas_wt} --atlas_file "/home/gid-dalcaav/projects/voxelmorph/code/voxelmorph-sandbox/voxelmorph-pd/data/x_med_similar_stochastic.npz" > ${LOSSES_DIR}/${name}.out 2> ${LOSSES_DIR}/${name}.err &
"""