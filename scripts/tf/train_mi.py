#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import pandas as pd

# disable eager execution
# tf.compat.v1.disable_eager_execution()

# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--config-file', required=True, help='json file to read the parameters')
spec = parser.parse_args()

if 0:
    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', help='optional atlas filename')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                        help='weight of gradient or KL loss (default: 0.01)')
    parser.add_argument('--kl-lambda', type=float, default=10,
                        help='prior lambda regularization for KL loss (default: 10)')
    parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                        help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')  # nopep8
    args = parser.parse_args()
else:
    import json 
    class ArgParser():
        def __init__(self, data):
            self.img_list            = data['img_list']
            self.val_list            = data['val_list']
            self.atlas               = data['atlas']
            self.model_dir           = data['model_dir']
            self.epochs              = data['epochs']
            self.enc                 = data['enc']
            self.dec                 = data['dec']
            self.steps_per_epoch     = data['steps_per_epoch']
            self.gpu                 = data['gpu']
            self.batch_size          = data['batch_size']
            self.initial_epoch       = data['initial_epoch']
            self.lr                  = data['lr']
            self.int_steps           = data['int_steps']
            self.int_downsize        = data['int_downsize']
            self.kl_lambda           = data['kl_lambda']
            self.image_sigma         = data['image_sigma']
            self.image_loss          = data['image_loss']
            self.multichannel        = data['multichannel']
            self.use_probs           = data['use_probs']
            self.bidir               = data['bidir']
            self.lambda_weight       = data['lambda_weight']
            self.img_prefix          = data['img_prefix']
            self.img_suffix          = data['img_suffix']
            self.load_weights        = data['load_weights']
            self.use_validation      = data['use_validation']

    # Opening JSON file
    f = open(spec.config_file)
    data = json.load(f)
    args = ArgParser(data)
    f.close()

print()
print("##############################################################")
print("img_list        type: {} and value: {}".format(type(args.img_list), args.img_list))
print("val_list        type: {} and value: {}".format(type(args.val_list), args.val_list))
print("atlas           type: {} and value: {}".format(type(args.atlas), args.atlas))
print("model_dir       type: {} and value: {}".format(type(args.model_dir), args.model_dir))
print("epochs          type: {} and value: {}".format(type(args.epochs), args.epochs))
print("enc             type: {} and value: {}".format(type(args.enc), args.enc))
print("dec             type: {} and value: {}".format(type(args.dec), args.dec))
print("steps_per_epoch type: {} and value: {}".format(type(args.steps_per_epoch), args.steps_per_epoch))
print("gpu             type: {} and value: {}".format(type(args.gpu), args.gpu))
print("batch_size      type: {} and value: {}".format(type(args.batch_size), args.batch_size))
print("initial_epoch   type: {} and value: {}".format(type(args.initial_epoch), args.initial_epoch))
print("lr              type: {} and value: {}".format(type(args.lr), args.lr))
print("int_steps       type: {} and value: {}".format(type(args.int_steps), args.int_steps))
print("int_downsize    type: {} and value: {}".format(type(args.int_downsize), args.int_downsize))
print("kl_lambda       type: {} and value: {}".format(type(args.kl_lambda), args.kl_lambda))
print("image_sigma     type: {} and value: {}".format(type(args.image_sigma), args.image_sigma))
print("image_loss      type: {} and value: {}".format(type(args.image_loss), args.image_loss))
print("multichannel    type: {} and value: {}".format(type(args.multichannel), args.multichannel))
print("use_probs       type: {} and value: {}".format(type(args.use_probs), args.use_probs))
print("bidir           type: {} and value: {}".format(type(args.bidir), args.bidir))
print("lambda_weight   type: {} and value: {}".format(type(args.lambda_weight), args.lambda_weight))
print("img_prefix      type: {} and value: {}".format(type(args.img_prefix), args.img_prefix))
print("img_suffix      type: {} and value: {}".format(type(args.img_suffix), args.img_suffix))
print("load_weights    type: {} and value: {}".format(type(args.load_weights), args.load_weights))
print("use_validation  type: {} and value: {}".format(type(args.use_validation), args.use_validation))
print("##############################################################")
print()

class GradientAccumulateModel(vxm.networks.VxmDense):
    """
    Model derived from VxmDense to perform gradient accumulation.
    """
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]
        tf.print("Harsha, n_acum_step: ", self.n_acum_step)
        tf.print("Harsha, n_gradients: ", self.n_gradients)

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # normalize accumulated gradients
        """"
        for i in range(len(self.gradient_accumulation)):
            div = self.n_gradients
            norm_value = tf.divide(self.gradient_accumulation[i], tf.cast(div, tf.float32))
            self.gradient_accumulation[i].assign(norm_value)
        """
        
        # apply normalized gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# load and prepare validation data
val_files = vxm.py.utils.read_file_list(args.val_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(val_files) > 0, 'Could not find any validation data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel # [hy23] because ours is grayscale, add_feat_axis := true.

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size,
                                             bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
    val_generator = vxm.generators.scan_to_atlas(val_files, atlas,
                                                 batch_size=args.batch_size,
                                                 bidir=args.bidir,
                                                 add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape and number of features from sampled input

sample_shape = next(generator)[0][0].shape

'''
[hy23]
next(generator)       := (invols, outvols)
next(generator)[0]    := invols            := [scan1, scan2]
next(generator)[0][0] := invols[0]         := [scan1]
'''

inshape = sample_shape[1:-1] # [hy23] 3 dimensions.
nfeats = sample_shape[-1]    # [hy23] := 1

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# device = '/cpu:0'
print("Harsha, device is %s\n".format(device))

assert np.mod(args.batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

print("Harsha, enc_nf is {}".format(enc_nf))
print("Harsha, enc_nf is {}".format(dec_nf))

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# build the model
model = GradientAccumulateModel(
    n_gradients=20, # GCD(560, 100)
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=args.bidir,
    use_probs=args.use_probs,
    int_steps=args.int_steps,
    int_resolution=args.int_downsize,
    src_feats=nfeats,
    trg_feats=nfeats
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE(args.image_sigma).loss
elif args.image_loss == 'mi':
    image_loss_func = vxm.losses.MutualInformation().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if args.bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
if args.use_probs:
    flow_shape = model.outputs[-1].shape[1:-1]
    losses += [vxm.losses.KL(args.kl_lambda, flow_shape).loss]
else:
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]

weights += [args.lambda_weight]

# multi-gpu support
if nb_devices > 1:
    save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
    model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
else:
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq='epoch')

model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

model.summary()

# save starting weights
model.save(save_filename.format(epoch=args.initial_epoch))

# log start time
tstart = tf.timestamp()

print("Harsha, the float precision is {}".format(tf.keras.backend.floatx()))

if(args.use_validation == False):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)
    history = model.fit(generator,
                        initial_epoch=args.initial_epoch,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        callbacks=[save_callback, early_stop_callback],
                        verbose=1
                        )
else:
    print("Harsha, running with validation data.")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1) # default: monitor='val_loss'
    history = model.fit(generator,
                        validation_data=val_generator,
                        initial_epoch=args.initial_epoch,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        validation_steps=100,
                        callbacks=[save_callback],
                        verbose=1
                        )


# log end time
tend = tf.timestamp()

# time spent in training
tspent = tend - tstart
print("Harsha, the training time is {}", tspent)

# log end time
tend = tf.timestamp()

# time spent in training
tspent = tend - tstart
print("Harsha, the training time is {}", tspent)

# convert the history.history dict to a pandas DataFrame:
# https://stackoverflow.com/a/55901240
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'I:\\03.masterarbeit_out\history_mi_lambda_2\history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
