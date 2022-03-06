#!/usr/bin/env python3

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at:
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm


# reference
ref = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca\n'
    '\tSynthMorph: learning contrast-invariant registration without acquired images\n'
    '\tIEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022\n'
    '\thttps://doi.org/10.1109/TMI.2021.3116879\n'
)


# parse command line
bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'Train a SynthMorph model on images synthesized from label maps. {ref}',
)

# data organization parameters
p.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
p.add_argument('--model-dir', default='models', help='model output directory')
p.add_argument('--log-dir', help='optional TensorBoard log directory')
p.add_argument('--sub-dir', help='optional subfolder for logs and model saves')

# generation parameters
p.add_argument('--same-subj', action='store_true', help='generate image pairs from same label map')
p.add_argument('--blur-std', type=float, default=1, help='maximum blurring std. dev.')
p.add_argument('--gamma', type=float, default=0.25, help='std. dev. of gamma')
p.add_argument('--vel-std', type=float, default=0.5, help='std. dev. of SVF')
p.add_argument('--vel-res', type=float, nargs='+', default=[16], help='SVF scale')
p.add_argument('--bias-std', type=float, default=0.3, help='std. dev. of bias field')
p.add_argument('--bias-res', type=float, nargs='+', default=[40], help='bias scale')
p.add_argument('--out-shape', type=int, nargs='+', help='output shape to pad to''')
p.add_argument('--out-labels', default='fs_labels.npy', help='labels to optimize, see README')

# training parameters
p.add_argument('--gpu', type=str, default='0', help='ID of GPU to use')
p.add_argument('--epochs', type=int, default=1500, help='training epochs')
p.add_argument('--batch-size', type=int, default=1, help='batch size')
p.add_argument('--init-weights', help='optional weights file to initialize with')
p.add_argument('--save-freq', type=int, default=20, help='epochs between model saves')
p.add_argument('--reg-param', type=float, default=1., help='regularization weight')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
p.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
p.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

# network architecture parameters
p.add_argument('--int-steps', type=int, default=5, help='number of integration steps')
p.add_argument('--enc', type=int, nargs='+', default=[64] * 4, help='U-Net encoder filters')
p.add_argument('--dec', type=int, nargs='+', default=[64] * 6, help='U-Net decorder filters')

arg = p.parse_args()


# TensorFlow handling
device, nb_devices = vxm.tf.utils.setup_device(arg.gpu)
assert np.mod(arg.batch_size, nb_devices) == 0, \
    f'batch size {arg.batch_size} not a multiple of the number of GPUs {nb_devices}'
assert tf.__version__.startswith('2'), f'TensorFlow version {tf.__version__} is not 2 or later'


# prepare directories
if arg.sub_dir:
    arg.model_dir = os.path.join(arg.model_dir, arg.sub_dir)
os.makedirs(arg.model_dir, exist_ok=True)

if arg.log_dir:
    if arg.sub_dir:
        arg.log_dir = os.path.join(arg.log_dir, arg.sub_dir)
    os.makedirs(arg.log_dir, exist_ok=True)


# labels and label maps
labels_in, label_maps = vxm.py.utils.load_labels(arg.label_dir)
gen = vxm.generators.synthmorph(
    label_maps,
    batch_size=arg.batch_size,
    same_subj=arg.same_subj,
    flip=True,
)
in_shape = label_maps[0].shape

if arg.out_labels.endswith('.npy'):
    labels_out = sorted(x for x in np.load(arg.out_labels) if x in labels_in)
elif arg.out_labels.endswith('.pickle'):
    with open(arg.out_labels, 'rb') as f:
        labels_out = {k: v for k, v in pickle.load(f).items() if k in labels_in}
else:
    labels_out = labels_in


# model configuration
gen_args = dict(
    in_shape=in_shape,
    out_shape=arg.out_shape,
    in_label_list=labels_in,
    out_label_list=labels_out,
    warp_std=arg.vel_std,
    warp_res=arg.vel_res,
    blur_std=arg.blur_std,
    bias_std=arg.bias_std,
    bias_res=arg.bias_res,
    gamma_std=arg.gamma,
)

reg_args = dict(
    int_steps=arg.int_steps,
    int_resolution=2,
    svf_resolution=2,
    nb_unet_features=(arg.enc, arg.dec),
)


# build model
strategy = 'MirroredStrategy' if nb_devices > 1 else 'get_strategy'
with getattr(tf.distribute, strategy)().scope():

    # generation
    gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
    gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
    ima_1, map_1 = gen_model_1.outputs
    ima_2, map_2 = gen_model_2.outputs

    # registration
    inputs = gen_model_1.inputs + gen_model_2.inputs
    reg_args['inshape'] = ima_1.shape[1:-1]
    reg_args['input_model'] = tf.keras.Model(inputs, outputs=(ima_1, ima_2))
    model = vxm.networks.VxmDense(**reg_args)
    flow = model.references.pos_flow
    pred = vxm.layers.SpatialTransformer(interp_method='linear', name='pred')([map_1, flow])

    # losses and compilation
    const = tf.ones(shape=arg.batch_size // nb_devices)
    model.add_loss(vxm.losses.Dice().loss(map_2, pred) + const)
    model.add_loss(vxm.losses.Grad('l2', loss_mult=arg.reg_param).loss(None, flow))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arg.lr))
    model.summary()


# callbacks
steps_per_epoch = 100
save_name = os.path.join(arg.model_dir, '{epoch:05d}.h5')
save = tf.keras.callbacks.ModelCheckpoint(
    save_name,
    save_freq=steps_per_epoch * arg.save_freq,
)
callbacks = [save]

if arg.log_dir:
    log = tf.keras.callbacks.TensorBoard(
        log_dir=arg.log_dir,
        write_graph=False,
    )
    callbacks.append(log)


# initialize and fit
if arg.init_weights:
    model.load_weights(arg.init_weights)
model.save(save_name.format(epoch=arg.init_epoch))
model.fit(
    gen,
    initial_epoch=arg.init_epoch,
    epochs=arg.epochs,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    verbose=arg.verbose,
)
print(f'\nThank you for using SynthMorph! {ref}')
