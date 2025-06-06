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


import pathlib
import argparse
import numpy as np
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm


# reference
ref = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, A Hoopes, B Fischl, AV Dalca\n'
    '\tAnatomy-specific acquisition-agnostic affine registration learned from fictitious images\n'
    '\tSPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023\n'
    '\thttps://doi.org/10.1117/12.2653251\n'
    '\thttps://synthmorph.io/#papers (PDF)\n'
)


# parse command line
bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'Train an affine SynthMorph model on images synthesized from label maps. {ref}',
)

# data organization
p.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
p.add_argument('--model-dir', type=pathlib.Path, default='models', help='model output directory')
p.add_argument('--log-dir', type=pathlib.Path, help='optional TensorBoard log directory')
p.add_argument('--sub-dir', help='optional subfolder for logs and model saves')

# synthesis
p.add_argument('--shift', type=float, default=30, help='maximum translation amplitude')
p.add_argument('--rotate', type=float, default=45, help='maximum rotation amplitude')
p.add_argument('--scale', type=float, default=0.1, help='maximum scaling offset from 1')
p.add_argument('--shear', type=float, default=0.1, help='maximum shearing amplitude')
p.add_argument('--crop-prob', type=float, default=1, help='edge-cropping probability')
p.add_argument('--blur-max', type=float, default=3.4, help='maximum blurring SD')
p.add_argument('--slice-prob', type=float, default=1, help='downsampling probability')
p.add_argument('--out-shape', type=int, default=[192] * 3, nargs='+', help='synthesis output shape')
p.add_argument('--out-labels', default='fs_lrc.pickle', help='labels to optimize, see README')

# training parameters
p.add_argument('--gpu', type=str, default='0', help='ID of GPU to use')
p.add_argument('--epochs', type=int, default=10000, help='training epochs')
p.add_argument('--batch-size', type=int, default=1, help='batch size')
p.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
p.add_argument('--init-weights', help='weights file to initialize model with')
p.add_argument('--save-freq', type=int, default=100, help='epochs between model saves')
p.add_argument('--lr', type=float, default=1e-5, help='learning rate')
p.add_argument('--mid-space', action='store_true', help='compute loss in affine mid-space')
p.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

# network architecture
p.add_argument('--enc', type=int, nargs='+', default=[256] * 4, help='encoder filters')
p.add_argument('--dec', type=int, nargs='+', default=[256] * 0, help='decoder filters')
p.add_argument('--add', type=int, nargs='+', default=[256] * 4, help='additional filters')
p.add_argument('--feat', type=int, default=64, help='number of feature maps')

arg = p.parse_args()


# TensorFlow
gpu, num_gpu = vxm.tf.utils.setup_device(arg.gpu)
assert tf.__version__.startswith('2'), f'TensorFlow version {tf.__version__} is not 2'


# output directories
if arg.sub_dir:
    arg.model_dir /= arg.sub_dir
    if arg.log_dir:
        arg.log_dir /= arg.sub_dir


# labels
labels_in, label_maps = vxm.py.utils.load_labels(arg.label_dir)
gen = vxm.generators.synthmorph(label_maps, batch_size=arg.batch_size)
in_shape = label_maps[0].shape

labels_out = labels_in
if arg.out_labels:
    labels_out = np.load(arg.out_labels, allow_pickle=True)
    if not isinstance(labels_out, dict):
        labels_out = {i: i for i in labels_out}
    labels_out = {k: v for k, v in labels_out.items() if k in labels_in}


# synthesis
arg_gen = dict(
    in_shape=in_shape,
    out_shape=arg.out_shape,
    labels_in=labels_in,
    labels_out=labels_out,
    aff_shift=arg.shift,
    aff_rotate=arg.rotate,
    aff_scale=arg.scale,
    aff_shear=arg.shear,
    blur_max=arg.blur_max,
    crop_prob=arg.crop_prob,
    slice_prob=arg.slice_prob,
)
gen_model_1 = ne.models.labels_to_image(**arg_gen, id=0)
gen_model_2 = ne.models.labels_to_image(**arg_gen, id=1)
ima_1, map_1 = gen_model_1.outputs
ima_2, map_2 = gen_model_2.outputs


# registration
inputs = (*gen_model_1.inputs, *gen_model_2.inputs)
model = vxm.networks.VxmAffineFeatureDetector(
    input_model=tf.keras.Model(inputs, outputs=(ima_1, ima_2)),
    enc_nf=arg.enc,
    dec_nf=arg.dec,
    add_nf=arg.add,
    num_feat=arg.feat,
    bidir=True,
    make_dense=True,
    return_trans_to_mid_space=arg.mid_space,
    return_trans_to_half_res=True,
)
aff_1, aff_2 = model.outputs


# moved labels
prop = dict(fill_value=0, shape=aff_1.shape[1:-1], shift_center=False)
mov_1 = vxm.layers.SpatialTransformer(**prop)((map_1, aff_1))
mov_2 = vxm.layers.SpatialTransformer(**prop)((map_2, aff_2))

scale_down = ne.layers.Constant(value=np.diag((*[2] * len(in_shape), 1)))([])
map_2 = vxm.layers.SpatialTransformer(**prop)((map_2, scale_down))
out = (mov_1, mov_2 if arg.mid_space else map_2)


class AddLoss(tf.keras.layers.Layer):
    def call(self, x):
        self.add_loss(vxm.losses.MSE().loss(*x))
        return x


# loss
model = tf.keras.Model(model.inputs, AddLoss()(out))
optim = tf.keras.optimizers.Adam(learning_rate=arg.lr)
model.compile(optim, jit_compile=False)
models = [m for m in model.layers if isinstance(m, tf.keras.Model)]
models[0].summary()


# callbacks
steps_per_epoch = 100
save = tf.keras.callbacks.ModelCheckpoint(
    filepath=arg.model_dir / '{epoch:05d}.weights.h5',
    save_freq=steps_per_epoch * arg.save_freq,
    save_weights_only=True,
)
callbacks = [save]

if arg.log_dir:
    log = tf.keras.callbacks.TensorBoard(log_dir=arg.log_dir, write_graph=False)
    callbacks.append(log)


# initialization
if arg.init_weights:
    model.load_weights(arg.init_weights)


# training
model.fit(
    gen,
    initial_epoch=arg.init_epoch,
    epochs=arg.epochs,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    verbose=arg.verbose,
)


print(f'\nThank you for using SynthMorph! {ref}')
