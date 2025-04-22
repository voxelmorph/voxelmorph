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


def generator(labels, batch_size=1):
    rand = np.random.default_rng()
    for out, true in vxm.generators.synthmorph(label_maps, arg.batch_size):
        yield (rand.uniform(size=batch_size), *out), true


# reference
ref = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, A Hoopes, DN Greve, B Fischl, AV Dalca\n'
    '\tAnatomy-aware and acquisition-agnostic joint registration with SynthMorph\n'
    '\tImaging Neuroscience, 2, pp 1-33, 2024\n'
    '\thttps://doi.org/10.1162/imag_a_00197\n'
)


# parse command line
bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'Train a joint SynthMorph model on images synthesized from label maps. {ref}',
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
p.add_argument('--out-labels', default='fs_large21.pickle', help='labels to optimize, see README')

# training parameters
p.add_argument('--gpu', type=str, default='0', help='ID of GPU to use')
p.add_argument('--epochs', type=int, default=10000, help='training epochs')
p.add_argument('--batch-size', type=int, default=1, help='batch size')
p.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
p.add_argument('--init-joint', help='weights file to initialize joint model with')
p.add_argument('--init-affine', help='weights file to initialize affine submodel with')
p.add_argument('--init-deform', help='weights file to initialize deformable submodel with')
p.add_argument('--save-freq', type=int, default=100, help='epochs between model saves')
p.add_argument('--lr', type=float, default=1e-5, help='learning rate')
p.add_argument('--loss-mult', type=float, default=10, help='similarity-loss weight')
p.add_argument('--mid-space', action='store_true', help='compute loss in affine mid-space')
p.add_argument('--freeze-aff', action='store_true', help='freeze the affine weights')
p.add_argument('--freeze-def', action='store_true', help='freeze the hypernetwork weights')
p.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

# affine architecture
p.add_argument('--aff-enc-nf', type=int, nargs='+', default=[256] * 4, help='encoder filters')
p.add_argument('--aff-dec-nf', type=int, nargs='+', default=[256] * 0, help='decoder filters')
p.add_argument('--aff-add-nf', type=int, nargs='+', default=[256] * 4, help='additional filters')
p.add_argument('--aff-num-feat', type=int, default=64, help='number of feature maps')

# deformable architecture
p.add_argument('--enc', type=int, nargs='+', default=[256] * 4, help='encoder filters')
p.add_argument('--dec', type=int, nargs='+', default=[256] * 4, help='decoder filters')
p.add_argument('--add', type=int, nargs='+', default=[256] * 4, help='additional filters')
p.add_argument('--units', type=int, nargs='+', default=[32] * 4, help='hypernetwork units')
p.add_argument('--int-steps', type=int, default=5, help='number of integration steps')

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
gen = generator(label_maps, arg.batch_size)
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
keys = list(f for f in vars(arg) if f.startswith('aff'))
hyp = tf.keras.layers.Input(shape=(1,), name='hyp')
inputs = (hyp, *gen_model_1.inputs, *gen_model_2.inputs)
model = vxm.networks.HyperVxmJoint(
    input_model=tf.keras.Model(inputs, outputs=(hyp, ima_1, ima_2)),
    hyp_units=arg.units,
    enc_nf=arg.enc,
    dec_nf=arg.dec,
    add_nf=arg.add,
    int_steps=arg.int_steps,
    mid_space=arg.mid_space,
    return_trans_to_half_res=True,
    return_def=True,
    **{k.replace('aff_', 'aff.'): vars(arg)[k] for k in keys},
)
total, warp = model.outputs


# moved labels
prop = dict(fill_value=0, shape=total.shape[1:-1], shift_center=False)
scale_down = ne.layers.Constant(value=np.diag((*[2] * len(in_shape), 1)))([])
mov_1 = vxm.layers.SpatialTransformer(**prop)((map_1, total))
map_2 = vxm.layers.SpatialTransformer(**prop)((map_2, scale_down))
out = (hyp, mov_1, map_2, warp)


# weight freezing
model_aff, model_def = (f for f in model.layers if isinstance(f, tf.keras.Model))
model_aff.trainable = not arg.freeze_aff
model_def.trainable = not arg.freeze_def


class AddLoss(tf.keras.layers.Layer):
    def call(self, x):
        hyp, mov_1, map_2, warp = x
        const = tf.zeros((arg.batch_size, 1))
        loss_sim = arg.loss_mult * vxm.losses.MSE().loss(mov_1, map_2) + const
        loss_reg = vxm.losses.Grad('l2').loss(None, warp)
        self.add_loss((1 - hyp) * loss_sim + hyp * loss_reg)
        return x


# loss
model = tf.keras.Model(model.inputs, AddLoss()(out))
optim = tf.keras.optimizers.Adam(learning_rate=arg.lr)
model.compile(optim, jit_compile=False)
model.summary()


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
if arg.init_joint:
    model.load_weights(arg.init_joint)

if arg.init_affine:
    model_aff.load_weights(arg.init_affine)

if arg.init_deform:
    model_def.load_weights(arg.init_deform)


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
