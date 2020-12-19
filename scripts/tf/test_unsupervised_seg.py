#!/usr/bin/env python

"""
Example script to test a segmentation network trained in an unsupervised fashion, using a
probabilistic atlas and unlabeled scans.

If you use this code, please cite the following 
    Unsupervised deep learning for Bayesian brain MRI segmentation 
    A.V. Dalca, E. Yu, P. Golland, B. Fischl, M.R. Sabuncu, J.E. Iglesias 
    MICCAI 2019.
    arXiv https://arxiv.org/abs/1904.11319

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
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()
parser.add_argument('image', help='input image to test')
parser.add_argument('seg', help='output segmentation file')
parser.add_argument('--model', required=True, help='keras model file')
parser.add_argument('--atlas', required=True, help='atlas npz file')
parser.add_argument('--mapping', required=True, help='atlas mapping filename')
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument("--max-feats", default=21,
                    help='number of label posteriors to compute on GPU at once')
parser.add_argument('--warped-atlas', help='save warped atlas to output vol file')
parser.add_argument('--posteriors', help='save posteriors to output vol file')
parser.add_argument('--warp', help='save warp to output vol file')
parser.add_argument('--stats', help='save stats to output npz file')
args = parser.parse_args()

# load reference atlas (group labels in tissue types if necessary)
atlas_full = vxm.py.utils.load_volfile(args.atlas, add_batch_axis=True)
mapping = np.load(args.mapping)['mapping'].astype('int').flatten()
assert len(mapping) == atlas_full.shape[-1], \
    'mapping shape %d is inconsistent with atlas shape %d' % (len(mapping), atlas_full.shape[-1])
nb_labels = int(1 + np.max(mapping))
atlas = np.zeros([*atlas_full.shape[:-1], nb_labels])
for i in range(np.max(mapping.shape)):
    atlas[0, ..., mapping[i]] = atlas[0, ..., mapping[i]] + atlas_full[0, ..., i]

# get input shape
inshape = atlas.shape[1:-1]

# load input scan
image, affine = vxm.py.utils.load_volfile(
    args.image, add_batch_axis=True, add_feat_axis=True, ret_affine=True)


# define an isolated method of computing posteriors
def make_k_functions(vol_shape, mapping, max_feats=None, norm_post=True):
    """
    Utility to build keras (gpu-runnable) functions that will warp the atlas and compute
    posteriors given the original full atlas and unnormalized log likelihood.

    norm_post (True): normalize posterior? Thi sis faster on GPU, so if possible should 
        be set to True
    max_feats (None): since atlas van be very large, warping full atlas can run out of memory on 
        current GPUs. 
        Providing a number here will avoid OOM error, and will return several keras functions that 
        each provide  the posterior computation for at most max_feats nb_full_labels. Stacking the 
        result of calling these functions will provide an *unnormalized* posterior (since it can't
         properly normalize)
    """
    nb_full_labels = len(mapping)
    nb_labels = np.max(mapping) + 1

    # compute maximum features and whether to return single function
    return_single_fn = max_feats is None
    if max_feats is None:
        max_feats = nb_full_labels
    else:
        assert not norm_post, 'cannot do normalized posterior if providing max_feats'

    # prepare ull and
    ull = tf.placeholder(tf.float32, vol_shape + (nb_labels, ))
    input_flow = tf.placeholder(tf.float32, vol_shape + (len(vol_shape), ))
    ul_pred = K.exp(ull)

    funcs = []
    for i in range(0, nb_full_labels, max_feats):
        end = min(i + max_feats, nb_full_labels)
        this_atlas_full_shape = vol_shape + (end - i, )
        this_mapping = mapping[i:end]

        # prepare atlas input
        input_atlas = tf.placeholder(tf.float32, this_atlas_full_shape)

        # warp atlas
        warped = vxm.tf.ne.transform(input_atlas, input_flow, interp_method='linear', indexing='ij')

        # normalized posterior
        post_lst = [ul_pred[..., this_mapping[j]] * warped[..., j]
                    for j in range(len(this_mapping))]
        posterior = K.stack(post_lst, -1)

        funcs.append(K.function([input_atlas, ull, input_flow], [posterior, warped]))

    if return_single_fn:
        if norm_post:
            posterior = posterior / (1e-12 + K.sum(posterior, -1, keepdims=True))
        funcs = funcs.append(K.function([input_atlas, ull, input_flow], [posterior, warped]))

    return funcs


# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

with tf.device(device):

    # get k-functions for posterior
    funcs = make_k_functions(inshape, mapping, max_feats=args.max_feats, norm_post=False)

    # load the model from file
    model = vxm.networks.ProbAtlasSegmentation.load(args.model).get_gaussian_warp_model()

    # predict log likelihood and flow
    outputs = model.predict([image, atlas])

    # remove unused batch dimension
    ull_pred, mus, sigmas, flow = [array[0, ...] for array in outputs]

    # run through label groups and compute posteriors
    posteriors = []
    warped_atlas = []
    total_labels = atlas_full.shape[-1]
    for li, i in enumerate(range(0, total_labels, args.max_feats)):
        slc = slice(i, min(i + args.max_feats, total_labels))
        po, wa = funcs[li]([atlas_full[0, ..., slc], ull_pred, flow])
        posteriors.append(po)
        warped_atlas.append(wa)
    posteriors = np.concatenate(posteriors, -1)
    warped_atlas = np.concatenate(warped_atlas, -1)

    # compute final segmentation
    # note: this requires about 0.5 seconds on the CPU. This can also be computed on the GPU
    # as part of the funcs, (which would be very fast), but would require the full posterior
    # estimation on the GPU in a single run. Some GPUs will have difficulty with this.
    segmentation = posteriors.argmax(-1)

# save segmentation
vxm.py.utils.save_volfile(segmentation.astype('int32'), args.seg, affine)

# save warped atlas
if args.warped_atlas:
    vxm.py.utils.save_volfile(warped_atlas, args.warped_atlas, affine)

# save posteriors
if args.posteriors:
    normalized = posteriors / (1e-12 + np.sum(posteriors, -1, keepdims=True))
    vxm.py.utils.save_volfile(normalized, args.posteriors, affine)

# save warp
if args.warp:
    vxm.py.utils.save_volfile(flow, args.warp, affine)

# save computed stats
if args.stats:
    np.savez_compressed(args.stats, means=mus, log_variances=sigmas)
