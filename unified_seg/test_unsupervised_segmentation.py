"""
Uses a network trained by train_unsupervised_segmentation to segment a new scan.

Unsupervised deep learning for Bayesian brain MRI segmentation
A.V. Dalca, E. Yu, P. Golland, B. Fischl, M.R. Sabuncu, J.E. Iglesias
Under Review. arXiv https://arxiv.org/abs/1904.11319
"""

# py imports
import os
import sys
import time
from argparse import ArgumentParser

# third party
import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils

# project
sys.path.append('../src/')
import networks

def test_unsupervised_segmentation(image_to_segment,
                                   out_seg,
                                   gpu_id,
                                   atlas_file,
                                   mapping_file,
                                   model,
                                   model_file,
                                   out_deformed_atlas,
                                   out_posteriors,
                                   out_warp,
                                   out_stats,
                                   init_stats,
                                   stat_post_warp,
                                   gpu_max_labels,
                                   time_seg):
    """
    model test function
    :param gpu_id: integer specifying the gpu to use
    :param image_to_segment: file name of image to be segmented
    :param atlas_file: file with probabilistic atlas (coregistered to images)
    :param mapping_file: file with mapping from labels to tissue types
    :param model: registration (voxelmorph) model: vm1, vm2, or vm2double
    :param model file: file with weights of the neural net
    :param out_seg: file with the output segmentation
    :param out_deformed_atlas: file with the deformed atlas
    :param out_posteriors: file with the output posteriors
    :param out_warp: file with the atlas deformation field
    :param out_stats: file with the estiamted Gaussian parameters (means, variances)
    :param init_stats: file with guesses for means and log-variances (vectors init_mu, init_sigma)
    :param stat_post_warp: set to 1  to use warped atlas to estimate Gaussian parameters
    """

    assert model_file, "A model file is necessary"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data and atlas
    img, img_nii = load_data(image_to_segment)
    atlas_full, _ = load_data(atlas_file)
    vol_shape = img.shape[1:-1]
       
    # Also: group labels in tissue types, if necessary
    if mapping_file is None:
        atlas_vol = atlas_full
        nb_labels = atlas_vol.shape[-1]

    else:
        mapping = np.load(mapping_file)['mapping'].astype('int').flatten()
        assert len(mapping) == atlas_full.shape[-1], \
            'mapping shape %d is inconsistent with atlas shape %d' % (len(mapping), atlas_full.shape[-1])
        nb_labels = 1 + np.max(mapping)
        atlas_vol = np.zeros([1, *atlas_full.shape[1:-1], nb_labels.astype('int')])
        for j in range(np.max(mapping.shape)):
            atlas_vol[0, ..., mapping[j]] = atlas_vol[0, ..., mapping[j]] + atlas_full[0, ..., j]

    with tf.device(gpu):

        # UNET filters for voxelmorph-1 and voxelmorph-2,
        # building on architectures presented in CVPR 2018
        nf_enc = [16, 32, 32, 32]
        if model == 'vm1':
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif model == 'vm2':
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:  # 'vm2double':
            nf_enc = [f * 2 for f in nf_enc]
            nf_dec = [f * 2 for f in [32, 32, 32, 32, 32, 16, 16]]

        # load guesses for means and variances
        init_mu = np.load(init_stats)['init_mu']
        assert nb_labels == len(init_mu), \
            '# labels %d does not match # mus %d' % (nb_labels, len(init_mu))
        init_sigma = np.load(init_stats)['init_std']
        assert nb_labels == len(init_sigma), \
            '# labels %d does not match # sigmas %d' % (nb_labels, len(init_sigma))

        # Prepare net and models
        net = networks.cvpr2018_net_probatlas(vol_shape, nf_enc, nf_dec, nb_labels,
                                              diffeomorphic=True,
                                              full_size=False,
                                              init_mu=init_mu,
                                              init_sigma=init_sigma,
                                              stat_post_warp=(stat_post_warp > 0))
        net.load_weights(model_file)

        model = keras.models.Model(net.inputs,
                                   [net.get_layer('unsup_likelihood').output,
                                    net.get_layer('comb_mu').output,
                                    net.get_layer('comb_sigma').output,
                                    net.outputs[-1]])

        # get k-functions for posterior
        fns = posterior_Kfn(vol_shape, mapping, max_feats=gpu_max_labels, norm_post=False)
        


        # predict log likelihood and flow
        # note that first time this is run, it will take significantly 
        # longer (e.g. 12 sec) than the rest (e.g. 2 sec)
        outputs = model.predict([img, atlas_vol])
        [ull_pred, mus, sigmas, flow] = [f[0,...] for f in outputs]
        atlas_full = atlas_full[0, ...]

        # compute posteriors on GPU
        post, warped_atlas = posterior(atlas_full, ull_pred, flow, fns, gpu_max_labels)
       
        # segmentation
        # requires ~0.5 sec on CPU with tested 3D volumes.
        # this can also be computed on the GPU as part of tehe fns, which would make this 
        # part take nearly no time, but would require the full posterior estimation on 
        # the GPU in a single run. Some GPUs have difficulty with this. 
        seg = post.argmax(-1)

        # timing. Note this is done after the first run of each model, to eliminate starting overhead. 
        # subsequence segmentations would not need this overhead.
        if time_seg:
            import timeit
            def timeme():
                outputs = model.predict([img, atlas_vol])
                [ull_pred, mus, sigmas, flow] = [f[0,...] for f in outputs]
                post, warped_atlas = posterior(atlas_full, ull_pred, flow, fns, gpu_max_labels)
                seg = post.argmax(-1)

            times = timeit.repeat(timeme, globals=globals(), number=1, repeat=3)
            print('Segmentation time: mean {:.2f}   min {:.2f}'.format(np.mean(times), np.min(times)))  

    # write results to disk as needed
    robust_save_file(seg.astype('float'), out_seg, img_nii)
    robust_save_file(warped_atlas, out_deformed_atlas, img_nii)
    if out_posteriors:
        normalized_posterior = posterior / (1e-12 + np.sum(posterior, -1, keepdims=True))
        robust_save_file(normalized_posterior, out_posteriors, img_nii)
    robust_save_file(flow, out_warp, img_nii)
    
    # also save some stats
    if out_stats is not None:
        np.savez_compressed(out_stats, means=mus, log_variances=sigmas)


def posterior(atlas_full, ull_pred, flow, fns, max_feats):
    """
    gpu-based implementation of posterior
    given original full atlas and unnormalized log likelihood, warps atlas and computes (normalized) posterior

    variables should be normal format, *not* keras format (i.e. not in batch)

    unfortunately, since full atlases can be quite large (many labels),
    we loop over groups of at most `max_feats` labels and stack at the end
    """

    # run through label groups
    # creating lists and concatenating at the end seems to be more efficient than
    # filling in rows of data of a large array.
    post = []
    warped_atlas = []
    for li, i in enumerate(range(0, atlas_full.shape[-1], max_feats)):
        slc = slice(i, min(i + max_feats, atlas_full.shape[-1]))
        po, wa = fns[li]([atlas_full[...,slc], ull_pred, flow])
        post.append(po)
        warped_atlas.append(wa)

    return np.concatenate(post, -1), np.concatenate(warped_atlas, -1)


def posterior_Kfn(vol_shape, mapping, max_feats=None, norm_post=True):
    """
    return a keras (gpu-runnable) function that, given original full atlas and 
    unnormalized log likelihood (from model), warps atlas and computes (normalized) posterior
    
    atlas_full: [*vol_shape, nb_full_labels]
    ull_shape: [*vol_shape, nb_merged_labels]
    flow_shape: [*vol_shape, ndims]
    norm_post (True): normalize posterior? Thi sis faster on GPU, so if possible should be set to True
    max_feats (None): since atlas van be very large, warping full atlas can run out of memory on current GPUs. 
        Providing a number here will avoid OOM error, and will return several keras functions that each provide 
        the posterior computation for at most max_feats nb_full_labels. Stacking the result of calling these
        functions will provide an *unnormalized* posterior (since it can't properly normalize)
    """
    nb_full_labels = len(mapping)
    nb_labels = np.max(mapping) + 1

    # compute maximum features and whether to return single
    return_single_fn = max_feats is None
    if max_feats is None:
        max_feats = nb_full_labels
    else:
        assert not norm_post, 'cannot do normalized posterior if providing max_feats'

    # prepare ull and 
    ull = tf.placeholder(tf.float32, vol_shape + (nb_labels, ))
    input_flow = tf.placeholder(tf.float32, vol_shape + (len(vol_shape), ))
    ul_pred = K.exp(ull)
    
    fns = []
    for i in range(0, nb_full_labels, max_feats):
        end = min(i+max_feats, nb_full_labels)
        this_atlas_full_shape = vol_shape + (end - i, )
        this_mapping = mapping[i:end]

        # prepare atlas input
        input_atlas = tf.placeholder(tf.float32, this_atlas_full_shape)

        # warp atlas
        warped = nrn_utils.transform(input_atlas, input_flow, interp_method='linear', indexing='ij')

        # normalized posterior
        post_lst = [ul_pred[..., this_mapping[j]] * warped[..., j] for j in range(len(this_mapping))]
        posterior = K.stack(post_lst, -1)
        
        fns.append(K.function([input_atlas, ull, input_flow], [posterior, warped]))

    if return_single_fn:
        if norm_post:
            posterior = posterior / (1e-12 + K.sum(posterior, -1, keepdims=True))
        fns = fns.append(K.function([input_atlas, ull, input_flow], [posterior, warped]))
        
    # return function
    return fns


def robust_save_file(data, filename, nii):
    """
    save file depending on whether the filename is provided, the filename extension, 
    and whether nifti was passed in as input
    """
    if filename is not None:
        if str(filename).endswith('npz'):
            np.savez_compressed(filename, vol_data=data)
        else:
            if nii is not None:
                nib.save(nib.Nifti1Image(np.squeeze(data), nii.affine), filename)
            else:
                nib.save(nib.Nifti1Image(np.squeeze(data), None), filename)


def load_data(filename, keras_format=True):
    """
    load data (npz or nii) from filename
    """
    if str(filename).endswith('npz'):
        img_nii = None
        img = np.load(filename)['vol_data']
    else:
        img_nii = nib.load(filename)
        img = img_nii.get_data()
    
    # add dimensions
    if keras_format:
        if img.ndim == 3:
            img = img[..., np.newaxis]
        img = img[np.newaxis, ...]

    # return
    return img, img_nii


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # positional arguments
    parser.add_argument("image_to_segment", type=str, default=None,
                        help="file name of image to be segmented")
    parser.add_argument("out_seg", type=str, default=None,
                        help="file name for output segmentation")

    # optional arguments
    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default='../data/prob_atlas_41_class.npz',
                        help="file with probabilistic atlas")
    parser.add_argument("--mapping_file", type=str,
                        dest="mapping_file", default='../data/atlas_class_mapping.npz',
                        help="file with mapping between labels and tissue types")
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2', 'vm2double'], default='vm2',
                        help="Voxelmorph architecture")
    parser.add_argument("--model_file", type=str,
                        dest="model_file", default='../models/unified_hardcodedstats_lambda_0.01_lr_0.001/60.h5',
                        help="models h5 file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--stat_post_warp", type=int,
                        dest="stat_post_warp", default=True,
                        help="compute Gaussian stats pre- (0) or post-warp (1)")
    parser.add_argument("--out_deformed_atlas", type=str, default=None,
                        dest="out_deformed_atlas", help="file name for output deformed atlas")
    parser.add_argument("--out_warp", type=str, default=None,
                        dest="out_warp", help="file name for output deformation field")
    parser.add_argument("--out_stats", type=str, default=None,
                        dest="out_stats", help="file name for estimated means and variances")
    parser.add_argument("--out_posteriors", type=str, default=None,
                        dest="out_posteriors", help="file name for label posterior volumes")
    parser.add_argument("--init_stats",
                        dest="init_stats", default='../data/meanstats_T1_WARP.npz',
                        help="npz with guesses for initial stats (vectors init_mu and init_sigma)")
    parser.add_argument("--gpu_max_labels",
                        dest="gpu_max_labels", default=21,
                        help="number of maximum labels to warp on the gpu at one time, limited by memory")
    parser.add_argument("--time_seg",
                        dest="time_seg", default=False,
                        help="Whether to run timing analysis for run (after model creation, etc)")

    args = parser.parse_args()
    test_unsupervised_segmentation(**vars(args))
