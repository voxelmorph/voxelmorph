import os
import sys
import numpy as np

from . import utils


def volgen(vol_names, batch_size=1, return_segs=False, np_var='vol', pad_shape=None, resize_factor=1, add_feat_axis=True):
    """
    Base generator for random volume loading. Corresponding segmentations are additionally
    loaded if return_segs is set to True. If loading segmentations, it's expected that
    vol_names is a list of npz files with 'vol' and 'seg' arrays.

    Parameters:
        vol_names: List of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    while True:
        # generate <batchsize> random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols  = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols  = [scan, atlas]
        outvols = [atlas, scan] if bidir else [atlas]
        if not no_warp:
            outvols.append(zeros)
        yield (invols, outvols)


def semisupervised(vol_names, labels, atlas_file=None, downsize=2):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. It's
    expected that vol_names and atlas_file are npz files with both 'vol' and 'seg' arrays.

    Parameters:
        vol_names: List of volume npz files to load.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = volgen(vol_names, return_segs=True, np_var='vol')
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    # cache target vols and segs if atlas is supplied
    if atlas_file:
        trg_vol = utils.load_volfile(atlas_file, np_var='vol', add_batch_axis=True, add_feat_axis=True)
        trg_seg = utils.load_volfile(atlas_file, np_var='seg', add_batch_axis=True, add_feat_axis=True)
        trg_seg = split_seg(trg_seg)

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)

        # load target vol and seg (if not provided by atlas)
        if not atlas_file:
            trg_vol, trg_seg = next(gen)
            trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols  = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros,   trg_seg]
        yield (invols, outvols)


def template_creation(vol_names, atlas, bidir=False, batch_size=1, **kwargs):
    """
    Generator for unconditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols  = [atlas, scan]  # TODO: this is opposite of the normal ordering and might be confusing
        outvols = [scan, atlas, zeros, zeros] if bidir else [scan, zeros, zeros]
        yield (invols, outvols)


def conditional_template_creation(vol_names, atlas, attributes, batch_size=1, np_var='vol', pad_shape=None, add_feat_axis=True):
    """
    Generator for conditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        attributes: Dictionary of phenotype data for each vol name.
        batch_size: Batch size. Default is 1.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load pheno from attributes dictionary
        pheno = np.stack([attributes[vol_names[i]] for i in indices], axis=0)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape)
        vols = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = np.concatenate(vols, axis=0)

        invols  = [pheno, atlas, vols]
        outvols = [vols, zeros, zeros, zeros]
        yield (invols, outvols)
