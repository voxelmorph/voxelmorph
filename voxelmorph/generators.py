import os
import sys
import numpy as np

from . import utils


def subj2atlas(volnames, atlas, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for subject to atlas registration.

    Parameters:
        volnames: List of volume files to load.
        atlas: Atlas volume data.
        bidir: Also yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to volgen.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(volnames, batch_size=batch_size, **kwargs)
    while True:
        subj = next(gen)[0]
        invols  = [subj, atlas]
        outvols = [atlas, subj] if bidir else [atlas]
        if not no_warp:
            outvols.append(zeros)
        yield (invols, outvols)


def subj2subj(volnames, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for subject to subject registration.

    Parameters:
        volnames: List of volume files to load.
        bidir: Also yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to volgen.
    """
    zeros = None
    gen = volgen(volnames, batch_size=batch_size, **kwargs)
    while True:
        subj1 = next(gen)[0]
        subj2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                subj1 = subj2
            else:
                subj2 = subj1

        if not no_warp and zeros is None:
            # cache zeros
            shape = subj1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols  = [subj1, subj2]
        outvols = [subj2, subj1] if bidir else [subj2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def volgen(vol_names, batch_size=1, return_segs=False, np_var='vol_data', pad_shape=None, zoom=1):
    """
    Generator for random volume loading. Corresponding segmentations are additionally
    loaded if return_segs is set to True.

    Parameters:
        volnames: List of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations by replacing any filename
            references of 'norm' to 'aseg'. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol_data'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        zoom: Volume resize factor. Default is 1.
    """
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        load_params = dict(np_var=np_var, add_axes=True, pad_shape=pad_shape, zoom=zoom)

        # load volumes and concatenate
        imgs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            seg_names = [vol_names[i].replace('norm', 'aseg') for i in indices]
            segs = [utils.load_volfile(s, **load_params) for s in seg_names]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)

# TODO swich zoom back to resize
