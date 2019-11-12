import os
import sys
import numpy as np

from . import utils


def subj2atlas(volnames, volshape, atlas, bidir=False, batch_size=1):
    """Generator for subject to atlas registration."""
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(volnames, batch_size=batch_size)
    while True:
        subj = next(gen)[0]
        invols  = [subj, atlas]
        outvols = [atlas, subj, zeros] if bidir else [atlas, zeros]
        yield (invols, outvols)


def subj2subj(volnames, volshape, bidir=False, batch_size=1):
    """Generator for subject to subject registration."""
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    gen = volgen(volnames, batch_size=batch_size)
    while True:
        subj1 = next(gen)[0]
        subj2 = next(gen)[0]
        invols  = [subj1, subj2]
        outvols = [subj2, subj1, zeros] if bidir else [subj2, zeros]
        yield (invols, outvols)


def volgen(vol_names, batch_size=1, return_segs=False, np_var='vol_data'):
    """Generator for random volume (and segmentaion) loading."""
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        vols = [utils.load_vol(vol_names[i], np_var=np_var, reshape=True) for i in indices]
        x = [np.concatenate(x, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            seg_names = [vol_names[i].replace('norm', 'aseg') for i in indices]
            segs = [utils.load_vol(s, np_var=np_var, reshape=True) for s in seg_names]
            x.append(np.concatenate(segs, axis=0))

        yield tuple(x)
