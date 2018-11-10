"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np


def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx])
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'))
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)
            
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)


def load_example_by_name(vol_name, seg_name):
    """
    load a specific volume and segmentation
    """
    X = load_volfile(vol_name)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data' 
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        X = np.load(datafile)['vol_data']

    return X

