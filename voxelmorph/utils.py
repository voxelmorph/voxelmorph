import os
import numpy as np


def get_backend():
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def load_volfile(filename, np_var='vol_data', add_axes=False):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        vol = nib.load(filename).get_data()
    elif filename.endswith('.npz'):
        if np_var is None:
            np_var = 'vol_data'
        vol = np.load(filename)[np_var]
    else:
        raise ValueError('unknown filetype for %s' % filename)

    return vol[np.newaxis, ..., np.newaxis] if add_axes else vol
