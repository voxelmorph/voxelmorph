import os
import numpy as np
import scipy
import csv
import functools

from skimage import measure


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def load_volfile(
        filename,
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False
    ):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format.

    Parameters:
        filename: Filename to load.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1,  0,  0,  0],
                               [ 0,  0,  1,  0],
                               [ 0, -1,  0,  0],
                               [ 0,  0,  0,  1]], dtype=float)
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    """
    if factor == 1:
        return array
    else:
        dim_factors = [factor for _ in array.shape[:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None):
    """
    Converts an affine shift to a matrix (over the identity).
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi,...,0] = clean_seg(x[...,0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)
    
    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')
    
    if not sdt:
        X_dt = np.abs(X_dt)
    
    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[...,0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize) for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def get_surface_pts_per_label(total_nb_surface_pts, layer_edge_ratios):
    """
    Gets the number of surface points per label, given the total number of surface points.
    """
    nb_surface_pts_sel = np.round(np.array(layer_edge_ratios) * total_nb_surface_pts).astype('int')
    nb_surface_pts_sel[-1] = total_nb_surface_pts - int(np.sum(nb_surface_pts_sel[:-1]))
    return nb_surface_pts_sel


def edge_to_surface_pts(X_edges, nb_surface_pts=None):
    """
    Converts edges to surface points.
    """

    # assumes X_edges is NOT in keras form
    surface_pts = np.stack(np.where(X_edges), 0).transpose()
    
    # random with replacements
    if nb_surface_pts is not None:
        chi = np.random.choice(range(surface_pts.shape[0]), size=nb_surface_pts)
        surface_pts = surface_pts[chi,:]

    return surface_pts


def sdt_to_surface_pts(X_sdt, nb_surface_pts, surface_pts_upsample_factor=2, thr=0.50001, resize_fn=None):
    """
    Converts a signed distance transform to surface points.
    """
    us = [surface_pts_upsample_factor] * X_sdt.ndim
    
    if resize_fn is None:
        resized_vol = scipy.ndimage.interpolation.zoom(X_sdt, us, order=1, mode='reflect')
    else:
        resized_vol = resize_fn(X_sdt)
        pred_shape = np.array(X_sdt.shape)*surface_pts_upsample_factor
        assert np.array_equal(pred_shape, resized_vol.shape), 'resizing failed'

    X_edges = np.abs(resized_vol) < thr
    sf_pts = edge_to_surface_pts(X_edges, nb_surface_pts=nb_surface_pts)

    # can't just correct by surface_pts_upsample_factor because of how interpolation works...
    return np.stack([sf_pts[..., f] * (X_sdt.shape[f] - 1) / (X_edges.shape[f] - 1) for f in range(X_sdt.ndim)], -1)
