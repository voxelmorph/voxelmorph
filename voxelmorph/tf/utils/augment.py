import numpy as np
import tensorflow as tf


def draw_flip_matrix(grid_shape, shift_center=True, last_row=True, dtype=tf.float32, seed=None):
    """
    Draw a matrix transform that randomly flips axes of N-dimensional space.

    Parameters:
        grid_shape: Spatial shape of the image in voxels, exluding batches and features.
        shift_center: Whether zero is at the center of the grid. Should be identical to the value.
            used for vxm.utils.affine_to_shift or vxm.layers.SpatialTransformer.
        last_row: Append the last row of the transform to return a square matrix.
        dtype: Floating-point output data type.
        seed: Integer for reproducible randomization.

    Returns:
        Flip matrix of shape (M, N + 1), where M is N or N + 1, depending on `last_row`.

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    dtype = tf.dtypes.as_dtype(dtype)
    ndims = len(grid_shape)
    grid_shape = tf.constant(grid_shape, dtype=dtype)

    # Decide which axes to flip.
    rand_bit = tf.greater(tf.random.normal(shape=(ndims,), seed=seed), 0)
    rand_bit = tf.cast(rand_bit, dtype)
    diag = tf.pow(tf.cast(-1, dtype), rand_bit)
    diag = tf.linalg.diag(diag)

    # Account for center shift if needed.
    shift = tf.multiply(grid_shape - 1, rand_bit)
    shift = tf.reshape(shift, shape=(-1, 1))
    if shift_center:
        shift = tf.zeros(shape=(ndims, 1), dtype=dtype)

    # Compose output.
    out = tf.concat((diag, shift), axis=1)
    if last_row:
        row = dtype.as_numpy_dtype((*[0] * ndims, 1))
        row = np.reshape(row, newshape=(1, -1))
        out = tf.concat((out, row), axis=0)

    return out


def draw_swap_matrix(ndims, last_row=True, dtype=tf.float32, seed=None):
    """
    Draw a matrix transform that randomly swaps axes of N-dimensional space.

    Parameters:
        ndims: Number of spatial dimensions, excluding batches and features.
        last_row: Append the last row of the transform to return a square matrix.
        dtype: Floating-point output data type.
        seed: Integer for reproducible randomization.

    Returns:
        Swap matrix of shape (M, N + 1), where M is N or N + 1, depending on `last_row`.

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    dtype = tf.dtypes.as_dtype(dtype)

    mat = tf.eye(ndims, ndims + 1, dtype=dtype)
    mat = tf.random.shuffle(mat, seed=seed)

    row = dtype.as_numpy_dtype((*[0] * ndims, 1))
    row = np.reshape(row, newshape=(1, -1))

    return tf.concat((mat, row), axis=0) if last_row else mat


def draw_affine_params(shift=None,
                       rot=None,
                       scale=None,
                       shear=None,
                       normal_shift=False,
                       normal_rot=False,
                       normal_scale=False,
                       normal_shear=False,
                       shift_scale=False,
                       ndims=3,
                       batch_shape=None,
                       concat=True,
                       dtype=tf.float32,
                       seeds={}):
    """
    Draw translation, rotation, scaling and shearing parameters defining an affine transform in
    N-dimensional space, where N is 2 or 3. Choose parameters wisely: there is no check for
    negative or zero scaling!

    Parameters:
        shift: Translation sampling range x around identity. Values will be sampled uniformly from
            [-x, x]. When sampling from a normal distribution, x is the standard deviation (SD).
            The same x will be used for each dimension, unless an iterable of length N is passed,
            specifying a value separately for each axis. None means 0.
        rot: Rotation sampling range (see `shift`). Accepts only one value in 2D.
        scale: Scaling sampling range x. Parameters will be sampled around identity as for `shift`,
            unless `shift_scale` is set. When sampling normally, scaling parameters will be
            truncated beyond two standard deviations.
        shear: Shear sampling range (see `shift`). Accepts only one value in 2D.
        normal_shift: Sample translations normally rather than uniformly.
        normal_rot: See `normal_shift`.
        normal_scale: Draw scaling parameters from a normal distribution, truncating beyond 2 SDs.
        normal_shear: See `normal_shift`.
        shift_scale: Add 1 to any drawn scaling parameter When sampling uniformly, this will
            result in scaling parameters falling in [1 - x, 1 + x] instead of [-x, x].
        ndims: Number of dimensions. Must be 2 or 3.
        normal: Sample parameters normally instead of uniformly.
        batch_shape: A list or tuple. If provided, the output will have leading batch dimensions.
        concat: Concatenate the output along the last axis to return a single tensor.
        dtype: Floating-point output data type.
        seeds: Dictionary of integers for reproducible randomization. Keywords must be in ('shift',
            'rot', 'scale', 'shear').

    Returns:
        A tuple of tensors with shapes (..., N), (..., M), (..., N), and (..., M) defining
        translation, rotation, scaling, and shear, respectively, where M is 3 in 3D and 1 in 2D.
        With `concat=True`, the function will concatenate the output along the last dimension.

    See also:
        vxm.layers.DrawAffineParams
        vxm.layers.ParamsToAffineMatrix
        vxm.utils.params_to_affine_matrix

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    assert ndims in (2, 3), 'only 2D and 3D supported'
    n = 1 if ndims == 2 else 3

    # Look-up tables.
    splits = dict(shift=ndims, rot=n, scale=ndims, shear=n)
    inputs = dict(shift=shift, rot=rot, scale=scale, shear=shear)
    trunc = dict(shift=False, rot=False, scale=True, shear=False)
    normal = dict(shift=normal_shift, rot=normal_rot, scale=normal_scale, shear=normal_shear)

    # Normalize inputs.
    shapes = {}
    ranges = {}
    for k, n in splits.items():
        x = np.ravel(0 if inputs[k] is None else inputs[k])
        if len(x) == 1:
            x = np.repeat(x, repeats=n)
        assert len(x) == n, f'unexpected number of parameters {len(x)} ({k})'
        ranges[k] = x
        shapes[k] = (n,) if batch_shape is None else tf.concat((batch_shape, [n]), axis=0)

    # Choose distribution.
    def sample(lim, shape, normal, trunc, seed):
        prop = dict(dtype=tf.dtypes.as_dtype(dtype), seed=seed, shape=shape)
        if normal:
            func = 'truncated_normal' if trunc else 'normal'
            prop.update(stddev=lim)
        else:
            func = 'uniform'
            prop.update(minval=-lim, maxval=lim)
        return getattr(tf.random, func)(**prop)

    # Sample parameters.
    par = {}
    seeds = seeds.copy()
    for k, lim in ranges.items():
        par[k] = sample(lim, shapes[k], normal[k], trunc[k], seed=seeds.pop(k, None))
    if shift_scale:
        par['scale'] += 1
    assert not seeds, f'unknown seeds {seeds}'

    # Output.
    order = ('shift', 'rot', 'scale', 'shear')
    out = tuple(par[k] for k in order)
    return tf.concat(out, axis=-1) if concat else out
