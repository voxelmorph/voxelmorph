"""
tensorflow/keras utilities for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

# internal python imports
import os
import warnings

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

# local imports
import neurite as ne
from .. import layers


def setup_device(gpuid=None):
    """
    Configures the appropriate TF device from a cuda device string.
    Returns the device id and total number of devices.
    """

    if gpuid is not None and not isinstance(gpuid, str):
        gpuid = str(gpuid)

    if gpuid is not None:
        nb_devices = len(gpuid.split(','))
    else:
        nb_devices = 1

    if gpuid is not None and (gpuid != '-1'):
        device = '/gpu:' + gpuid
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

        # GPU memory configuration differs between TF 1 and 2
        if hasattr(tf, 'ConfigProto'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            tf.keras.backend.set_session(tf.Session(config=config))
        else:
            tf.config.set_soft_device_placement(True)
            for pd in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(pd, True)
    else:
        device = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return device, nb_devices


def value_at_location(x, single_vol=False, single_pts=False, force_post_absolute_val=True):
    """
    Extracts value at given point.

    TODO: needs documentation
    """

    # vol is batch_size, *vol_shape, nb_feats
    # loc_pts is batch_size, nb_surface_pts, D or D+1
    vol, loc_pts = x

    fn = lambda y: ne.utils.interpn(y[0], y[1])
    z = tf.map_fn(fn, [vol, loc_pts], fn_output_signature=tf.float32)

    if force_post_absolute_val:
        z = K.abs(z)

    return z


###############################################################################
# deformation utilities
###############################################################################


def transform(vol, loc_shift, interp_method='linear', fill_value=None,
              shift_center=True, shape=None):
    """Apply affine or dense transforms to images in N dimensions.

    Essentially interpolates the input ND tensor at locations determined by
    loc_shift. The latter can be an affine transform or dense field of location
    shifts in the sense that at location x we now have the data from x + dx, so
    we moved the data.

    Parameters:
        vol: tensor or array-like structure  of size vol_shape or
            (*vol_shape, C), where C is the number of channels.
        loc_shift: Affine transformation matrix of shape (N, N+1) or a shift
            volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
            where C is the number of channels, and D is the dimensionality
            D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
            transform applies to all channels of the input tensor.
        interp_method: 'linear' or 'nearest'.
        fill_value: Value to use for points sampled outside the domain. If
            None, the nearest neighbors will be used.
        shift_center: Shift grid to image center when converting affine
            transforms to dense transforms. Assumes the input and output spaces are identical.
        shape: ND output shape used when converting affine transforms to dense
            transforms. Includes only the N spatial dimensions. If None, the
            shape of the input image will be used. Incompatible with `shift_center=True`.

    Returns:
        Tensor whose voxel values are the values of the input tensor
        interpolated at the locations defined by the transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    if shape is not None and shift_center:
        raise ValueError('`shape` option incompatible with `shift_center=True`')

    # convert data type if needed
    ftype = tf.float32
    if not tf.is_tensor(vol) or not vol.dtype.is_floating:
        vol = tf.cast(vol, ftype)
    if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
        loc_shift = tf.cast(loc_shift, ftype)

    # convert affine to location shift (will validate affine shape)
    if is_affine_shape(loc_shift.shape):
        loc_shift = affine_to_dense_shift(loc_shift,
                                          shape=vol.shape[:-1] if shape is None else shape,
                                          shift_center=shift_center)

    # parse spatial location shape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing='ij')  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def batch_transform(vol, loc_shift, batch_size=None, interp_method='linear', fill_value=None):
    """ apply transform along batch. Compared to _single_transform, reshape inputs to move the 
    batch axis to the feature/channel axis, then essentially apply single transform, and 
    finally reshape back. Need to know/fix batch_size.

    Important: loc_shift is currently implemented only for shape [B, *new_vol_shape, C, D]. 
        to implement loc_shift size [B, *new_vol_shape, D] (as transform() supports), 
        we need to figure out how to deal with the second-last dimension.

    Other notes:
        - We couldn't use ne.utils.flatten_axes() because that computes the axes size from
          tf.shape(), whereas we get the batch size as an input to avoid 'None'.
        - There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
          indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
          removed it in favor of default ij-indexing to minimize the potential for confusion.

    Args:
        vol (Tensor): volume with size vol_shape or [B, *vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [B, *new_vol_shape, C, D]
            where C is the number of channels, and D is the dimensionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    # input management
    ndim = len(vol.shape) - 2
    assert ndim in range(1, 4), 'Dimension {} can only be in [1, 2, 3]'.format(ndim)
    vol_shape_tf = tf.shape(vol)

    if batch_size is None:
        batch_size = vol_shape_tf[0]
        assert batch_size is not None, 'batch_transform: provide batch_size or valid Tensor shape'
    else:
        tf.debugging.assert_equal(vol_shape_tf[0],
                                  batch_size,
                                  message='Tensor has wrong batch size '
                                  '{} instead of {}'.format(vol_shape_tf[0], batch_size))
    BC = batch_size * vol.shape[-1]

    assert len(loc_shift.shape) == ndim + 3, \
        'vol dim {} and loc dim {} are not appropriate'.format(ndim + 2, len(loc_shift.shape))
    assert loc_shift.shape[-1] == ndim, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(ndim, vol.shape[:-1], loc_shift.shape[-1])

    # reshape vol [B, *vol_shape, C] --> [*vol_shape, C * B]
    vol_reshape = K.permute_dimensions(vol, list(range(1, ndim + 2)) + [0])
    vol_reshape = K.reshape(vol_reshape, list(vol.shape[1:ndim + 1]) + [BC])

    # reshape loc_shift [B, *vol_shape, C, D] --> [*vol_shape, C * B, D]
    loc_reshape = K.permute_dimensions(loc_shift, list(range(1, ndim + 2)) + [0] + [ndim + 2])
    loc_reshape_shape = list(vol.shape[1:ndim + 1]) + [BC] + [loc_shift.shape[ndim + 2]]
    loc_reshape = K.reshape(loc_reshape, loc_reshape_shape)

    # transform (output is [*vol_shape, C*B])
    vol_trf = transform(vol_reshape,
                        loc_reshape,
                        interp_method=interp_method,
                        fill_value=fill_value)

    # reshape vol back to [*vol_shape, C, B]
    new_shape = tf.concat([vol_shape_tf[1:], vol_shape_tf[0:1]], 0)
    vol_trf_reshape = K.reshape(vol_trf, new_shape)

    # reshape back to [B, *vol_shape, C]
    return K.permute_dimensions(vol_trf_reshape, [ndim + 1] + list(range(ndim + 1)))


def compose(transforms, interp_method='linear', shift_center=True, shape=None):
    """
    Compose a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = compose([A, B, C])

    Parameters:
        transforms: List or tuple of affine and/or dense transforms to compose.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        shift_center: Shift grid to image center when converting matrices to dense transforms.
        shape: ND output shape used for converting matrices to dense transforms. Includes only the
            N spatial dimensions. Only used once, if the rightmost transform is a matrix. If None
            or if the rightmost transform is a warp, the shape of the rightmost warp will be used.
            Incompatible with `shift_center=True`.

    Returns:
        Composed affine or dense transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if len(transforms) == 0:
        raise ValueError('Compose transform list cannot be empty')

    curr = None
    for next in reversed(transforms):

        if not tf.is_tensor(next) or not next.dtype.is_floating:
            next = tf.cast(next, tf.float32)

        if curr is None:
            curr = next
            continue

        # Dense warp on left: interpolate. Shape will be ignored unless the current transform is a
        # matrix. Once the current transform is a warp field, it will stay a warp field.
        if not is_affine_shape(next.shape):
            if is_affine_shape(curr.shape):
                curr = affine_to_dense_shift(curr,
                                             shape=next.shape[:-1] if shape is None else shape,
                                             shift_center=shift_center)
            curr += transform(next, curr, interp_method=interp_method)

        # Matrix on left, dense warp on right: matrix-vector product.
        elif not is_affine_shape(curr.shape):
            curr = affine_to_dense_shift(next,
                                         shape=curr.shape[:-1],
                                         shift_center=shift_center,
                                         warp_right=curr)

        # No dense warp: matrix product.
        else:
            next = make_square_affine(next)
            curr = make_square_affine(curr)
            curr = tf.linalg.matmul(next, curr)[:-1]

    return curr


def rescale_dense_transform(transform, factor, interp_method='linear'):
    """
    Rescales a dense transform. This involves resizing and rescaling the vector field.

    Parameters:
        transform: A dense warp of shape [..., D1, ..., DN, N].
        factor: Scaling factor.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
    """

    def single_batch(trf):
        if factor < 1:
            trf = ne.utils.resize(trf, factor, interp_method=interp_method)
            trf = trf * factor
        else:
            # multiply first to save memory (multiply in smaller space)
            trf = trf * factor
            trf = ne.utils.resize(trf, factor, interp_method=interp_method)
        return trf

    # enable batched or non-batched input
    if len(transform.shape) > (transform.shape[-1] + 1):
        rescaled = tf.map_fn(single_batch, transform)
    else:
        rescaled = single_batch(transform)

    return rescaled


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in tensorflow

    Aside from directly using tensorflow's numerical integration odeint(), also implements 
    "scaling and squaring", and quadrature. Note that the diff. equation given to odeint
    is the one used in quadrature.   

    Parameters:
        vec: the Tensor field to integrate. 
            If vol_size is the size of the intrinsic volume, and vol_ndim = len(vol_size),
            then vector shape (vec_shape) should be 
            [vol_size, vol_ndim] (if stationary)
            [vol_size, vol_ndim, nb_time_steps] (if time dependent)
        time_dep: bool whether vector is time dependent
        method: 'scaling_and_squaring' or 'ss' or 'ode' or 'quadrature'

        if using 'scaling_and_squaring': currently only supports integrating to time point 1.
            nb_steps: int number of steps. Note that this means the vec field gets broken
            down to 2**nb_steps. so nb_steps of 0 means integral = vec.

        if using 'ode':
            out_time_pt (optional): a time point or list of time points at which to evaluate
                Default: 1
            init (optional): if using 'ode', the initialization method.
                Currently only supporting 'zero'. Default: 'zero'
            ode_args (optional): dictionary of all other parameters for 
                tf.contrib.integrate.odeint()

    Returns:
        int_vec: integral of vector field.
        Same shape as the input if method is 'scaling_and_squaring', 'ss', 'quadrature', 
        or 'ode' with out_time_pt not a list. Will have shape [*vec_shape, len(out_time_pt)]
        if method is 'ode' with out_time_pt being a list.

    Todo:
        quadrature for more than just intrinsically out_time_pt = 1
    """

    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2**nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2**nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])

            disp = svec[0, :]

        else:
            vec = vec / (2**nb_steps)
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        # TODO: could output more than a single timepoint!
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: transform(vec, disp)

        # process time point.
        out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
        out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
        len_out_time_pt = out_time_pt.get_shape().as_list()[0]
        assert len_out_time_pt is not None, 'len_out_time_pt is None :('
        # initializing with something like tf.zeros(1) gives a control flow issue.
        z = out_time_pt[0:1] * 0.0
        K_out_time_pt = K.concatenate([z, out_time_pt], 0)

        # enable a new integration function than tf.contrib.integrate.odeint
        odeint_fn = tf.contrib.integrate.odeint
        if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
            odeint_fn = kwargs['odeint_fn']

        # process initialization
        if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
            disp0 = vec * 0  # initial displacement is 0
        else:
            raise ValueError('non-zero init for ode method not implemented')

        # compute integration with odeint
        if 'ode_args' not in kwargs.keys():
            kwargs['ode_args'] = {}
        disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
        disp = K.permute_dimensions(disp[1:len_out_time_pt + 1, :], [*range(1, len(disp.shape)), 0])

        # return
        if len_out_time_pt == 1:
            disp = disp[..., 0]

    return disp


def point_spatial_transformer(x, single=False, sdt_vol_resize=1):
    """
    Transforms surface points with a given deformation.
    Note that the displacement field that moves image A to image B will be "in the space of B".
    That is, `trf(p)` tells you "how to move data from A to get to location `p` in B". 
    Therefore, that same displacement field will warp *landmarks* in B to A easily 
    (that is, for any landmark `L(p)`, it can easily find the appropriate `trf(L(p))`
    via interpolation.

    TODO: needs documentation
    """

    # surface_points is a N x D or a N x (D+1) Tensor
    # trf is a *volshape x D Tensor
    surface_points, trf = x
    trf = trf * sdt_vol_resize
    surface_pts_D = surface_points.get_shape().as_list()[-1]
    trf_D = trf.get_shape().as_list()[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = K.expand_dims(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]

    # just need to interpolate.
    # at each location determined by surface point, figure out the trf...
    # note: if surface_points are on the grid, gather_nd should work as well
    fn = lambda x: ne.utils.interpn(x[0], x[1])
    diff = tf.map_fn(fn, [trf, surface_points], fn_output_signature=tf.float32)
    ret = surface_points + diff

    if surface_pts_D == trf_D + 1:
        ret = tf.concat((ret, li_surface_pts), -1)

    return ret


# TODO: needs work

def keras_transform(img, trf, interp_method='linear', rescale=None):
    """
    Applies a transform to an image. Note that inputs and outputs are
    in tensor format i.e. (batch, *imshape, nchannels).

    # TODO: it seems that the main addition of this function of the SpatialTransformer 
    # or the transform function is integrating it with the rescale operation? 
    # This needs to be incorporated.
    """
    img_input = tf.keras.Input(shape=img.shape[1:])
    trf_input = tf.keras.Input(shape=trf.shape[1:])
    trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)
    y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, trf_scaled])
    return tf.keras.Model([img_input, trf_input], y_img).predict([img, trf])


###############################################################################
# affine utilities
###############################################################################


def is_affine_shape(shape):
    """
    Determine whether the given shape (single-batch) represents an N-dimensional affine matrix of
    shape (M, N + 1), with `N in (2, 3)` and `M in (N, N + 1)`.

    Parameters:
        shape: Tuple or list of integers excluding the batch dimension.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False


def validate_affine_shape(shape):
    """
    Validate whether the input shape represents a valid affine matrix of shape (..., M, N + 1),
    where N is the number of dimensions, and M is N or N + 1. Throws an error if the shape is
    invalid.

    Parameters:
        shape: Tuple or list of integers.
    """
    ndim = shape[-1] - 1
    rows = shape[-2]
    if ndim not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, got {ndim}D')
    if rows not in (ndim, ndim + 1):
        raise ValueError(f'{ndim}D affine matrix must have {ndim} or {ndim + 1} rows, got {rows}.')


def make_square_affine(mat):
    """
    Convert an ND affine matrix of shape (..., N, N + 1) to square shape (..., N + 1, N + 1).

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., N + 1, N + 1).
    """
    validate_affine_shape(mat.shape)
    if mat.shape[-2] == mat.shape[-1]:
        return mat

    # Support dynamic shapes by keeping them in tensors.
    shape_input = tf.shape(mat)
    shape_batch = shape_input[:-2]
    shape_zeros = tf.concat((shape_batch, (1,), shape_input[-2:-1]), axis=0)
    shape_one = tf.concat((shape_batch, (1, 1)), axis=0)

    # Append last row.
    zeros = tf.zeros(shape_zeros, dtype=mat.dtype)
    one = tf.ones(shape_one, dtype=mat.dtype)
    row = tf.concat((zeros, one), axis=-1)
    return tf.concat((mat, row), axis=-2)


def affine_add_identity(mat):
    """
    Add the identity matrix to an N-dimensional 'shift' affine.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    rows, ndp1 = mat.shape[-2:]
    return mat + tf.eye(ndp1)[:rows]


def affine_remove_identity(mat):
    """
    Subtract the identity matrix from an N-dimensional affine.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    rows, ndp1 = mat.shape[-2:]
    return mat - tf.eye(ndp1)[:rows]


def invert_affine(mat):
    """
    Compute the multiplicative inverse of an N-dimensional affine matrix.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    rows = mat.shape[-2]
    return tf.linalg.inv(make_square_affine(mat))[..., :rows, :]


def rescale_affine(mat, factor):
    """
    Rescales affine matrix by some factor.

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
        factor: Zoom factor.
    """
    scaled_translation = tf.expand_dims(mat[..., -1] * factor, -1)
    scaled_matrix = tf.concat([mat[..., :-1], scaled_translation], -1)
    return scaled_matrix


def affine_to_dense_shift(matrix, shape, shift_center=True, warp_right=None):
    """
    Convert N-dimensional (ND) matrix transforms to dense displacement fields.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply matrices to each index coordinate.
        3. Subtract grid.

    Parameters:
        matrix: Affine matrix of shape (..., M, N + 1), where M is N or N + 1. Can have any batch
            dimensions.
        shape: ND shape of the output space.
        shift_center: Shift grid to image center.
        warp_right: Right-compose the matrix transform with a displacement field of shape
            (..., *shape, N), with batch dimensions broadcastable to those of `matrix`.

    Returns:
        Dense shift (warp) of shape (..., *shape, N).

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # coordinate grid
    mesh = (tf.range(s, dtype=matrix.dtype) for s in shape)
    if shift_center:
        mesh = (m - 0.5 * (s - 1) for m, s in zip(mesh, shape))
    mesh = [tf.reshape(m, shape=(-1,)) for m in tf.meshgrid(*mesh, indexing='ij')]
    mesh = tf.stack(mesh)  # N x nb_voxels
    out = mesh

    # optionally right-compose with warp field
    if warp_right is not None:
        if not tf.is_tensor(warp_right) or warp_right.dtype != matrix.dtype:
            warp_right = tf.cast(warp_right, matrix.dtype)
        flat_shape = tf.concat((tf.shape(warp_right)[:-1 - ndims], (-1, ndims)), axis=0)
        warp_right = tf.reshape(warp_right, flat_shape)  # ... x nb_voxels x N
        out += tf.linalg.matrix_transpose(warp_right)  # ... x N x nb_voxels

    # compute locations, subtract grid to obtain shift
    out = matrix[..., :ndims, :-1] @ out + matrix[..., :ndims, -1:]  # ... x N x nb_voxels
    out = tf.linalg.matrix_transpose(out - mesh)  # ... x nb_voxels x N

    # restore shape
    shape = tf.concat((tf.shape(matrix)[:-2], (*shape, ndims)), axis=0)
    return tf.reshape(out, shape)  # ... x in_shape x N


def angles_to_rotation_matrix(ang, deg=True, ndims=3):
    """
    Construct N-dimensional rotation matrices from angles, where N is 2 or
    3. The direction of rotation for all axes follows the right-hand rule: the
    thumb being the rotation axis, a positive angle defines a rotation in the
    direction pointed to by the other fingers. Rotations are intrinsic, that
    is, applied in the body-centered frame of reference. The function supports
    inputs with or without batch dimensions.

    In 3D, rotations are applied in the order ``R = X @ Y @ Z``, where X, Y,
    and Z are matrices defining rotations about the x, y, and z-axis,
    respectively.

    Arguments:
        ang: Array-like input angles of shape (..., M), specifying rotations
            about the first M axes of space. M must not exceed N. Any missing
            angles will be set to zero. Lists and tuples will be stacked along
            the last dimension.
        deg: Interpret `ang` as angles in degrees instead of radians.
        ndims: Number of spatial dimensions. Must be 2 or 3.

    Returns:
        mat: Rotation matrices of shape (..., N, N) constructed from `ang`.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(ang, (list, tuple)):
        ang = tf.stack(ang, axis=-1)

    if not tf.is_tensor(ang) or not ang.dtype.is_floating:
        ang = tf.cast(ang, dtype='float32')

    # Add dimension to scalars
    if not ang.shape.as_list():
        ang = tf.reshape(ang, shape=(1,))

    # Validate shape
    num_ang = 1 if ndims == 2 else 3
    shape = ang.shape.as_list()
    if shape[-1] > num_ang:
        raise ValueError(f'Number of angles exceeds value {num_ang} expected for dimensionality.')

    # Set missing angles to zero
    width = np.zeros((len(shape), 2), dtype=np.int32)
    width[-1, -1] = max(num_ang - shape[-1], 0)
    ang = tf.pad(ang, paddings=width)

    # Compute sine and cosine
    if deg:
        ang *= np.pi / 180
    c = tf.split(tf.cos(ang), num_or_size_splits=num_ang, axis=-1)
    s = tf.split(tf.sin(ang), num_or_size_splits=num_ang, axis=-1)

    # Construct matrices
    if ndims == 2:
        out = tf.stack((
            tf.concat([c[0], -s[0]], axis=-1),
            tf.concat([s[0], c[0]], axis=-1),
        ), axis=-2)

    else:
        one, zero = tf.ones_like(c[0]), tf.zeros_like(c[0])
        rot_x = tf.stack((
            tf.concat([one, zero, zero], axis=-1),
            tf.concat([zero, c[0], -s[0]], axis=-1),
            tf.concat([zero, s[0], c[0]], axis=-1),
        ), axis=-2)
        rot_y = tf.stack((
            tf.concat([c[1], zero, s[1]], axis=-1),
            tf.concat([zero, one, zero], axis=-1),
            tf.concat([-s[1], zero, c[1]], axis=-1),
        ), axis=-2)
        rot_z = tf.stack((
            tf.concat([c[2], -s[2], zero], axis=-1),
            tf.concat([s[2], c[2], zero], axis=-1),
            tf.concat([zero, zero, one], axis=-1),
        ), axis=-2)
        out = tf.matmul(rot_x, tf.matmul(rot_y, rot_z))

    return tf.squeeze(out) if len(shape) < 2 else out


def params_to_affine_matrix(par,
                            deg=True,
                            shift_scale=False,
                            last_row=False,
                            ndims=3):
    """
    Construct N-dimensional transformation matrices from affine parameters,
    where N is 2 or 3. The transforms operate in a right-handed frame of
    reference, with right-handed intrinsic rotations (see
    angles_to_rotation_matrix for details), and are constructed by matrix
    product ``T @ R @ S @ E``, where T, R, S, and E are matrices representing
    translation, rotation, scale, and shear, respectively. The function
    supports inputs with or without batch dimensions.

    Arguments:
        par: Array-like input parameters of shape (..., M), defining an affine
            transformation in N-D space. The size M of the right-most dimension
            must not exceed ``N * (N + 1)``. This axis defines, in order:
            translations, rotations, scaling, and shearing parameters. In 3D,
            for example, the first three indices specify translations along the
            x, y, and z-axis, and similarly for the remaining parameters. Any
            missing parameters will bet set to identity. Lists and tuples will
            be stacked along the last dimension.
        deg: Interpret input angles as specified in degrees instead of radians.
        shift_scale: Add 1 to any specified scaling parameters. May be
            desirable when the input parameters are estimated by a network.
        last_row: Append the last row and return a full matrix.
        ndims: Number of dimensions. Must be 2 or 3.

    Returns:
        mat: Affine transformation matrices of shape (..., N, N + 1) or
            (..., N + 1, N + 1), depending on `last_row`. The left-most
            dimensions depend on the input shape.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(par, (list, tuple)):
        par = tf.stack(par, axis=-1)

    if not tf.is_tensor(par) or not par.dtype.is_floating:
        par = tf.cast(par, dtype='float32')

    # Add dimension to scalars
    if not par.shape.as_list():
        par = tf.reshape(par, shape=(1,))

    # Validate shape
    num_par = 6 if ndims == 2 else 12
    shape = par.shape.as_list()
    if shape[-1] > num_par:
        raise ValueError(f'Number of params exceeds value {num_par} expected for dimensionality.')

    # Set defaults if incomplete and split by type
    width = np.zeros((len(shape), 2), dtype=np.int32)
    splits = (2, 1) * 2 if ndims == 2 else (3,) * 4
    for i in (2, 3, 4):
        width[-1, -1] = max(sum(splits[:i]) - shape[-1], 0)
        default = 1. if i == 3 and not shift_scale else 0.
        par = tf.pad(par, paddings=width, constant_values=default)
        shape = par.shape.as_list()
    shift, rot, scale, shear = tf.split(par, num_or_size_splits=splits, axis=-1)

    # Construct shear matrix
    s = tf.split(shear, num_or_size_splits=splits[-1], axis=-1)
    one, zero = tf.ones_like(s[0]), tf.zeros_like(s[0])
    if ndims == 2:
        mat_shear = tf.stack((
            tf.concat([one, s[0]], axis=-1),
            tf.concat([zero, one], axis=-1),
        ), axis=-2)
    else:
        mat_shear = tf.stack((
            tf.concat([one, s[0], s[1]], axis=-1),
            tf.concat([zero, one, s[2]], axis=-1),
            tf.concat([zero, zero, one], axis=-1),
        ), axis=-2)

    mat_scale = tf.linalg.diag(scale + 1. if shift_scale else scale)
    mat_rot = angles_to_rotation_matrix(rot, deg=deg, ndims=ndims)
    out = tf.matmul(mat_rot, tf.matmul(mat_scale, mat_shear))

    # Append translations
    shift = tf.expand_dims(shift, axis=-1)
    out = tf.concat((out, shift), axis=-1)

    # Append last row: store shapes as tensors to support batched inputs
    if last_row:
        shape_batch = tf.shape(shift)[:-2]
        shape_zeros = tf.concat((shape_batch, (1,), splits[:1]), axis=0)
        zeros = tf.zeros(shape_zeros, dtype=shift.dtype)
        shape_one = tf.concat((shape_batch, (1,), (1,)), axis=0)
        one = tf.ones(shape_one, dtype=shift.dtype)
        row = tf.concat((zeros, one), axis=-1)
        out = tf.concat([out, row], axis=-2)

    return tf.squeeze(out) if len(shape) < 2 else out


def rotation_matrix_to_angles(mat, deg=True):
    """Compute Euler angles from an N-dimensional rotation matrix.

    We apply right-handed intrinsic rotations as R = X @ Y @ Z, where X, Y,
    and Z are matrices describing rotations about the x, y, and z-axis,
    respectively (see angles_to_rotation_matrix). Labeling these axes with
    indices 1-3 in the 3D case, we decompose the matrix

            [            c2*c3,             −c2*s3,      s2]
        R = [ s1*s2*c3 + c1*s3,  −s1*s2*s3 + c1*c3,  −s1*c2],
            [−c1*s2*c3 + s1*s3,   c1*s2*s3 + s1*c3,   c1*c2]

    where si and ci are the sine and cosine of the angle of rotation about
    axis i. When the angle of rotation about the y-axis is 90 or -90 degrees,
    the system loses one degree of freedom, and the solution is not unique.
    In this gimbal lock case, we set the angle `ang[0]` to zero and solve for
    `ang[2]`.

    Arguments:
        mat: Array-like input matrix to derive rotation angles from, of shape
            (..., N, N + 1) or (..., N + 1, N + 1), where N is 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        ang: Tensor of shape (..., M) holding the derived rotation angles. The
        size M of the right-most dimension is 3 in 3D and 1 in 2D.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if not tf.is_tensor(mat) or mat.dtype != tf.float32:
        mat = tf.cast(mat, tf.float32)

    # Input shape.
    num_dim = mat.shape[-1]
    assert num_dim in (2, 3), f'only 2D and 3D supported'
    assert mat.shape[-2] == num_dim, 'invalid matrix shape'

    # Clip input to inverse trigonometric functions as rounding errors can
    # move them out of the interval [-1, 1].
    clip = lambda x: tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)

    if num_dim == 2:
        y = clip(mat[..., 1, -2])
        x = clip(mat[..., 0, -2])
        ang = tf.atan2(y, x)[..., tf.newaxis]

    else:
        ang2 = tf.asin(clip(mat[..., 0, 2]))

        # Case abs(ang2) == 90 deg. Make ang1 zero as solution is not unique.
        ang1_a = tf.zeros_like(ang2)
        ang3_a = tf.atan2(y=clip(mat[..., 1, 0]), x=clip(mat[..., 1, 1]))

        # Case abs(ang2) != 90 deg. Use safe divide, as we will always compute
        # both cases, even if c2 is zero.
        c2 = tf.cos(ang2)
        y = tf.math.divide_no_nan(-mat[..., 1, 2], c2)
        x = tf.math.divide_no_nan(mat[..., 2, 2], c2)
        ang1_b = tf.atan2(clip(y), clip(x))
        y = tf.math.divide_no_nan(-mat[..., 0, 1], c2)
        x = tf.math.divide_no_nan(mat[..., 0, 0], c2)
        ang3_b = tf.atan2(clip(y), clip(x))

        # Choose between cases.
        is_case = tf.abs((tf.abs(ang2) - 0.5 * np.pi)) < 1e-6
        ang1 = tf.where(is_case, ang1_a, ang1_b)
        ang3 = tf.where(is_case, ang3_a, ang3_b)
        ang = tf.stack((ang1, ang2, ang3), axis=-1)

    if deg:
        ang *= 180 / np.pi
    return ang


def affine_matrix_to_params(mat, deg=True):
    """Derive affine parameters from an N-dimensional transformation matrix.

    The affine transform operates in a right-handed frame of reference, with
    right-handed intrinsic rotations (see params_to_affine_matrix).

    Arguments:
        mat: Array-like input matrix to derive affine parameters from, of
            shape (..., N, N + 1) or (..., N + 1, N + 1), as the last row
            always is ``(*[0] * N, 1)``. N can be 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        par: Tensor of shape (..., K) holding the affine parameters derived
            from `mat`, where K is ``N * (N + 1)``. The parameters along the
            last axis represent, in order: translation, rotation, scaling, and
            shear. In 3D, for example, the first three indices specify
            translations along the x, y, and z-axis, and similarly for the
            remaining indices.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if not tf.is_tensor(mat) or mat.dtype != tf.float32:
        mat = tf.cast(mat, tf.float32)

    # Input shape.
    num_dim = mat.shape[-1] - 1
    assert num_dim in (2, 3), f'invalid dimensionality {num_dim}'
    assert mat.shape[-2] - num_dim in (0, 1), f'invalid shape {mat.shape}'

    # Translation and scaling. Fix negative determinants.
    shift = mat[..., :num_dim, -1]
    mat = mat[..., :num_dim, :num_dim]
    lower = tf.linalg.cholesky(tf.linalg.matrix_transpose(mat) @ mat)
    scale = tf.linalg.diag_part(lower)
    scale0 = scale[..., 0] * tf.sign(tf.linalg.det(mat))
    scale = tf.concat((scale0[..., tf.newaxis], scale[..., 1:]), axis=-1)

    # Strip scaling. Shear as upper triangular part.
    strip = tf.linalg.diag(scale)
    upper = tf.linalg.matrix_transpose(lower)
    upper = tf.linalg.inv(strip) @ upper
    flat_shape = tf.concat((tf.shape(scale0), [num_dim ** 2]), axis=0)
    upper = tf.reshape(upper, flat_shape)
    ind = (1,) if num_dim == 2 else (1, 2, 5)
    shear = tf.gather(upper, ind, axis=-1)

    # Rotations after stripping scale and shear. Treat shape as a tensor to
    # support dynamically sized input matrices.
    zero_shape = tf.concat((tf.shape(scale0), [(num_dim - 1) * 3]), axis=0)
    zero = tf.zeros(zero_shape)
    par = tf.concat((zero, scale, shear), axis=-1)
    strip = params_to_affine_matrix(par, ndims=num_dim)[..., :-1]
    mat = mat @ tf.linalg.inv(strip)
    rot = rotation_matrix_to_angles(mat, deg=deg)

    return tf.concat((shift, rot, scale, shear), axis=-1)


def fit_affine(x_source, x_target, weights=None):
    """Fit an affine transform between two sets of corresponding points.

    Fit an N-dimensional affine transform between two sets of M corresponding
    points in an ordinary or weighted least-squares sense. Note that when
    working with images, source coordinates correspond to the target image and
    vice versa.

    Arguments:
        x_source: Array-like source coordinates of shape (..., M, N).
        x_target: Array-like target coordinates of shape (..., M, N).
        weights: Optional array-like weights of shape (..., M) or (..., M, 1).

    Returns:
        mat: Affine transformation matrix of shape (..., N, N + 1), fitted such
            that ``x_t = mat[..., :-1] @ x_s + mat[..., -1:]``, where x_s is
            ``x_s = tf.linalg.matrix_transpose(x_t)``, and similarly for x_t
            and `x_target`. The last row of `mat` is omitted as it is always
            ``(*[0] * N, 1)``.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    shape = tf.concat((tf.shape(x_target)[:-1], [1]), axis=0)
    ones = tf.ones(shape, dtype=x_target.dtype)
    x = tf.concat((x_target, ones), axis=-1)
    x_transp = tf.linalg.matrix_transpose(x)
    y = x_source

    if weights is not None:
        if len(weights.shape) == len(x.shape):
            weights = weights[..., 0]
        x_transp *= tf.expand_dims(weights, axis=-2)

    beta = tf.linalg.inv(x_transp @ x) @ x_transp @ y
    return tf.linalg.matrix_transpose(beta)
