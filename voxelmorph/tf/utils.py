"""
tensorflow/keras utilities for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# internal python imports
import os

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

# local imports
import neurite as ne
from . import layers


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


###############################################################################
# deformation utilities
###############################################################################


def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    TODO: 
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    if isinstance(volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()
    
    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        shape1 = '(%d x %d)' % (nb_dims + 1, nb_dims + 1)
        shape2 = '(%d x %s)' % (nb_dims, nb_dims + 1)
        true_shape = str(affine_matrix.shape)
        raise Exception('Affine shape should match %s or %s, but got: %s' % (shape1, shape2, true_shape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)  
    mesh = [tf.cast(f, 'float32') for f in mesh]
    
    if shift_center:
        mesh = [mesh[f] - (volshape[f]-1)/2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)


def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.
    
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes

    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def compose(disp_1, disp_2, indexing='ij'):
    """
    compose two dense deformations specified by their displacements

    We have two fields
        A --> B (so field is in space of B)
        and
        B --> C (so field is in the space of C)
    this function gives a new warp field
        A --> C (so field is in the sapce of C)

    Parameters:
        disp_1: first displacement (A-->B) with size [*vol_shape, ndims]
        disp_2: second  displacement (B-->C) with size [*vol_shape, ndims]
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'

    Returns:
        composed field disp_3 which takes data from A to C
    """

    assert indexing == 'ij', "currently only ij indexing is implemented in compose"

    return disp_2 + transform(disp_1, disp_2, interp_method='linear', indexing=indexing)


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

            svec = svec/(2**nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2,:], svec[0::2,:])

            disp = svec[0, :]

        else:
            vec = vec/(2**nb_steps)
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        # TODO: could output more than a single timepoint!
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec/nb_steps

        if time_dep:
            disp = vec[...,0]
            for si in range(nb_steps-1):
                disp += transform(vec[...,si+1], disp)
        else:
            disp = vec
            for _ in range(nb_steps-1):
                disp += transform(vec, disp)

    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: transform(vec, disp)  

        # process time point.
        out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
        out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
        len_out_time_pt = out_time_pt.get_shape().as_list()[0]
        assert len_out_time_pt is not None, 'len_out_time_pt is None :('
        z = out_time_pt[0:1]*0.0  # initializing with something like tf.zeros(1) gives a control flow issue.
        K_out_time_pt = K.concatenate([z, out_time_pt], 0)

        # enable a new integration function than tf.contrib.integrate.odeint
        odeint_fn = tf.contrib.integrate.odeint
        if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
            odeint_fn = kwargs['odeint_fn']

        # process initialization
        if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
            disp0 = vec*0  # initial displacement is 0
        else:
            raise ValueError('non-zero init for ode method not implemented')

        # compute integration with odeint
        if 'ode_args' not in kwargs.keys():
            kwargs['ode_args'] = {}
        disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
        disp = K.permute_dimensions(disp[1:len_out_time_pt+1, :], [*range(1,len(disp.shape)), 0])

        # return
        if len_out_time_pt == 1: 
            disp = disp[...,0]

    return disp


def is_affine(shape):
    """
    TODO: needs documentation
    """

    return len(shape) == 1 or (len(shape) == 2 and shape[0] + 1 == shape[1])


def extract_affine_ndims(shape):
    """
    TODO: needs documentation
    """

    if len(shape) == 1:
        # if vector, just compute ndims since length = N * (N + 1)
        return int((np.sqrt(4 * int(shape[0]) + 1) - 1) / 2)
    else:
        return int(shape[0])


def affine_shift_to_identity(trf):
    """
    TODO: needs documentation
    """

    ndims = extract_affine_ndims(trf.shape.as_list())
    trf = tf.reshape(trf, [ndims, ndims + 1])
    trf = tf.concat([trf, tf.zeros((1, ndims + 1))], axis=0)
    trf += tf.eye(ndims + 1)
    return trf


def affine_identity_to_shift(trf):
    """
    TODO: needs documentation
    """

    ndims = int(trf.shape.as_list()[-1]) - 1
    trf = trf - tf.eye(ndims + 1)
    trf = trf[:ndims, :]
    return tf.reshape(trf, [ndims * (ndims + 1)])


def value_at_location(x, single_vol=False, single_pts=False, force_post_absolute_val=True):
    """
    Extracts value at given point.

    TODO: needs documentation
    """
    
    # vol is batch_size, *vol_shape, nb_feats
    # loc_pts is batch_size, nb_surface_pts, D or D+1
    vol, loc_pts = x

    fn = lambda y: ne.utils.interpn(y[0], y[1])
    z = tf.map_fn(fn, [vol, loc_pts], dtype=tf.float32)

    if force_post_absolute_val:
        z = K.abs(z)

    return z


def point_spatial_transformer(x, single=False, sdt_vol_resize=1):
    """
    Transforms surface points with a given deformation.
    Note that the displacement field that moves image A to image B will be "in the space of B".
    That is, `trf(p)` tells you "how to move data from A to get to location `p` in B". 
    Therefore, that same displacement field will warp *landmarks* in B to A easily 
    (that is, for any landmark `L(p)`, it can easily find the appropriate `trf(L(p))` via interpolation.

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
    diff = tf.map_fn(fn, [trf, surface_points], dtype=tf.float32)
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
