import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

from . import neuron as ne
from . import layers


def transform(img, trf, interp_method='linear', rescale=None):
    """
    Applies a transform to an image. Note that inputs and outputs are
    in tensor format i.e. (batch, *imshape, nchannels).
    """
    img_input = tf.keras.Input(shape=img.shape[1:])
    trf_input = tf.keras.Input(shape=trf.shape[1:])
    trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)
    y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, trf_scaled])
    return tf.keras.Model([img_input, trf_input], y_img).predict([img, trf])


def is_affine(shape):
    return len(shape) == 1 or (len(shape) == 2 and shape[0] + 1 == shape[1])


def extract_affine_ndims(shape):
    if len(shape) == 1:
        # if vector, just compute ndims since length = N * (N + 1)
        return int((np.sqrt(4 * int(shape[0]) + 1) - 1) / 2)
    else:
        return int(shape[0])


def affine_shift_to_identity(trf):
    ndims = extract_affine_ndims(trf.shape.as_list())
    trf = tf.reshape(trf, [ndims, ndims + 1])
    trf = tf.concat([trf, tf.zeros((1, ndims + 1))], axis=0)
    trf += tf.eye(ndims + 1)
    return trf


def affine_identity_to_shift(trf):
    ndims = int(trf.shape.as_list()[-1]) - 1
    trf = trf - tf.eye(ndims + 1)
    trf = trf[:ndims, :]
    return tf.reshape(trf, [ndims * (ndims + 1)])


def gaussian_blur(tensor, sigma):
    """
    Blur a tensor of standard Keras shape (batch, space ..., feature) using an
    isotropic Gaussian kernel. Nothing is done if sigma is 0 or None.
    """
    if sigma is None or sigma == 0:
        return tensor
    if sigma < 0:
        raise ValueError(f'Gaussian smoothing sigma {sigma} is less than 0')

    # Make dimensions: batch, feature, space.
    shape = tensor.shape.as_list()
    ind = np.arange(len(shape))
    forward = (ind[0], ind[-1], *ind[1:-1])
    backward = (ind[0], *ind[2:], ind[1])
    tensor = KL.Lambda(lambda x: tf.transpose(x, perm=forward))(tensor)

    # Apply kernel once per batch, feature.
    num_dim = len(shape) - 2
    kernel = ne.utils.gaussian_kernel([sigma] * num_dim)[..., None, None]
    func_feat = lambda x: tf.nn.convolution(x[None,...,None], kernel,
        padding='SAME')[0,...,0]
    func_batch = lambda x: tf.map_fn(func_feat, x, dtype='float32')
    func_tensor = lambda x: tf.map_fn(func_batch, x, dtype='float32')
    tensor = KL.Lambda(func_tensor)(tensor)

    # Restore order of dimensions.
    return KL.Lambda(lambda x: tf.transpose(x, perm=backward))(tensor)


def value_at_location(x, single_vol=False, single_pts=False, force_post_absolute_val=True):
    """
    Extracts value at given point.
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
