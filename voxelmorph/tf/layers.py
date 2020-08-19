"""
tensorflow/keras layers for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

License: GPLv3
"""

# internal python imports
import os


# third party
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


# local utils
import neurite as ne
# TODO: simply import utils and use utils.is_affine, etc...
from .utils import is_affine, extract_affine_ndims, affine_shift_to_identity, affine_identity_to_shift
from .utils import transform, integrate_vec, affine_to_shift
from . import utils


class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms. 
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from 
    the identity matrix (unless specified otherwise).

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 add_identity=True,
                 shift_center=True,
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.
            add_identity (default: True): whether the identity matrix is added
                to affine transforms.
            shift_center (default: True): whether the grid is shifted to the center
                of the image when converting affine transforms to warp fields.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.add_identity = add_identity
        self.shift_center = shift_center
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'add_identity': self.add_identity,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')
        
        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1] or [N, N+1]
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        is_matrix = len(trf_shape) == 2 and trf_shape[0] in (self.ndims, self.ndims+1) and trf_shape[1] == self.ndims+1
        self.is_affine = len(trf_shape) == 1 or is_matrix

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d' 
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # convert matrix to warp field
        if self.is_affine:
            ncols = self.ndims + 1
            nrows = self.ndims
            if np.prod(trf.shape.as_list()[1:]) == (self.ndims + 1) ** 2:
                nrows += 1
            if len(trf.shape[1:]) == 1:
                trf = tf.reshape(trf, shape=(-1, nrows, ncols))
            if self.add_identity:
                trf += tf.eye(nrows, ncols, batch_shape=(tf.shape(trf)[0],))
            fun = lambda x: affine_to_shift(x, vol.shape[1:-1], shift_center=self.shift_center)
            trf = tf.map_fn(fun, trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0,:]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)


class VecInt(Layer):
    """
    Vector Integration Layer

    Enables vector integration via several methods 
    (ode or quadrature for time-dependent vector fields, 
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, indexing='ij', method='ss', int_steps=7, out_time_pt=1, 
                 ode_args=None,
                 odeint_fn=None, **kwargs):
        """        
        Parameters:
            method can be any of the methods in neuron.utils.integrate_vec
            indexing can be 'xy' (switches first two dimensions) or 'ij'
            int_steps is the number of integration steps
            out_time_pt is time point at which to output if using odeint integration
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {'rtol':1e-6, 'atol':1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'indexing': self.indexing,
            'method': self.method,
            'int_steps': self.int_steps,
            'out_time_pt': self.out_time_pt,
            'ode_args': self.ode_args,
            'odeint_fn': self.odeint_fn,
        })
        return config

    def build(self, input_shape):
        # confirm built
        self.built = True
        
        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d'
                % (trf_shape[-1], len(trf_shape) - 2))

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi_gpu models...
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], '_keras_shape'):
            loc_shift._keras_shape = inputs[0]._keras_shape
        
        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        if len(inputs) > 1:
            assert self.out_time_pt is None, 'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out = tf.map_fn(self._single_int, [loc_shift] + inputs[1:], dtype=tf.float32)
        if hasattr(inputs[0], '_keras_shape'):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                      nb_steps=self.int_steps,
                      ode_args=self.ode_args,
                      out_time_pt=out_time_pt,
                      odeint_fn=self.odeint_fn)
       

# full wording.
VecIntegration = VecInt


class RescaleTransform(Layer):
    """ 
    Rescales a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)
        self.init_config = {'zoom_factor': zoom_factor, 'interp_method': interp_method, **kwargs}

    def get_config(self):
        return self.init_config

    def build(self, input_shape):

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('RescaleTransform must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        self.is_affine = is_affine(input_shape[1:])
        self.ndims = extract_affine_ndims(input_shape[1:]) if self.is_affine else int(input_shape[-1])

        super().build(input_shape)

    def call(self, inputs):

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            trf = inputs[0]
        else:
            trf = inputs

        if self.is_affine:
            return tf.map_fn(self._single_affine_rescale, trf, dtype=tf.float32)
        else:
            if self.zoom_factor < 1:
                # resize
                trf = ne.layers.Resize(self.zoom_factor, name=self.name + '_resize')(trf)
                return ne.layers.RescaleValues(self.zoom_factor, name=self.name + '_rescale')(trf)
            else:
                # multiply first to save memory (multiply in smaller space)
                trf = ne.layers.RescaleValues(self.zoom_factor, name=self.name + '_rescale')(trf)
                return ne.layers.Resize(self.zoom_factor, name=self.name + '_resize')(trf)

    def _single_affine_rescale(self, trf):
        matrix = affine_shift_to_identity(trf)
        scaled_translation = tf.expand_dims(matrix[:, -1] * self.zoom_factor, 1)
        scaled_matrix = tf.concat([matrix[:, :-1], scaled_translation], 1)
        return affine_identity_to_shift(scaled_matrix)

    def compute_output_shape(self, input_shape):
        if self.is_affine:
            return (input_shape[0], self.ndims * (self.ndims + 1))
        else:
            output_shape = [int(dim * self.zoom_factor) for dim in input_shape[1:-1]]
            return (input_shape[0], *output_shape, self.ndims)


class ComposeTransform(Layer):
    """ 
    Composes two transforms specified by their displacements. Affine transforms
    can also be provided. If only affines are provided, the returned transform
    is an affine, otherwise it will return a displacement field.

    We have two transforms:

    A --> B (so field/result is in the space of B)
    B --> C (so field/result is in the space of C)
    
    This layer composes a new transform.

    A --> C (so field/result is in the space of C)
    """

    def build(self, input_shape, **kwargs):

        if len(input_shape) != 2:
            raise Exception('ComposeTransform must be called on a input list of length 2.')

        # figure out if any affines were provided
        self.input_1_is_affine = is_affine(input_shape[0][1:])
        self.input_2_is_affine = is_affine(input_shape[1][1:])
        self.return_affine = self.input_1_is_affine and self.input_2_is_affine

        if self.return_affine:
            # extract dimension information from affine
            shape = input_shape[0][1:]
            if len(shape) == 1:
                # if vector, just compute ndims since length = N * (N + 1)
                self.ndims = int((np.sqrt(4 * int(shape[0]) + 1) - 1) / 2)
            else:
                self.ndims = int(shape[0])
        else:
            # extract shape information whichever is the dense transform
            dense_idx = 1 if self.input_1_is_affine else 0
            self.ndims = input_shape[dense_idx][-1]
            self.volshape = input_shape[dense_idx][1:-1]

        super().build(input_shape)

    def call(self, inputs):
        """
        Parameters
            inputs: list with two dense deformations
        """
        assert len(inputs) == 2, 'inputs has to be len 2, found: %d' % len(inputs)

        input_1 = inputs[0]
        input_2 = inputs[1]

        if self.return_affine:
            return tf.map_fn(self._single_affine_compose, [input_1, input_2], dtype=tf.float32)
        else:
            # if necessary, convert affine to dense transform
            if self.input_1_is_affine:
                input_1 = AffineToDense(self.volshape)(input_1)
            elif self.input_2_is_affine:
                input_2 = AffineToDense(self.volshape)(input_2)

            # dense composition
            return tf.map_fn(self._single_dense_compose, [input_1, input_2], dtype=tf.float32)

    def _single_dense_compose(self, inputs):
        return utils.compose(inputs[0], inputs[1])

    def _single_affine_compose(self, inputs):
        affine_1 = affine_shift_to_identity(inputs[0])
        affine_2 = affine_shift_to_identity(inputs[1])
        composed = tf.linalg.matmul(affine_1, affine_2)
        return affine_identity_to_shift(composed)

    def compute_output_shape(self, input_shape):
        if self.return_affine:
            return (input_shape[0], self.ndims * (self.ndims + 1))
        else:
            return (input_shape[0], *self.volshape, self.ndims)


class AffineToDense(Layer):
    """
    Converts an affine transform to a dense shift transform. The affine must represent
    the shift between images (not over the identity).
    """

    def __init__(self, volshape, **kwargs):
        self.volshape = volshape
        self.ndims = len(volshape)
        super().__init__(**kwargs)

    def build(self, input_shape):

        shape = input_shape[1:]

        if len(shape) == 1:
            ex = self.ndims * (self.ndims + 1)            
            if shape[0] != ex:
                raise ValueError('Expected flattened affine of len %d but got %d' % (ex, shape[0]))

        if len(shape) == 2 and (shape[0] != self.ndims or shape[1] != self.ndims + 1):
            err_msg = 'Expected affine matrix of shape %s but got %s'
            raise ValueError(err_msg % (str((self.ndims, self.ndims + 1)), str(shape)))

        super().build(input_shape)

    def call(self, trf):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """

        return tf.map_fn(self._single_aff_to_shift, trf, dtype=tf.float32)

    def _single_aff_to_shift(self, trf):
        # go from vector to matrix
        if len(trf.shape) == 1:
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        trf += tf.eye(self.ndims + 1)[:self.ndims, :]  # add identity, hence affine is a shift from identity
        return affine_to_shift(trf, self.volshape, shift_center=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.volshape, self.ndims)


class InvertAffine(Layer):
    """
    Inverts an affine transform. The transform must represent
    the shift between images (not over the identity).
    """

    def build(self, input_shape):
        self.ndims = extract_affine_ndims(input_shape[1:])
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, trf):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """
        return tf.map_fn(self._single_invert, trf, dtype=tf.float32)

    def _single_invert(self, trf):
        matrix = affine_shift_to_identity(trf)
        inverse = tf.linalg.inv(matrix)
        return affine_identity_to_shift(inverse)


class AffineTransformationsToMatrix(Layer):
    """
    Computes the corresponding (flattened) affine from a vector of transform
    components. The components are in the order of (translation, rotation), so the
    input must a 1D array of length (ndim * 2).

    TODO: right now only supports 4x4 transforms - make this dimension-independent
    TODO: allow for scaling and shear components
    """

    def __init__(self, ndims, scale=False, **kwargs):
        self.ndims = ndims
        self.scale = scale
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('rigid registration is limited to 3D for now')

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, vector):
        """
        Parameters
            vector: tensor of affine components
        """
        return tf.map_fn(self._single_conversion, vector, dtype=tf.float32)

    def _single_conversion(self, vector):

        if self.ndims == 3:
            # extract components of input vector
            translation = vector[:3]
            angle_x = vector[3]
            angle_y = vector[4]
            angle_z = vector[5]

            # x rotation matrix
            cosx  = tf.math.cos(angle_x)
            sinx  = tf.math.sin(angle_x)
            x_rot = tf.convert_to_tensor([
                [1,    0,     0],
                [0, cosx, -sinx],
                [0, sinx,  cosx]
            ], name='x_rot')

            # y rotation matrix
            cosy  = tf.math.cos(angle_y)
            siny  = tf.math.sin(angle_y)
            y_rot = tf.convert_to_tensor([
                [cosy,  0, siny],
                [0,     1,    0],
                [-siny, 0, cosy]
            ], name='y_rot')
            
            # z rotation matrix
            cosz  = tf.math.cos(angle_z)
            sinz  = tf.math.sin(angle_z)
            z_rot = tf.convert_to_tensor([
                [cosz, -sinz, 0],
                [sinz,  cosz, 0],
                [0,        0, 1]
            ], name='z_rot')

            # compose matrices
            t_rot = tf.tensordot(x_rot, y_rot, 1)
            m_rot = tf.tensordot(t_rot, z_rot, 1)

            # build scale matrix
            s = vector[6] if self.scale else 1.0
            m_scale = tf.convert_to_tensor([
                [s, 0, 0],
                [0, s, 0],
                [0, 0, s]
            ], name='scale')

        elif self.ndims == 2:
            # extract components of input vector
            translation = vector[:2]
            angle = vector[2]

            # rotation matrix
            cosz  = tf.math.cos(angle)
            sinz  = tf.math.sin(angle)
            m_rot = tf.convert_to_tensor([
                [cosz, -sinz],
                [sinz,  cosz]
            ], name='rot')

            s = vector[3] if self.scale else 1.0
            m_scale = tf.convert_to_tensor([
                [s, 0],
                [0, s]
            ], name='scale')

        # we want to encode shift transforms, so remove identity
        m_rot -= tf.eye(self.ndims)

        # scale the matrix
        m_rot = tf.tensordot(m_rot, m_scale, 1)

        # concat the linear translation
        matrix = tf.concat([m_rot, tf.expand_dims(translation, 1)], 1)

        # flatten
        affine = tf.reshape(matrix, [self.ndims * (self.ndims + 1)])
        return affine
