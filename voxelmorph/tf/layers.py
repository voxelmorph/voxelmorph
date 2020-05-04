import numpy as np
import neuron as ne
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from .utils import is_affine, extract_affine_ndims, affine_shift_to_identity, affine_identity_to_shift


# make the following neuron layers directly available from vxm
SpatialTransformer = ne.layers.SpatialTransformer
LocalParam = ne.layers.LocalParam


class Rescale(Layer):
    """ 
    Rescales a layer by some factor.
    """

    def __init__(self, scale_factor, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return x * self.scale_factor

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleTransform(Layer):
    """ 
    Rescales a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)

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
                return Rescale(self.zoom_factor, name=self.name + '_rescale')(trf)
            else:
                # multiply first to save memory (multiply in smaller space)
                trf = Rescale(self.zoom_factor, name=self.name + '_rescale')(trf)
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
        return ne.utils.compose(inputs[0], inputs[1])

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


class LocalParamWithInput(Layer):
    """ 
    Update 9/29/2019 - TODO: should try ne.layers.LocalParam() again after update.

    The neuron.layers.LocalParam has an issue where _keras_shape gets lost upon calling get_output :(

    tried using call() but this requires an input (or i don't know how to fix it)
    the fix was that after the return, for every time that tensor would be used i would need to do something like
    new_vec._keras_shape = old_vec._keras_shape

    which messed up the code. Instead, we'll do this quick version where we need an input, but we'll ignore it.

    this doesn't have the _keras_shape issue since we built on the input and use call()
    """

    def __init__(self, shape, initializer='RandomNormal', mult=1.0, **kwargs):
        self.shape = shape
        self.initializer = initializer
        self.biasmult = mult
        print('LocalParamWithInput: Consider using neuron.layers.LocalParam()')
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        b = 0 * K.batch_flatten(x)[:, 0:1] + 1
        params = K.expand_dims(K.flatten(self.kernel * self.biasmult), 0)
        z = K.reshape(K.dot(b, params), [-1, *self.shape])
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape)


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
            raise ValueError('Expected affine matrix of shape %s but got %s' % (str((self.ndims, self.ndims + 1)), str(shape)))

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
        return ne.utils.affine_to_shift(trf, self.volshape, shift_center=True)

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
