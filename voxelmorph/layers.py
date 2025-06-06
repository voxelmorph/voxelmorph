"""
tensorflow/keras layers for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import warnings
import numpy as np
import neurite as ne

# tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# local utils
from . import utils


class SpatialTransformer(Layer):
    """
    N-dimensional (ND) spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network.

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 shape=None,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms. Assumes the input and output spaces are identical.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used. Incompatible with `shift_center=True`.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if 'indexing' in kwargs:
            del kwargs['indexing']
            warnings.warn('The `indexing` argument to SpatialTransformer no longer exists. If you '
                          'loaded a model, save it again to be able to load it in the future.')

        self.interp_method = interp_method
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
            'shape': self.shape,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]

        # make sure transform has reasonable shape (is_affine_shape throws error if not)
        if not utils.is_affine_shape(input_shape[1][1:]):
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1] or [B, N+1, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        return utils.transform(inputs[0],
                               inputs[1],
                               interp_method=self.interp_method,
                               fill_value=self.fill_value,
                               shift_center=self.shift_center,
                               shape=self.shape)


class VecInt(Layer):
    """
    Vector integration layer

    Enables vector integration via several methods (ode or quadrature for
    time-dependent vector fields and scaling-and-squaring for stationary fields)

    If you find this function useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self,
                 method='ss',
                 int_steps=7,
                 out_time_pt=1,
                 ode_args=None,
                 odeint_fn=None,
                 **kwargs):
        """
        Parameters:
            method: Must be any of the methods in neuron.utils.integrate_vec.
            int_steps: Number of integration steps.
            out_time_pt: Time point at which to output if using odeint integration.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if 'indexing' in kwargs:
            del kwargs['indexing']
            warnings.warn('The `indexing` argument to VecInt no longer exists. If you loaded a '
                          'model, save it again to be able to load it in the future.')

        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
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

        # necessary for multi-gpu models
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], '_keras_shape'):
            loc_shift._keras_shape = inputs[0]._keras_shape

        if len(inputs) > 1:
            assert self.out_time_pt is None, \
                'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out = tf.map_fn(self._single_int,
                        [loc_shift] + inputs[1:],
                        fn_output_signature=loc_shift.dtype)
        if hasattr(inputs[0], '_keras_shape'):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return utils.integrate_vec(vel, method=self.method,
                                   nb_steps=self.int_steps,
                                   ode_args=self.ode_args,
                                   out_time_pt=out_time_pt,
                                   odeint_fn=self.odeint_fn)


# full wording.
VecIntegration = VecInt


class RescaleTransform(Layer):
    """ 
    Rescale transform layer

    Rescales a dense or affine transform. If dense, this involves resizing and
    rescaling the vector field.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        """
        Parameters:
            zoom_factor: Scaling factor.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'zoom_factor': self.zoom_factor,
            'interp_method': self.interp_method,
        })
        return config

    def build(self, input_shape):
        # check if transform is affine
        self.is_affine = utils.is_affine_shape(input_shape[1:])
        self.ndims = input_shape[-1] - 1 if self.is_affine else input_shape[-1]

    def compute_output_shape(self, input_shape):
        if self.is_affine:
            return (input_shape[0], self.ndims, self.ndims + 1)
        else:
            shape = [int(d * self.zoom_factor) for d in input_shape[1:-1]]
            return (input_shape[0], *shape, self.ndims)

    def call(self, transform):
        """
        Parameters
            transform: Transform to rescale. Either a dense warp of shape [B, D1, ..., DN, N]
            or an affine matrix of shape [B, N, N+1].
        """
        if self.is_affine:
            return utils.rescale_affine(transform, self.zoom_factor)
        else:
            return utils.rescale_dense_transform(transform, self.zoom_factor,
                                                 interp_method=self.interp_method)


class ComposeTransform(Layer):
    """ 
    Composes a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = ComposeTransform()([A, B, C])
    """

    def __init__(self, interp_method='linear', shift_center=True, shape=None, **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            shift_center: Shift grid to image center when converting matrices to dense transforms.
            shape: ND output shape used for converting matrices to dense transforms. Includes only
                the N spatial dimensions. Only used once, if the rightmost transform is a matrix.
                If None or if the rightmost transform is a warp, the shape of the rightmost warp
                will be used. Incompatible with `shift_center=True`.
        """
        self.interp_method = interp_method
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'shift_center': self.shift_center,
            'shape': self.shape,
        })
        return config

    def build(self, input_shape, **kwargs):

        # sanity check on the inputs
        if not isinstance(input_shape, (list, tuple)):
            raise Exception('ComposeTransform must be called for a list of transforms.')

    def call(self, transforms):
        """
        Parameters:
            transforms: List of affine or dense transforms to compose.
        """
        if len(transforms) == 1:
            return transforms[0]

        compose = lambda trf: utils.compose(trf,
                                            interp_method=self.interp_method,
                                            shift_center=self.shift_center,
                                            shape=self.shape)
        return tf.map_fn(compose, transforms, fn_output_signature=transforms[0].dtype)


class AddIdentity(Layer):
    """
    Adds the identity matrix to the input. This is useful when predicting
    affine parameters directly.
    """

    def build(self, input_shape):
        shape = input_shape[1:]

        if len(shape) == 1:
            # let's support 1D flattened affines here, since it's
            # likely the input is coming right from a dense layer
            length = shape[0]
            if length == 6:
                self.ndims = 2
            elif length == 12:
                self.ndims = 3
            else:
                raise ValueError('Flat affine must be of length 6 (2D) or 12 (3D), got {length}.')
            self.nrows = self.ndims
        elif len(shape) == 2:
            # or it could be a 2D matrix
            utils.validate_affine_shape(input_shape)
            self.ndims = shape[1] - 1
            self.nrows = shape[0]
        else:
            raise ValueError('Input to AddIdentity must be a flat 1D array or 2D matrix, '
                             f'got shape {input_shape}.')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nrows, self.ndims + 1)

    def call(self, transform):
        """
        Parameters
            transform: Affine transform of shape [B, N, N+1] or [B, N+1, N+1] or [B, N*(N+1)].
        """
        transform = tf.reshape(transform, (-1, self.nrows, self.ndims + 1))
        return utils.affine_add_identity(transform)


class InvertAffine(Layer):
    """
    Inverts an affine transform.
    """

    def build(self, input_shape):
        utils.validate_affine_shape(input_shape)
        self.nrows = input_shape[-2]
        self.ndims = input_shape[-1] - 1

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nrows, self.ndims + 1)

    def call(self, matrix):
        """
        Parameters
            matrix: Affine matrix of shape [B, N, N+1] or [B, N+1, N+1] to invert.
        """
        return utils.invert_affine(matrix)


class ParamsToAffineMatrix(Layer):
    """
    Constructs an affine transformation matrix from translation, rotation, scaling and shearing
    parameters in 2D or 3D.

    If you find this layer useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self, ndims=3, deg=True, shift_scale=False, last_row=False, **kwargs):
        """
        Parameters:
            ndims: Dimensionality of transform matrices. Must be 2 or 3.
            deg: Whether the input rotations are specified in degrees.
            shift_scale: Add 1 to any specified scaling parameters. This may be desirable
                when the parameters are estimated by a network.
            last_row: Whether to return a full matrix, including the last row.
        """
        self.ndims = ndims
        self.deg = deg
        self.shift_scale = shift_scale
        self.last_row = last_row
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ndims': self.ndims,
            'deg': self.deg,
            'shift_scale': self.shift_scale,
            'last_row': self.last_row,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims + int(self.last_row), self.ndims + 1)

    def call(self, params):
        """
        Parameters:
            params: Parameters as a vector which corresponds to translations, rotations, scaling
                    and shear. The size of the last axis must not exceed (N, N+1), for N
                    dimensions. If the size is less than that, the missing parameters will be
                    set to the identity.
        """
        return utils.params_to_affine_matrix(par=params,
                                             deg=self.deg,
                                             shift_scale=self.shift_scale,
                                             ndims=self.ndims,
                                             last_row=self.last_row)


class AffineToDenseShift(Layer):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
        """
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'shift_center': self.shift_center,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)

    def build(self, input_shape):
        utils.validate_affine_shape(input_shape)

    def call(self, mat):
        """
        Parameters:
            mat: Affine matrices of shape (B, N, N+1).
        """
        return utils.affine_to_dense_shift(mat, self.shape, shift_center=self.shift_center)


class DrawAffineParams(Layer):
    """
    Draw translation, rotation, scaling and shearing parameters defining an affine transform in
    N-dimensional space, where N is 2 or 3. Choose parameters wisely: there is no check for
    negative or zero scaling! The batch dimension will be inferred from the input tensor.

    Returns:
        A tuple of tensors with shapes (..., N), (..., M), (..., N), and (..., M) defining
        translation, rotation, scaling, and shear, respectively, where M is 3 in 3D and 1 in 2D.
        With `concat=True`, the layer will concatenate the output along the last dimension.

    See also:
        ParamsToAffineMatrix

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 shift=None,
                 rot=None,
                 scale=None,
                 shear=None,
                 normal_shift=False,
                 normal_rot=False,
                 normal_scale=False,
                 normal_shear=False,
                 shift_scale=False,
                 ndims=3,
                 concat=True,
                 out_type=tf.float32,
                 seeds={},
                 **kwargs):
        """
        Parameters:
            shift: Translation sampling range x around identity. Values will be sampled uniformly
                from [-x, x]. When sampling from a normal distribution, x is the standard
                deviation (SD). The same x will be used for each dimension, unless an iterable of
                length N is passed, specifying a value separately for each axis. None means 0.
            rot: Rotation sampling range (see `shift`). Accepts only one value in 2D.
            scale: Scaling sampling range x. Parameters will be sampled around identity as for
                `shift`, unless `shift_scale` is set. When sampling normally, scaling parameters
                will be truncated beyond two standard deviations.
            shear: Shear sampling range (see `shift`). Accepts only one value in 2D.
            normal_shift: Sample translations normally rather than uniformly.
            normal_rot: See `normal_shift`.
            normal_scale: Draw scaling parameters normally, truncating beyond 2 SDs.
            normal_shear: See `normal_shift`.
            shift_scale: Add 1 to any drawn scaling parameter When sampling uniformly, this will
                result in scaling parameters falling in [1 - x, 1 + x] instead of [-x, x].
            ndims: Number of dimensions. Must be 2 or 3.
            normal: Sample parameters normally instead of uniformly.
            concat: Concatenate the output along the last axis to return a single tensor.
            out_type: Floating-point output data type.
            seeds: Dictionary of integers for reproducible randomization. Keywords must be in
                ('shift', 'rot', 'scale', 'shear').
        """
        self.shift = shift
        self.rot = rot
        self.scale = scale
        self.shear = shear
        self.normal_shift = normal_shift
        self.normal_rot = normal_rot
        self.normal_scale = normal_scale
        self.normal_shear = normal_shear
        self.shift_scale = shift_scale
        self.ndims = ndims
        self.concat = concat
        self.out_type = tf.dtypes.as_dtype(out_type)
        self.seeds = seeds
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shift': self.shift,
            'rot': self.rot,
            'scale': self.scale,
            'shear': self.shear,
            'normal_shift': self.normal_shift,
            'normal_rot': self.normal_rot,
            'normal_scale': self.normal_scale,
            'normal_shear': self.normal_shear,
            'shift_scale': self.shift_scale,
            'ndims': self.ndims,
            'concat': self.concat,
            'out_type': self.out_type,
            'seeds': self.seeds,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input tensor that we derive the batch dimension from.
        """
        return utils.draw_affine_params(shift=self.shift,
                                        rot=self.rot,
                                        scale=self.scale,
                                        shear=self.shear,
                                        normal_shift=self.normal_shift,
                                        normal_rot=self.normal_rot,
                                        normal_scale=self.normal_scale,
                                        normal_shear=self.normal_shear,
                                        shift_scale=self.shift_scale,
                                        ndims=self.ndims,
                                        batch_shape=tf.shape(x)[:1],
                                        concat=self.concat,
                                        dtype=self.out_type,
                                        seeds=self.seeds)


class DownUpSample(Layer):
    """
    Symmetrically downsample a tensor by a factor f (stride) using
    nearest-neighbor interpolation and upsample again, to reduce its
    resolution. Both f and the downsampling axes can be randomized. The
    layer does not bother with anti-aliasing, as it is intended for
    augmentation after a random blurring step.

    Notes:
        The layer differs from ne.utils.Subsample in that it downsamples along
        several axes, always restores the shape of the input tensor, and moves
        the image content less.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 stride_min=1,
                 stride_max=8,
                 axes=None,
                 prob=1,
                 interp_method='linear',
                 seed=None,
                 **kwargs):
        """
        Parameters:
            stride_min: Minimum downsampling factor.
            stride_max: Maximum downsampling factor.
            axes: Spatial axes to draw the downsampling axis from. None means all.
            prob: Downsampling probability. A value of 1 means always, 0 never.
            interp_method: Upsampling method. Choose 'linear' or 'nearest'.
            seed: Integer for reproducible randomization.
        """
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.axes = axes
        self.prob = prob
        self.interp_method = interp_method
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'stride_min': self.stride_min,
            'stride_max': self.stride_max,
            'axes': self.axes,
            'prob': self.prob,
            'interp_method': self.interp_method,
            'seed': self.seed,
        })
        return config

    def build(self, in_shape):
        ndims = len(in_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'

        allowed = range(1, ndims + 1)
        self.axes = ne.py.utils.normalize_axes(self.axes, in_shape, allowed, none_means_all=True)

        self.rand = tf.random.Generator.from_non_deterministic_state()
        if self.seed is not None:
            self.rand.reset_from_seed(self.seed)

        super().build(in_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor to downsample.
        """
        if self.prob == 0 or self.stride_max == 1:
            return x

        func = lambda f: utils.down_up_sample(f,
                                              stride_min=self.stride_min,
                                              stride_max=self.stride_max,
                                              axes=tuple(ax - 1 for ax in self.axes),
                                              prob=self.prob,
                                              interp_method=self.interp_method,
                                              rand=self.rand)

        return tf.map_fn(func, x)
