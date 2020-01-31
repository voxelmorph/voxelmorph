import neuron as ne
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer


class ResizeTransform(Layer):
    """ 
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super(ResizeTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('ResizeTransform must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        super(ResizeTransform, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            trf = inputs[0]
        else:
            trf = inputs

        # multiply first to save memory (multiply Tensor in smaller space)
        trf = trf * self.zoom_factor
        return ne.utils.resize(trf, self.zoom_factor, self.interp_method)

    def compute_output_shape(self, input_shape):
        return input_shape


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
        self.shape=shape
        self.initializer = initializer
        self.biasmult = mult
        print('LocalParamWithInput: Consider using neuron.layers.LocalParam()')
        super(LocalParamWithInput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalParamWithInput, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        b = 0 * K.batch_flatten(x)[:, 0:1] + 1
        params = K.expand_dims(K.flatten(self.kernel * self.biasmult), 0)
        z = K.reshape(K.dot(b, params), [-1, *self.shape])
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape)
