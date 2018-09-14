''' initializations for the neuron project '''

# general imports
import os
import numpy as np
import keras.backend as K


def output_init(shape, name=None, dim_ordering=None):
    ''' initialization for output weights'''
    size = (shape[0], shape[1], shape[2] - shape[3], shape[3])

    # initialize output weights with random and identity
    rpart = np.random.random(size)
#     idpart_ = np.eye(size[3])
    idpart_ = np.ones((size[3], size[3]))
    idpart = np.expand_dims(np.expand_dims(idpart_, 0), 0)
    value = np.concatenate((rpart, idpart), axis=2)
    return K.variable(value, name=name)
