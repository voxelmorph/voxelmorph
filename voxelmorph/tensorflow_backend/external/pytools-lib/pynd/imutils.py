''' image utilities '''

import numpy as np

def gray2color(gray, color):
    ''' 
    transform a gray image (2d array) to a color image given the color (1x3 vector) 
    untested
    '''

    return np.stack((gray * c for c in color), -1)


def rgb2gray(rgb, mixing=[0.2989, 0.5870, 0.1140], keepdims = False):
    ''' 
    transform a rgb image (i.e. array with last dimension of 3) to grayscale
    (which reduces the last dimension)
    '''
    gray = np.dot(rgb[...,:3], mixing)
    if keepdims:
        gray = gray[..., np.newaxis]
    return gray
