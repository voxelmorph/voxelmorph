''' image utilities '''

from numpy import np

def gray2color(gray, color):
    ''' 
    transform a gray image (2d array) to a color image given the color (1x3 vector) 
    untested
    '''

    return np.concatenate((gray * c for c in color), 2)