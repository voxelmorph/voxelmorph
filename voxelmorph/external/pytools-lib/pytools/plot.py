"""
function to help in plotting
"""

import numpy as np
import six
import matplotlib
import matplotlib.pylab as plt


def jitter(n=256, colmap="hsv", nargout=1):
    """
    jitter colormap of size [n x 3]. The jitter colormap will (likely) have distinct colors, with
    neighburing colors being quite different

    Parameters:
        n (optional): the size of the colormap. default:256
        colmap: the colormap to scramble. Either a string passable to plt.get_cmap,
            or a n-by-3 or n-by-4 array

    Algorithm: given a (preferably smooth) colormap as a starting point (default "hsv"), jitter
    reorders the colors by skipping roughly a quarter of the colors. So given jitter(9, "hsv"),
    jitter would take color numbers, in order, 1, 3, 5, 7, 9, 2, 4, 6, 8.

    Contact: adalca@csail.mit.edu
    """

    # get a 1:n vector
    idx = range(n)

    # roughly compute the quarter mark. in hsv, a quarter is enough to see a significant col change
    m = np.maximum(np.round(0.25 * n), 1).astype(int)

    # compute a new order, by reshaping this index array as a [m x ?] matrix, then vectorizing in
    # the opposite direction

    # pad with -1 to make it transformable to a square
    nb_elems = np.ceil(n / m) * m
    idx = np.pad(idx, [0, (nb_elems - n).astype(int)], 'constant', constant_values=-1)

    # permute elements by resizing to a matrix, transposing, and re-flatteneing
    idxnew = np.array(np.reshape(idx, [m, (nb_elems // m).astype(int)]).transpose().flatten())

    # throw away the extra elements
    idxnew = idxnew[np.where(idxnew >= 0)]
    assert len(idxnew) == n, "jitter: something went wrong with some inner logic :("

    # get colormap and scramble it
    if isinstance(colmap, six.string_types):
        cmap = plt.get_cmap(colmap, nb_elems)
        scrambled_cmap = cmap(idxnew)
    else:
        # assumes colmap is a nx3 or nx4
        assert colmap.shape[0] == n
        assert colmap.shape[1] == 3 or colmap.shape[1] == 4
        scrambled_cmap = colmap[idxnew, :]

    new_cmap = matplotlib.colors.ListedColormap(scrambled_cmap)
    if nargout == 1:
        return new_cmap
    else:
        assert nargout == 2
        return (new_cmap, scrambled_cmap)
