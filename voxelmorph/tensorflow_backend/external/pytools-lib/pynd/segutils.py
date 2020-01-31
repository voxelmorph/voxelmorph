'''
nd segmentation (label map) utilities

Contact: adalca@csail.mit.edu
'''

import numpy as np
from . import ndutils as nd

def seg2contour(seg, exclude_zero=True, contour_type='inner', thickness=1):
    '''
    transform nd segmentation (label maps) to contour maps

    Parameters
    ----------
    seg : nd array
        volume of labels/segmentations
    exclude_zero : optional logical
        whether to exclude the zero label.
        default True
    contour_type : string
        where to draw contour voxels relative to label 'inner','outer', or 'both'

    Output
    ------
    con : nd array
        nd array (volume) of contour maps

    See Also
    --------
    seg_overlap
    '''

    # extract unique labels
    labels = np.unique(seg)
    if exclude_zero:
        labels = np.delete(labels, np.where(labels == 0))

    # get the contour of each label
    contour_map = seg * 0
    for lab in labels:

        # extract binary label map for this label
        label_map = seg == lab

        # extract contour map for this label
        thickness = thickness + 0.01
        label_contour_map = nd.bw2contour(label_map, type=contour_type, thr=thickness)

        # assign contour to this label
        contour_map[label_contour_map] = lab

    return contour_map



def seg_overlap(vol, seg, do_contour=True, do_rgb=True, cmap=None, thickness=1.0):
    '''
    overlap a nd volume and nd segmentation (label map)

    do_contour should be None, boolean, or contour_type from seg2contour

    not well tested yet.
    '''

    # compute contours for each label if necessary
    if do_contour is not None and do_contour is not False:
        if not isinstance(do_contour, str):
            do_contour = 'inner'
        seg = seg2contour(seg, contour_type=do_contour, thickness=thickness)

    # compute a rgb-contour map
    if do_rgb:
        if cmap is None:
            nb_labels = np.max(seg).astype(int) + 1
            colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
            colors[0, :] = [0, 0, 0]
        else:
            colors = cmap[:, 0:3]

        olap = colors[seg.flat, :]
        sf = seg.flat == 0
        for d in range(3):
            olap[sf, d] = vol.flat[sf]
        olap = np.reshape(olap, vol.shape + (3, ))

    else:
        olap = seg
        olap[seg == 0] = vol[seg == 0]

    return olap


def seg_overlay(vol, seg, do_rgb=True, seg_wt=0.5, cmap=None):
    '''
    overlap a nd volume and nd segmentation (label map)

    not well tested yet.
    '''

    # compute contours for each label if necessary

    # compute a rgb-contour map
    if do_rgb:
        if cmap is None:
            nb_labels = np.max(seg) + 1
            colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
            colors[0, :] = [0, 0, 0]
        else:
            colors = cmap[:, 0:3]

        seg_flat = colors[seg.flat, :]
        seg_rgb = np.reshape(seg_flat, vol.shape + (3, ))

        # get the overlap image
        olap = seg_rgb * seg_wt + np.expand_dims(vol, -1) * (1-seg_wt)

    else:
        olap = seg * seg_wt + vol * (1-seg_wt)

    return olap

