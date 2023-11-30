"""
Code to convert between flow field conventions for SpatialTransformer and the
Baker et al. colorwheel. code

Note that when using a transformer, the sign of flow is the opposite of the 'offset'
direction. That is, negative values in the x direction means the transformer samples
from coordinates to the left of the original coordinates, which makes the image shift
to the right (i.e. it results in a positive x-offset). When using the Baker et al.
colorwheel, negative x values indicate a shift to the left.
I refer to these below as 'sampling' or 'transformer' flow (for the spatialTransformer)
 vs 'offset' or 'directional' flow (used for the colorwheel).

Also note that the Voxelmorph SpatialTransformer module uses 'ij' indexing (first flow 
dimension = vertical flow) whereas the Baker et al. colorwheel uses xy indexing
(first flow dimension = horizontal flow)

Another difference is that the flow in the PyTorch transformer assumes that the
'flow dimension' (the axis along which the x, y, (and z) flow channels are stacked)
is the first dimension. The colorwheel code (and the flow for the transformer
implemented in Tensorflow)  assumes that the flow dimension is the last
dimension.

The two functions below convert between these two formats:
    tform2dir_flow : Converts from Transformer (sampling) flow to Directional flow
    dir2tform_flow : Converts from Directional to Transformer (sampling) flow

Note that in both systems, it is more straightforward to use 'matrix' coordinates
(i.e. (0, 0) is at the top left corner, vs cartesian coordinates origin (i.e. (0, 0) at
bottom left corner). So when plotting, it is a good idea to make sure that the yaxis
 is inverted (to show matrix coordinates).
(This is the default when using matplotlib.pyplot.imshow(...))

"""
import numpy as np


def tform2dir_flow(tform_flow: np.ndarray):
    """
    Transformer flow (for SpatialTransformer) has:
        channels first (e.g. nDim, H, W, [D]) [for flow in PyTorch]
        ij-indexing (first dimension: vertical flow. second dim: horizontal flow)
        sampling flow (negative: shift forward. positive: shift backward)

    Directional flow (for flow_to_color) has:
        channels last (e.g. H, W, [D,] nDim)
        xy indexing (first dimension: horizontal flow. second dim: vertical flow)
        directional flow (positive: shift forward. negative: shift backward)
    """
    dir_flow = tform_flow.copy()

    # 1. Reorder axes to channel-last (if necessary: if the axes are channel-last
    # already, we can skip this step)
    shape = tform_flow.shape
    ndim = tform_flow.ndim
    flow_size = ndim - 1
    # The flow dimension should either be the first or the last dimension.
    assert (shape[0] == flow_size) != (shape[-1] == flow_size), \
        (f"Could not determine flow dimension. Expected the flow dimension "
         f"(of size {flow_size}) to be the first or last dimension, but "
         f"received input of shape {shape}")
    if shape[0] == flow_size:
        new_order = list(range(1, ndim)) + [0]
        dir_flow = np.transpose(tform_flow, new_order)

    # 2. Flip flow axes order (ij to xy)
    dir_flow = np.flip(dir_flow, axis=-1).copy()

    # 3. Take negative (sampling flow to directional flow)
    dir_flow = -dir_flow
    return dir_flow


def dir2tform_flow(dir_flow: np.ndarray):
    """
    Transformer flow (for SpatialTransformer) has:
        channels first (e.g. nDim, H, W) [for flow in PyTorch]
        ij-indexing (first dimension: vertical flow. second dim: horizontal flow)
        sampling flow (negative: shift forward. positive: shift backward)

    Directional flow (for flow_to_color) has:
        channels last (e.g. H, W, nDim)
        xy indexing (first dimension: horizontal flow. second dim: vertical flow)
        directional flow (positive: shift forward. negative: shift backward)
    """
    tform_flow = dir_flow.copy()

    # 1. Reorder axes to channel-first (if necessary: if the axes are channel-first
    # already, we can skip this step)
    shape = tform_flow.shape
    ndim = tform_flow.ndim
    flow_size = ndim - 1
    # The flow dimension should either be the first or the last dimension.
    assert (shape[0] == flow_size) != (shape[-1] == flow_size), \
        (f"Could not determine flow dimension. Expected the flow dimension "
         f"(of size {flow_size}) to be the first or last dimension, but "
         f"received input of shape {shape}")

    if shape[-1] == flow_size:
        new_order = [ndim - 1] + list(range(ndim - 1))  # flow channel: last to first
        tform_flow = np.transpose(tform_flow, new_order)

    # 2. Flip flow axes order (xy to ij)
    tform_flow = np.flip(tform_flow, axis=0).copy()

    # 3. Take negative (directional flow to sampling flow)
    tform_flow = -tform_flow
    return tform_flow
