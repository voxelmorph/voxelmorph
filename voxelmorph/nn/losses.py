"""
Loss functions for image registration.
"""

# Core library imports
import math

# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        raise NotImplementedError(
            'voxelmorph.nn.losses.NCC is deprecated. Use neurite.nn.modules.NCC instead.'
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            'voxelmorph.nn.losses.NCC is deprecated. Use neurite.nn.modules.NCC instead.'
        )


class MSE:
    """
    Deprecated. Use neurite.nn.modules.MSE instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "voxelmorph.nn.losses.MSE is deprecated. Use neurite.nn.modules.MSE instead."
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.MSE is deprecated. Use neurite.nn.modules.MSE instead."
        )


class Dice:
    """
    Deprecated. Use neurite.nn.modules.Dice instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Dice is deprecated. Use neurite.nn.modules.Dice instead."
        )

    def loss(self, y_true, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Dice is deprecated. Use neurite.nn.modules.Dice instead."
        )


class Grad:
    """
    Deprecated. Use `neurite.nn.modules.SpatialGradient` instead.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. "
            "Use neurite.nn.modules.SpatialGradient instead."
        )

    def _diffs(self, y):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. "
            "Use neurite.nn.modules.SpatialGradient instead."
        )

    def loss(self, y_pred):
        raise NotImplementedError(
            "voxelmorph.nn.losses.Grad is deprecated. "
            "Use neurite.nn.modules.SpatialGradient instead."
        )
