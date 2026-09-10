"""
Neural network building blocks for VoxelMorph.
"""

# Core library imports
from typing import Union, Optional

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import neurite as ne

# Local imports
import voxelmorph.nn.functional as vxf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial transformation according to a deformation field.

    Wrapper around voxelmorph.nn.functional.spatial_transform that maintains the
    nn.Module interface for composability in neural network architectures.

    References
    ----------
    If you find this helpful, please cite the following paper:

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.
    """

    def __init__(
        self,
        interpolation_mode: str = "linear",
        align_corners: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize `SpatialTransformer`.

        Parameters
        ----------
        interpolation_mode : str, default='linear'
            Algorithm used for interpolating the warped image. Options are: {'linear', 'nearest'}
        align_corners : bool, default=True
            Map the corner points of the moving image to the corner points of the warped image.
        device : str or torch.device or None, optional
            Deprecated. No longer used. Kept for backward compatibility.
        """
        super().__init__()

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

    def forward(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of `SpatialTransformer`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Tensor to be spatially transformed by `deformation_field`.
            Shape: (B, C, *spatial_dims).
        deformation_field : torch.Tensor
            Field causing the spatial transformation of `moving_image`.
            Shape: (B, ndim, *spatial_dims).

        Returns
        -------
        torch.Tensor
            Warped `moving_image` according to the `deformation_field`.
            Output shape matches `moving_image` shape: (B, C, *spatial_dims).

        Notes
        -----
        - Expects deformation_field in channels-first format: (B, ndim, *spatial_dims)
        - Uses voxelmorph.nn.functional.spatial_transform for the canonical
          (B, C, *spatial) tensor format.
        """
        assert moving_image.dim() >= 4, (
            f"moving_image must have >=4 dims (B,C,*spatial), got {moving_image.shape}"
        )
        assert deformation_field.dim() == moving_image.dim(), (
            f"dim mismatch: moving={moving_image.dim()}, field={deformation_field.dim()}"
        )

        # Allocate or reallocate meshgrid if spatial shape changed
        spatial_shape = moving_image.shape[2:]
        if not hasattr(self, 'meshgrid') or self.meshgrid.shape[1:] != spatial_shape:
            self.meshgrid = ne.volshape_to_ndgrid(
                size=spatial_shape,
                device=moving_image.device,
                dtype=moving_image.dtype,
                stack=True
            )

        return vxf.spatial_transform(
            image=moving_image,
            trf=deformation_field,
            method=self.interpolation_mode,
            isdisp=True,
            meshgrid=self.meshgrid,
            align_corners=self.align_corners,
            padding_mode='zeros'
        )


class IntegrateVelocityField(nn.Module):
    """
    Integrates a velocity field over multiple steps using the scaling and squaring method.

    This module ensures that transformations caused by a velocity field is diffeomorphic by
    compounding small, intermediate transformations (by recursive scaling and squaring). This
    ensures the resultant is both smooth and invertable.

    Attributes
    ----------
    steps : int
        The number of squaring steps used for integration.
    scale : float
        Scaling factor for the initial velocity field, determined as `1 / (2^steps)`.
    transformer : nn.Module
        A spatial transformer module used to iteratively warp the vector field.

    Examples
    -------
    ### Integrate a 2D velocity field over multiple steps:
    >>> integrator = IntegrateVelocityField(steps=256)
    >>> velocity_field = torch.randn(1, 2, 128, 128)  # (B, C, H, W)
    >>> disp = integrator(velocity_field)
    >>> disp.shape
    torch.Size([1, 2, 128, 128])

    ### Perform integration on a 3D velocity field with a single scaling step:
    >>> integrator = IntegrateVelocityField(steps=1)
    >>> velocity_field = torch.randn(1, 3, 64, 64, 64)  # (B, C, D, H, W)
    >>> disp = integrator(velocity_field)
    >>> disp.shape
    torch.Size([1, 3, 64, 64, 64])
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        steps: int = 1,
        interpolation_mode: str = "linear",
        align_corners: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize `IntegrateVelocityField`

        Parameters
        ----------
        shape : tuple or None, optional
            Deprecated. No longer used. Kept for backward compatibility.
        steps : int, default=1
            Number of integration steps. A higher value leads to a more smooth and accurate
            integration at the cost of higher/longer computation.
        interpolation_mode : str, default='linear'
            Algorithm used for interpolating the warped image. Options are {'linear', 'nearest'}
        align_corners : bool, default=True
            Map the corner points of the moving image to the corner points of the warped image.
        device : str or None, optional
            Deprecated. No longer used. Kept for backward compatibility.
        """

        super().__init__()

        assert steps >= 0, f"steps must be >= 0, got {steps}"

        self.steps = steps
        self.scale = 1.0 / (2 ** self.steps)  # Initial downscaling factor

        # Make the transformer which will perform the warping operation
        self.transformer = SpatialTransformer(
            interpolation_mode=interpolation_mode,
            align_corners=align_corners
        )

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """
        Integrates the input velocity field using scaling and squaring.

        Parameters
        ----------
        vector_field : torch.Tensor
            A velocity field of shape (B, C, *spatial_dims), where B is batch size,
            C is the number of vector components (typically spatial dimensions),
            and `spatial_dims` represent the grid dimensions.

        Returns
        -------
        torch.Tensor
            The integrated displacement field with the same shape as the input.
        """

        return vxf.integrate_vec(
            velocity_field,
            steps=self.steps,
            scale=self.scale,
            transformer=self.transformer,
        )


class ResizeDisplacementField(nn.Module):
    """
    Resize and rescale a displacement field.

    Resizd a displacement field both spatially (via interpolation) and in magnitude (via scaling).

    Examples
    -------
    ### Resize a 2D displacement field
    >>> resize_field = ResizeDisplacementField(scale_factor=2.0, interpolation_mode="bilinear")
    >>> disp = torch.rand(1, 2, 16, 16)  # Example displacement field in 2d
    >>> resized_disp = resize_field(disp)
    >>> print(resized_disp.shape)  # Should be larger if scale_factor > 1
    torch.Size([1, 2, 32, 32])
    """

    def __init__(
        self,
        scale_factor: Optional[Union[float, int]] = 1.0,
        interpolation_mode: str = "linear",
        align_corners: bool = True,
    ):
        """
        Instantiate the `ResizeDisplacementField` module.

        Parameters
        ----------
        scale_factor : Optional[Union[float, int]], optional
            Factor by which to stretch or shrink the spatial dimensions of the displacement field.
            Values of `scale_factor` > 1 stretch/expand the field, and values < 1 shrink it. By
            default 1.0.
        interpolation_mode : str
            Algorithm used for interpolating the warped image. Options are {'linear', 'nearest',
            'bicubic'}
        align_corners : bool
            Map the corner points of the moving image to the corner points of the warped image.
        """
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Resize and rescale the displacement field.

        Parameters
        ----------
        disp : torch.Tensor
            Vector field of shape (B, C, *spatial) representing a displacement field, where C
            represents each spatial component of the vector field.

        Returns
        -------
        torch.Tensor
            Resized displacement field.
        """
        # Infer PyTorch interpolation mode from spatial dimensions
        mode = self.interpolation_mode
        if mode == 'linear':
            num_spatial = disp.ndim - 2  # Subtract batch and channel dims
            mode = ne.utils.infer_linear_interpolation_mode(num_spatial)

        # Use the scale factor to resize the displacement field
        resized_disp = nnf.interpolate(
            disp * self.scale_factor,  # Scale the magnitudes of the displacement field
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=self.align_corners,
        )

        return resized_disp
