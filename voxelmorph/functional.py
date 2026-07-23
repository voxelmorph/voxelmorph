"""
Single tensor operations (no B, C, dimensions assumption)
"""
# Core library imports
from typing import Union, Sequence, Tuple, Literal, Optional

# Third-party imports
import numpy as np
import torch
import neurite as ne
import voxelmorph.nn.functional as vxf

__all__ = [
    'affine_to_disp',
    'angles_to_rotation_matrix',
    'params_to_affine',
    'random_affine',
    'disp_to_trf',
    'trf_to_disp',
    'disp_to_coords',
    'coords_to_disp',
    'spatial_transform',
    'integrate_disp',
    'resize_disp',
    'constant_shift_field',
    'compose',
    'is_affine_shape',
    'make_square_affine',
    'random_field',
    'random_disp',
    'random_transform',
]


def angles_to_rotation_matrix(
    rotation: torch.Tensor,
    degrees: bool = True
) -> torch.Tensor:
    """
    Compute a rotation matrix from the given rotation angles.

    Parameters
    ----------
    rotation : Tensor
        A tensor containing the rotation angles. If `degrees` is True, the angles
        are in degrees, otherwise they are in radians.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.

    Returns
    -------
    Tensor
        The computed `(ndim, ndim)` rotation matrix.
    """
    rotation = torch.as_tensor(rotation, dtype=torch.float64)
    if degrees:
        rotation = torch.deg2rad(rotation)
    rotation = torch.atleast_1d(rotation)
    n_angles = len(rotation)
    assert n_angles in (1, 3), f"expected 1 or 3 rotation angles, got {n_angles}"

    zero = rotation.new_zeros(())
    one = rotation.new_ones(())

    if n_angles == 1:
        c, s = torch.cos(rotation[0]), torch.sin(rotation[0])
        matrix = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ])
    else:
        cx, sx = torch.cos(rotation[0]), torch.sin(rotation[0])
        cy, sy = torch.cos(rotation[1]), torch.sin(rotation[1])
        cz, sz = torch.cos(rotation[2]), torch.sin(rotation[2])

        rx = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, sx]),
            torch.stack([zero, -sx, cx]),
        ])
        ry = torch.stack([
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ])
        rz = torch.stack([
            torch.stack([cz, sz, zero]),
            torch.stack([-sz, cz, zero]),
            torch.stack([zero, zero, one]),
        ])
        matrix = rx @ ry @ rz

    return matrix


def params_to_affine(
    ndim: int,
    translation: Union[torch.Tensor, None] = None,
    rotation: Union[torch.Tensor, None] = None,
    scale: Union[torch.Tensor, None] = None,
    shear: Union[torch.Tensor, None] = None,
    degrees: bool = True,
    device: Union[torch.device, None] = None
) -> torch.Tensor:
    """
    Makes an affine matrix from translation, rotation, scale, and shear transform components.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the affine matrix. Must be 2 or 3.
    translation : Tensor, optional
        The translation vector. Must be a vector of size `ndim`.
    rotation : Tensor, optional
        The rotation angles. Must be a scalar value for 2D affine matrices,
        and a tensor of size 3 for 3D affine matrices.
    scale : Tensor, optional
        The scaling factor. Can be scalar or vector of size `ndim`.
    shear : Tensor, optional
        The shearing factor. Must be a scalar value for 2D affine matrices,
        and a tensor of size 3 for 3D affine matrices.
    degrees : bool, optional
        Whether to interpret the rotation angles as degrees.
    device : torch.device, optional
        The device of the returned matrix.

    Returns
    -------
    Tensor
        The composed affine matrix, as a tensor of shape `(ndim + 1, ndim + 1)`.
    """
    assert ndim in (2, 3), f'affine transform must be 2D or 3D, got ndim {ndim}'

    n_rotation_angles = 3 if ndim == 3 else 1

    # Validate and default translation
    translation = torch.zeros(ndim) if translation is None else torch.as_tensor(translation)
    assert len(translation) == ndim, f'Translation must be of shape ({ndim},)'

    # Validate and default rotation
    rotation = torch.zeros(n_rotation_angles) if rotation is None else torch.as_tensor(rotation)
    rotation = torch.atleast_1d(rotation)
    assert rotation.shape[0] == n_rotation_angles, f'Rotation must be shape ({n_rotation_angles},)'

    # Validate and default scale
    scale = torch.ones(ndim) if scale is None else torch.as_tensor(scale)
    if scale.ndim == 0:
        scale = scale.repeat(ndim)
    assert scale.shape[0] == ndim, f'scale must be of size {ndim}'

    # Validate and default shear
    shear = torch.zeros(n_rotation_angles) if shear is None else torch.as_tensor(shear)
    shear = torch.atleast_1d(shear)
    assert shear.shape[0] == n_rotation_angles, f'shear must be of shape ({n_rotation_angles},)'

    # start from translation
    T = torch.eye(ndim + 1, dtype=torch.float64)
    T[:ndim, -1] = translation

    # rotation matrix
    R = torch.eye(ndim + 1, dtype=torch.float64)
    R[:ndim, :ndim] = angles_to_rotation_matrix(rotation, degrees=degrees)

    # scaling
    Z = torch.diag(torch.cat([scale, torch.ones(1, dtype=torch.float64)]))

    # shear matrix
    S = torch.eye(ndim + 1, dtype=torch.float64)
    S[0][1] = shear[0]
    if ndim == 3:
        S[0][2] = shear[1]
        S[1][2] = shear[2]

    # compose component matrices
    matrix = T @ R @ Z @ S

    return torch.as_tensor(matrix, dtype=torch.float32, device=device)


def random_affine(
    ndim: int,
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 1,
    device: Union[torch.device, None] = None,
    sampling: bool = True
) -> torch.Tensor:
    """
    Generate random affine transformation matrix.

    This function generates random affine parameters (translation, rotation, scaling)
    and composes them into an affine transformation matrix.

    Parameters
    ----------
    ndim : int
        Spatial dimensionality of the transformation (2 or 3).
    max_translation : float, default=0
        Range to sample translation parameters from. Scalar values define the max
        deviation from 0.0 (-max_translation, max_translation).
    max_rotation : float, default=0
        Range to sample rotation parameters from. Scalar values define the max
        deviation from 0.0 (-max_rotation, max_rotation).
    max_scaling : float, default=1
        Max to sample scale parameters from.
        It is converted into a 2-element array defines the (min, max) deviation from 1.0.
    device : torch.device or None, default=None
        Device for the output tensor.
    sampling : bool, default=True
        If True, sample random parameters within the specified ranges.
        If False, use the maximum values directly.

    Returns
    -------
    torch.Tensor
        Affine transformation matrix of shape (ndim+1, ndim+1).

    Examples
    --------
    >>> import voxelmorph as vxm
    >>> # Generate random 3D affine with translation
    >>> affine = vxm.random_affine(ndim=3, max_translation=10)
    >>> affine.shape
    torch.Size([4, 4])

    >>> # Generate 2D affine with rotation and scaling
    >>> affine = vxm.random_affine(
    ...     ndim=2,
    ...     max_rotation=30,
    ...     max_scaling=1.2,
    ...     device=torch.device('cuda')
    ... )
    >>> affine.shape
    torch.Size([3, 3])
    """
    n_rotation_angles = 1 if ndim == 2 else 3

    if not sampling:
        translation = np.array([max_translation] * ndim)
        rotation = np.array([max_rotation] * n_rotation_angles)
        scale = np.array([max_scaling] * ndim)
    else:
        assert max_scaling >= 1, "max_scaling must be >= 1 (scales sampled in [1/max, max])"

        translation = np.random.uniform(-max_translation, max_translation, size=ndim)
        rotation = np.random.uniform(-max_rotation, max_rotation, size=n_rotation_angles)
        scale_direction = np.random.choice([-1, 1], size=ndim)
        scale = np.random.uniform(1, max_scaling, size=ndim) ** scale_direction

    return params_to_affine(
        ndim=ndim,
        translation=translation,
        rotation=rotation,
        scale=scale,
        device=device
    )


def affine_to_disp(
    affine: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None,
    warp_right: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Convert an affine transformation matrix to a displacement field.

    Parameters
    ----------
    affine : Tensor
        Affine transformation matrix of shape (N, N+1) or (N+1, N+1), or batched
        affine of shape (B, N, N+1) or (B, N+1, N+1).
        Expected to be a vox2vox target-to-source transformation.
    meshgrid : Tensor, optional
        Pre-computed meshgrid tensor of shape (N, *spatial_shape), where N is the spatial
        dimensionality. If None, will be computed from `shape` parameter.
    origin_at_center : bool, optional
        If True, place the coordinate system origin at the image center. If False, origin
        is at the top-left corner. Default is True.
    shape : Sequence[int], optional
        Spatial shape (N dimensions) to create meshgrid if `meshgrid` is not provided.
        Required if `meshgrid` is None.
    warp_right : Tensor, optional
        Right-compose the affine with this displacement field of shape (N, *spatial_shape)
        or (B, N, *spatial_shape) for batched.
        Computes affine(x + warp_right(x)) - x. Useful for composing transforms.

    Returns
    -------
    Tensor
        Displacement field of shape (N, *spatial_shape) for single affine, or
        (B, N, *spatial_shape) for batched affine.

    Examples
    --------
    >>> # Basic usage with pre-computed meshgrid
    >>> import neurite as ne
    >>> affine = torch.tensor(
    >>> ... [[1., 0., 5.],
    >>> ... [0., 1., 3.]]
    >>> )
    >>> grid = ne.volshape_to_ndgrid((64, 64), stack=True)
    >>> disp = affine_to_disp(affine, meshgrid=grid)

    >>> # Using shape parameter instead
    >>> disp = affine_to_disp(affine, shape=(64, 64))
    >>> disp.shape
    torch.Size([2, 64, 64])

    >>> # Compose affine with existing displacement field
    >>> warp = torch.randn(2, 64, 64)  # (ndim, H, W)
    >>> composed = affine_to_disp(affine, shape=(64, 64), warp_right=warp)

    >>> # Batched affine matrices
    >>> affines = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (2, 3, 3)
    >>> disp = affine_to_disp(affines, shape=(64, 64))
    >>> disp.shape
    torch.Size([2, 2, 64, 64])
    """
    assert (meshgrid is None) != (shape is None), "Provide exactly one of `meshgrid` or `shape`"

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(
            size=shape, device=affine.device, dtype=affine.dtype, stack=True
        )

    assert isinstance(meshgrid, torch.Tensor)
    ndim = meshgrid.shape[0]
    spatial_shape = meshgrid.shape[1:]
    is_batched = affine.ndim == 3

    assert affine.shape[-1] == ndim + 1, (
        f"affine dim ({affine.shape[-1] - 1}D) != meshgrid dim ({ndim}D)"
    )

    # Center origin if requested
    grid = meshgrid
    if origin_at_center:
        center_offsets = [(s - 1) / 2 for s in spatial_shape]
        center_offsets = torch.tensor(center_offsets, device=meshgrid.device).view(-1, *[1] * ndim)
        grid = meshgrid - center_offsets

    # Flatten grid: (ndim, *spatial) -> (ndim, num_voxels)
    coords = grid.reshape(ndim, -1)

    # Right-compose with displacement field if provided
    if warp_right is not None:
        assert warp_right.shape[-ndim:] == spatial_shape, (
            f"warp_right shape {warp_right.shape[-ndim:]} != meshgrid {spatial_shape}"
        )
        coords = coords + warp_right.reshape(*warp_right.shape[:-ndim], -1)

    # Apply affine: A @ coords + t, then subtract original to get displacement
    transformed = affine[..., :ndim, :ndim] @ coords + affine[..., :ndim, -1:]
    disp_flat = transformed - grid.reshape(ndim, -1)

    # Reshape back to spatial
    output_shape = (affine.shape[0], ndim, *spatial_shape) if is_batched else (ndim, *spatial_shape)
    return disp_flat.reshape(*output_shape)


def disp_to_trf(
    disp: torch.Tensor,
    grid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Convert displacement field to transformation (deformation) field.

    Adds an identity coordinate grid to the displacement field to produce
    absolute sampling coordinates.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (ndim, *spatial) or (B, ndim, *spatial).
        The ndim dimension contains vector components and is not considered spatial.
    grid : torch.Tensor or None, default=None
        Pre-computed identity coordinate grid of shape (ndim, *spatial). If None,
        computed from displacement field shape. Useful to avoid recomputing the
        grid in training loops.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Batch dimensions preceding the ndim dimension. Use (0,) for batched input
        (B, ndim, *spatial). If None, assumes unbatched (ndim, *spatial).

    Returns
    -------
    torch.Tensor
        Transformation field with the same shape as input.

    Examples
    --------
    >>> import torch
    >>> import voxelmorph as vxm
    >>> # 2d displacement field
    >>> disp = torch.zeros(2, 64, 64)
    >>> trf = vxm.disp_to_trf(disp)
    >>> trf.shape
    torch.Size([2, 64, 64])

    >>> # With batch dimension
    >>> disp = torch.zeros(4, 2, 64, 64)
    >>> trf = vxm.disp_to_trf(disp, non_spatial_dims=(0,))
    >>> trf.shape
    torch.Size([4, 2, 64, 64])

    >>> # With pre-computed grid (for performance in training loops)
    >>> import neurite as ne
    >>> grid = ne.volshape_to_ndgrid((64, 64), stack=True)
    >>> trf = vxm.disp_to_trf(disp, grid=grid, non_spatial_dims=(0,))

    See Also
    --------
    trf_to_disp : Inverse operation.
    """
    if grid is None:
        num_non_spatial, _ = ne.functional.parse_non_spatial_dims(non_spatial_dims, disp.dim())
        spatial_shape = disp.shape[num_non_spatial + 1:]
        grid = ne.volshape_to_ndgrid(
            size=spatial_shape, device=disp.device, dtype=disp.dtype, stack=True
        )
    return disp + grid


def trf_to_disp(
    trf: torch.Tensor,
    grid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Convert transformation (deformation) field to displacement field.

    Subtracts an identity coordinate grid from the transformation field.

    Parameters
    ----------
    trf : torch.Tensor
        Transformation field with shape (ndim, *spatial) or (B, ndim, *spatial).
        The ndim dimension contains vector components and is not considered spatial.
    grid : torch.Tensor or None, default=None
        Pre-computed identity coordinate grid of shape (ndim, *spatial). If None,
        computed from transformation field shape. Useful to avoid recomputing the
        grid in training loops.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Batch dimensions preceding the ndim dimension. Use (0,) for batched input
        (B, ndim, *spatial). If None, assumes unbatched (ndim, *spatial).

    Returns
    -------
    torch.Tensor
        Displacement field with the same shape as input.

    Examples
    --------
    >>> import torch
    >>> import voxelmorph as vxm
    >>> # Identity transformation produces zero displacement
    >>> import neurite as ne
    >>> trf = ne.volshape_to_ndgrid((8, 8), stack=True)
    >>> disp = vxm.trf_to_disp(trf)
    >>> disp.abs().max()
    tensor(0.)

    >>> # Round-trip conversion
    >>> original = torch.randn(2, 32, 32)
    >>> recovered = vxm.trf_to_disp(vxm.disp_to_trf(original))
    >>> torch.allclose(original, recovered, atol=1e-5)
    True

    >>> # With pre-computed grid
    >>> grid = ne.volshape_to_ndgrid((32, 32), stack=True)
    >>> disp = vxm.trf_to_disp(trf, grid=grid)

    See Also
    --------
    disp_to_trf : Inverse operation.
    """
    if grid is None:
        num_non_spatial, _ = ne.functional.parse_non_spatial_dims(non_spatial_dims, trf.dim())
        spatial_shape = trf.shape[num_non_spatial + 1:]
        grid = ne.volshape_to_ndgrid(
            size=spatial_shape, device=trf.device, dtype=trf.dtype, stack=True
        )
    return trf - grid


def disp_to_coords(
    disp: torch.Tensor,
    meshgrid: Optional[torch.Tensor] = None,
    non_spatial_dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Convert displacement field to normalized coordinates in [-1, 1] range for grid_sample.

    Adds displacement to base meshgrid coordinates and normalizes to [-1, 1].

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (ndim, *spatial) or (B, ndim, *spatial) if batched.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from displacement field shape.
    non_spatial_dims : tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions preceding the ndim dimension:
        - None: tensor is (ndim, *spatial), unbatched
        - (0,): tensor is (B, ndim, *spatial), batched

    Returns
    -------
    torch.Tensor
        Normalized coordinates in range [-1, 1] with same shape as input.

    Examples
    --------
    >>> # 2d displacement field (ndim, H, W)
    >>> disp = torch.randn(2, 64, 64)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([2, 64, 64])

    >>> # Batched displacement field (B, ndim, H, W)
    >>> disp = torch.randn(4, 2, 64, 64)
    >>> coords = disp_to_coords(disp, non_spatial_dims=(0,))
    >>> coords.shape
    torch.Size([4, 2, 64, 64])
    """
    return vxf.disp_to_coords(
        disp=disp,
        meshgrid=meshgrid,
        non_spatial_dims=non_spatial_dims,
    )


def coords_to_disp(
    coords: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Convert normalized coordinates to displacement field.

    This inverse operation is not implemented yet and currently raises NotImplementedError.

    Parameters
    ----------
    coords : torch.Tensor
        Normalized coordinates in range [-1, 1] (output from grid_sample or disp_to_coords).
        Shape: (*non_spatial, *spatial, ndim) - channels-last format from grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from coordinate field spatial shape.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Which leading dimensions are non-spatial (before spatial dims):
        - None: pure spatial coordinate field (*spatial, ndim)
        - (0,): first dimension is non-spatial, e.g., (B, *spatial, ndim)

    Returns
    -------
    torch.Tensor
        Displacement field in channels-first format once implemented.

    Raises
    ------
    NotImplementedError
        Always raised by the current implementation.
    """
    return vxf.coords_to_disp(
        coords=coords,
        meshgrid=meshgrid,
        non_spatial_dims=non_spatial_dims,
    )


def spatial_transform(
    image: torch.Tensor,
    trf: Union[torch.Tensor, None],
    mode: Literal['linear', 'nearest'] = 'linear',
    isdisp: bool = True,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    align_corners: bool = True,
    padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros'
) -> torch.Tensor:
    """
    Apply spatial transformation to image using displacement or coordinate field.

    Shape-agnostic implementation that works with any tensor dimensionality.

    Parameters
    ----------
    image : torch.Tensor
        Input image to transform. Shape depends on non_spatial_dims:
        - (*spatial,) if non_spatial_dims=None
        - (C, *spatial) if non_spatial_dims=(0,)
        - (B, C, *spatial) if non_spatial_dims=(0, 1)
        - etc...
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Batched affine matrix: shape (B, N+1, N+1) or (B, N, N+1)
        - Displacement field: shape (N, *spatial) or (B, N, *spatial) - channels-first
        - None: returns image unchanged
    mode : {'linear', 'nearest'}, default='linear'
        Interpolation mode. 'linear' will auto-detect appropriate mode (bilinear/trilinear) based
        on spatial dimensionality.
    isdisp : bool, default=True
        If True, treat trf as displacement field (ndim, *spatial) and normalize to [-1, 1].
        If False, treat trf as already-normalized coordinates (ndim, *spatial).
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed from image shape.
    origin_at_center : bool, default=True
        Place origin at image center when converting affine matrices to displacement.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Which dimensions of image are non-spatial:
        - None: pure spatial tensor
        - (0,): first dimension is non-spatial (e.g., channel)
        - (0, 1): first two dimensions are non-spatial (e.g., batch, channel)
        - etc...
    align_corners : bool, default=True
        Align corners parameter for grid_sample.
    padding_mode : {'zeros', 'border', 'reflection'}, default='zeros'
        Padding mode for grid_sample when sampling outside the input bounds.

    Returns
    -------
    torch.Tensor
        Transformed image with same shape as input.

    Examples
    --------
    >>> # Pure spatial image (H, W) with displacement (ndim, H, W)
    >>> image = torch.randn(64, 64)
    >>> disp = torch.randn(2, 64, 64)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([64, 64])

    >>> # Image with channel dimension (C, H, W)
    >>> image = torch.randn(3, 64, 64)
    >>> disp = torch.randn(2, 64, 64)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0,))
    >>> warped.shape
    torch.Size([3, 64, 64])

    >>> # Image with batch and channel (B, C, H, W), batched displacement (B, ndim, H, W)
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(2, 2, 64, 64)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0, 1))
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # Batched affine transformations (different transform per batch)
    >>> image = torch.randn(2, 3, 64, 64)
    >>> affines = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (2, 3, 3)
    >>> warped = spatial_transform(image, affines, non_spatial_dims=(0, 1))
    >>> warped.shape
    torch.Size([2, 3, 64, 64])
    """
    return vxf.spatial_transform(
        image=image,
        trf=trf,
        method=mode,
        isdisp=isdisp,
        meshgrid=meshgrid,
        origin_at_center=origin_at_center,
        non_spatial_dims=non_spatial_dims,
        align_corners=align_corners,
        padding_mode=padding_mode,
    )


def integrate_disp(
    disp: torch.Tensor,
    steps: int,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> torch.Tensor:
    """
    Integrate a stationary velocity field to produce a displacement field.

    Uses the scaling-and-squaring method to efficiently compute the exponential
    map of the velocity field.

    Parameters
    ----------
    disp : torch.Tensor
        Velocity field with shape (ndim, *spatial) or (B, ndim, *spatial) if batched.
    steps : int
        Number of integration steps. The velocity is divided by 2^steps and then
        composed with itself 2^steps times. More steps = more accurate but slower.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from displacement field shape.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is (ndim, *spatial), unbatched
        - (0,): tensor is (B, ndim, *spatial), batched

    Returns
    -------
    torch.Tensor
        Integrated displacement field with same shape as input.

    Examples
    --------
    >>> import voxelmorph as vxm
    >>> # Unbatched velocity field
    >>> vel = torch.randn(2, 64, 64) * 0.1
    >>> disp = vxm.integrate_disp(vel, steps=7)
    >>> disp.shape
    torch.Size([2, 64, 64])

    >>> # Batched velocity field
    >>> vel = torch.randn(4, 2, 64, 64) * 0.1
    >>> disp = vxm.integrate_disp(vel, steps=7, non_spatial_dims=(0,))
    >>> disp.shape
    torch.Size([4, 2, 64, 64])
    """
    return vxf.integrate_disp(
        disp=disp,
        steps=steps,
        meshgrid=meshgrid,
        non_spatial_dims=non_spatial_dims,
    )


def resize_disp(
    disp: torch.Tensor,
    scale_factor: Union[float, Sequence[float], None] = None,
    shape: Union[Sequence[int], None] = None,
    mode: Literal['linear', 'nearest'] = 'linear',
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> torch.Tensor:
    """
    Resize a displacement field spatially and scale magnitudes proportionally.

    When resizing a displacement field, the vector magnitudes must be scaled along with
    the spatial dimensions. A 1-pixel displacement in a 64x64 field should become a
    2-pixel displacement when upsampled to 128x128.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (ndim, *spatial) or (B, ndim, *spatial) if batched.
    scale_factor : float, Sequence[float], or None, default=None
        Factor by which to scale spatial dimensions. Values > 1 upsample, < 1 downsample.
        Can be a scalar (uniform scaling) or a sequence with one factor per spatial dimension.
        Mutually exclusive with `shape`.
    shape : Sequence[int] or None, default=None
        Target spatial shape. Mutually exclusive with `scale_factor`.
    mode : {'linear', 'nearest'}, default='linear'
        Interpolation mode for spatial resizing.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is (ndim, *spatial), unbatched
        - (0,): tensor is (B, ndim, *spatial), batched

    Returns
    -------
    torch.Tensor
        Resized displacement field with same batch structure as input.

    Examples
    --------
    >>> # Upsample 2x using scale_factor
    >>> disp = torch.randn(2, 32, 32)
    >>> resized = resize_disp(disp, scale_factor=2.0)
    >>> resized.shape
    torch.Size([2, 64, 64])

    >>> # Downsample to specific shape
    >>> disp = torch.randn(3, 64, 64, 64)
    >>> resized = resize_disp(disp, shape=(32, 32, 32))
    >>> resized.shape
    torch.Size([3, 32, 32, 32])

    >>> # Magnitude scaling: 1-pixel shift becomes 2-pixel shift when upsampled 2x
    >>> disp = torch.ones(2, 4, 4)  # constant 1-pixel shift
    >>> resized = resize_disp(disp, scale_factor=2.0)
    >>> resized[0, 0, 0].item()  # now 2-pixel shift
    2.0

    >>> # Batched displacement field
    >>> disp = torch.randn(4, 2, 32, 32)
    >>> resized = resize_disp(disp, scale_factor=2.0, non_spatial_dims=(0,))
    >>> resized.shape
    torch.Size([4, 2, 64, 64])

    Notes
    -----
    Exactly one of `scale_factor` or `shape` must be provided.
    """
    assert (scale_factor is None) != (shape is None), (
        "Exactly one of `scale_factor` or `shape` must be provided"
    )

    # Parse dimensions
    num_non_spatial, num_spatial = ne.functional.parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=disp.ndim - 1  # subtract 1 for ndim dimension
    )

    has_batch = num_non_spatial == 1

    # Determine spatial shape
    if has_batch:
        spatial_shape = disp.shape[2:]
    else:
        spatial_shape = disp.shape[1:]

    ndim = len(spatial_shape)

    if shape is not None:
        assert len(shape) == ndim, (
            f"shape has {len(shape)} dims but disp has {ndim} spatial dims"
        )
        scale_factors = [shape[i] / spatial_shape[i] for i in range(ndim)]

    elif isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == ndim, (
            f"scale_factor has {len(scale_factor)} elements but disp has {ndim} spatial dims"
        )
        scale_factors = list(scale_factor)
    else:
        scale_factors = [scale_factor] * ndim

    if all(s == 1.0 for s in scale_factors):
        return disp

    # Determine interpolation mode and parameters
    if mode == 'linear':
        interp_mode = ne.utils.infer_linear_interpolation_mode(ndim)
        align_corners = True
    else:
        interp_mode = 'nearest'
        align_corners = None

    # interpolate expects (B, C, *spatial) - add batch dim if unbatched
    if not has_batch:
        disp = disp.unsqueeze(0)

    disp = torch.nn.functional.interpolate(
        disp,
        size=tuple(shape) if shape is not None else None,
        scale_factor=tuple(scale_factors) if shape is None else None,
        mode=interp_mode,
        align_corners=align_corners
    )

    if not has_batch:
        disp = disp.squeeze(0)

    # Scale each displacement component by its corresponding dimension's factor
    # Shape: (1, ndim, 1, 1, ...) for batched, (ndim, 1, 1, ...) for unbatched
    if has_batch:
        scale_tensor = torch.tensor(
            scale_factors, device=disp.device, dtype=disp.dtype
        ).view(1, -1, *[1] * ndim)
    else:
        scale_tensor = torch.tensor(
            scale_factors, device=disp.device, dtype=disp.dtype
        ).view(-1, *[1] * ndim)

    disp = disp * scale_tensor

    return disp


def compose(
    transforms: Sequence[torch.Tensor],
    interpolation_mode: str = 'linear',
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None,
) -> torch.Tensor:
    """
    Compose a single transform from a series of transforms.

    Supports both affine matrices and dense displacement fields. Returns a displacement
    field unless all inputs are affine matrices. For transforms [A, B, C], the composed
    transform T satisfies T(x) = A(B(C(x))), meaning C is applied first, then B, then A.

    Parameters
    ----------
    transforms : Sequence[Tensor]
        List or tuple of affine matrices and/or displacement fields to compose.
        - Affine matrices: shape (..., N, N+1) or (..., N+1, N+1)
        - Displacement fields: shape (ndim, *spatial) or (B, ndim, *spatial)
    interpolation_mode : str, default='linear'
        Interpolation method for composing displacement fields. Options are:
        {'linear', 'nearest'}.
    origin_at_center : bool, default=True
        Shift grid origin to image center when converting affine matrices to displacement fields.
    shape : Sequence[int] or None, default=None
        Spatial shape (N dimensions) for converting affine matrices to displacement fields.
        Only used if the rightmost transform is an affine matrix. If None and the rightmost
        transform is an affine, you must have at least one displacement field in the list.
        Incompatible with origin_at_center=False.

    Returns
    -------
    torch.Tensor
        Composed transform as either:
        - Affine matrix of shape (..., N, N+1) if all inputs are affine
        - Displacement field of shape (N, *spatial_shape) otherwise

    Examples
    --------
    >>> import voxelmorph as vxm
    >>> # Compose two affine matrices
    >>> translate = torch.tensor([[1., 0., 10.],
    ...                           [0., 1., 5.]])
    >>> scale = torch.tensor([[2., 0., 0.],
    ...                       [0., 2., 0.]])
    >>> composed = vxm.compose([translate, scale])
    >>> # Result is affine: scale applied first, then translate

    >>> # Compose affine with unbatched displacement field (ndim, H, W)
    >>> disp = torch.randn(2, 64, 64)
    >>> affine = torch.tensor([[1., 0., 5.],
    ...                        [0., 1., 3.]])
    >>> composed = vxm.compose([affine, disp])
    >>> composed.shape
    torch.Size([2, 64, 64])

    >>> # Compose batched displacement fields (B, ndim, H, W)
    >>> disp1 = torch.randn(4, 2, 64, 64)
    >>> disp2 = torch.randn(4, 2, 64, 64)
    >>> composed = vxm.compose([disp1, disp2])
    >>> composed.shape
    torch.Size([4, 2, 64, 64])

    Notes
    -----
    The composition uses matrix indexing ('ij') consistently. When composing displacement
    fields, the left field is interpolated using the right field as sampling coordinates.

    Batch dimensions are automatically detected for displacement fields by comparing
    tensor.ndim to the expected ndim + 1 (unbatched) or ndim + 2 (batched).
    """
    return vxf.compose(
        transforms=transforms,
        interpolation_mode=interpolation_mode,
        origin_at_center=origin_at_center,
        shape=shape,
    )


def constant_shift_field(
    spatial_shape: Sequence[int],
    shift_size: Union[int, float, Sequence[Union[int, float]], torch.Tensor] = 1,
    normalize: bool = False,
    device: Union[str, torch.device] = 'cpu',
) -> torch.Tensor:
    """
    Generate a constant displacement field for N-dimensional space.

    This function creates a displacement field where every spatial location has the same
    displacement vector. Each channel represents displacement along one spatial axis.

    Parameters
    ----------
    spatial_shape : Sequence[int]
        Shape of the spatial dimensions, e.g., (H, W) for 2D or (D, H, W) for 3D.
    shift_size : int, float, Sequence[int or float], or torch.Tensor, default=1
        Displacement magnitude for each spatial axis.
        - If scalar: same displacement for all axes
        - If Sequence: length must equal number of spatial dimensions
        - If Tensor: must have shape (n_spatial_dims,)
    normalize : bool, default=False
        If True, normalize the first spatial channel by (size - 1), where
        size is the extent of the first spatial dimension.
    device : str or torch.device, default='cpu'
        Device on which to create the tensor.

    Returns
    -------
    torch.Tensor
        Displacement field with shape (n_spatial_dims, *spatial_shape).
        Channel i contains the displacement along spatial axis i.

    Examples
    --------
    >>> import voxelmorph as vxm
    >>> # Create 2D displacement field
    >>> flow = vxm.constant_shift_field((4, 4), shift_size=1.0)
    >>> flow.shape
    torch.Size([2, 4, 4])
    >>> # All locations shift by 1 in both x and y
    >>> flow[:, 0, 0]
    tensor([1., 1.])

    >>> # Create 3D field with different shift per axis
    >>> flow = vxm.constant_shift_field((4, 4, 4), shift_size=[1.0, 2.0, 3.0])
    >>> flow.shape
    torch.Size([3, 4, 4, 4])
    >>> flow[:, 0, 0, 0]
    tensor([1., 2., 3.])

    >>> # Normalized shift for first dimension
    >>> flow = vxm.constant_shift_field((5, 5), shift_size=4.0, normalize=True)
    >>> flow[0, 0, 0]  # 4.0 / (5 - 1) = 1.0
    tensor(1.)
    >>> flow[1, 0, 0]  # Unchanged
    tensor(4.)
    """
    ndim = len(spatial_shape)

    # Normalize shift_size to tensor
    if isinstance(shift_size, (int, float)):
        shift = torch.full((ndim,), shift_size, dtype=torch.float32)
    elif isinstance(shift_size, torch.Tensor):
        shift = shift_size.float()
    else:
        shift = torch.tensor(shift_size, dtype=torch.float32)

    assert shift.shape[0] == ndim, f'shift_size must have {ndim} elements, got {shift.shape[0]}'

    # Broadcast shift values to full field: (ndim, *spatial_shape)
    shift = shift.view(-1, *[1] * ndim).to(device=device)
    field = shift.expand(ndim, *spatial_shape).clone()

    if normalize:
        field[0] /= (spatial_shape[0] - 1)

    return field


def is_affine_shape(shape: tuple) -> bool:
    """
    Determine whether the given shape represents an N-dimensional affine matrix.

    An affine matrix has shape (..., M, N+1) where:
    - N is the spatial dimensionality (2 or 3)
    - M is either N or N+1 (compact or square form)

    Parameters
    ----------
    shape : tuple
        Shape of the tensor to check.

    Returns
    -------
    bool
        True if shape represents an affine matrix, False otherwise.
    """
    if len(shape) < 2:
        return False

    rows, cols = shape[-2], shape[-1]

    # Cols should be N+1 where N is 2 or 3
    ndim = cols - 1
    if ndim not in (2, 3):
        return False

    # rows should be N or N+1
    if rows not in (ndim, ndim + 1):
        return False

    return True


def make_square_affine(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert affine matrix from compact form (..., N, N+1) to square form (..., N+1, N+1).

    Adds the homogeneous row [0, 0, ..., 0, 1] to the bottom of the matrix.

    Parameters
    ----------
    mat : Tensor
        Affine matrix of shape (..., M, N+1) where M is N or N+1.

    Returns
    -------
    Tensor
        Square affine matrix of shape (..., N+1, N+1).

    Examples
    --------
    >>> affine = torch.tensor(
    >>> ... [[1., 0., 5.],
    >>> ... [0., 1., 3.]]
    >>> )
    >>> square = make_square_affine(affine)
    >>> square.shape
    torch.Size([3, 3])
    >>> square[-1]
    tensor([0., 0., 1.])
    """
    assert is_affine_shape(mat.shape), f'Invalid affine shape: {mat.shape}'

    # Already square
    if mat.shape[-2] == mat.shape[-1]:
        return mat

    # Get dimensions
    *batch_dims, rows, cols = mat.shape

    # Create bottom row as [0, 0, ..., 0, 1]
    bottom_row = torch.zeros(*batch_dims, 1, cols, dtype=mat.dtype, device=mat.device)
    bottom_row[..., 0, -1] = 1.0

    return torch.cat([mat, bottom_row], dim=-2)


def random_field(
    shape: Sequence[int],
    scales: Union[float, int, Sequence[Union[float, int]]] = 10,
    magnitude: Union[float, int] = 10,
    integrations: int = 0,
    voxsize: Union[float, int] = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample'
) -> torch.Tensor:
    """Generate a random multi-component field using fractal noise.

    Creates one independent fractal-noise component per spatial dimension and stacks the
    components in channels-first format. When `integrations` is greater than zero, the field is
    interpreted as a stationary velocity field and integrated for compatibility with
    `random_disp`.

    Parameters
    ----------
    shape : Sequence[int]
        Shape from which to generate the field. Interpretation depends on `non_spatial_dims`:
        - `non_spatial_dims=None`: `(*spatial,)`, output is `(ndim, *spatial)`
        - `non_spatial_dims=(0,)`: `(B, *spatial)`, output is `(B, ndim, *spatial)`
        - `non_spatial_dims=(0, 1)`: `(B, C, *spatial)`, channel `C` is ignored
    scales : float, int, or Sequence[float or int], default=10
        Smoothing scale or scales for fractal noise, divided by `voxsize`. Interpretation depends
        on `fractal_mode`:
        - `fractal_mode='blur'`: sigma values for Gaussian smoothing
        - `fractal_mode='upsample'`: downsampling factors for upsampled noise
    magnitude : float or int, default=10
        Standard deviation of field values, divided by `voxsize`.
    integrations : int, default=0
        Number of integration steps for a diffeomorphic field. If zero, no integration is applied.
    voxsize : float or int, default=1
        Voxel size used to scale `scales` and `magnitude`.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid of shape `(ndim, *spatial)` used for integration. If None and
        `integrations > 0`, the grid is computed internally.
    non_spatial_dims : Sequence[int] or None, default=None
        Indices of non-spatial dimensions. At most batch and channel dimensions are supported.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal-noise generation method:
        - `'blur'`: generate noise and apply Gaussian smoothing (higher quality)
        - `'upsample'`: generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Random field in channels-first format. The component axis has length `ndim`, the number of
        spatial dimensions.

    Examples
    --------
    >>> # Pure spatial 2D field
    >>> field = random_field(shape=(64, 64), scales=5.0, magnitude=3.0)
    >>> field.shape
    torch.Size([2, 64, 64])

    >>> # 3D field with integration
    >>> field = random_field(shape=(32, 32, 32), integrations=5)
    >>> field.shape
    torch.Size([3, 32, 32, 32])

    >>> # Batched field
    >>> field = random_field(shape=(4, 64, 64), non_spatial_dims=(0,))
    >>> field.shape
    torch.Size([4, 2, 64, 64])
    """
    return vxf.random_field(
        shape=shape,
        scales=scales,
        magnitude=magnitude,
        integrations=integrations,
        voxsize=voxsize,
        meshgrid=meshgrid,
        non_spatial_dims=non_spatial_dims,
        device=device,
        fractal_mode=fractal_mode,
    )


def random_disp(
    shape: Sequence[int],
    scales: Union[float, int, Sequence[Union[float, int]]] = 10,
    magnitude: Union[float, int] = 10,
    integrations: int = 0,
    voxsize: Union[float, int] = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample'
) -> torch.Tensor:
    """Generate a random displacement field using fractal noise.

    Creates a displacement field by generating independent fractal noise for each spatial
    dimension and stacking them in channels-first format. This backward-compatible function
    delegates to `random_field`; use `random_field` when the values have another interpretation.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the displacement field. Interpretation depends on `non_spatial_dims`:
        - `non_spatial_dims=None`: `(*spatial,)`, output is `(ndim, *spatial)`
        - `non_spatial_dims=(0,)`: `(B, *spatial)`, output is `(B, ndim, *spatial)`
    scales : float, int, or Sequence[float or int], default=10
        Smoothing scale or scales for fractal noise, divided by `voxsize`. Interpretation depends
        on `fractal_mode`:
        - `fractal_mode='blur'`: sigma values for Gaussian smoothing
        - `fractal_mode='upsample'`: downsampling factors for upsampled noise
    magnitude : float or int, default=10
        Standard deviation of displacement in voxel coordinates, divided by `voxsize`.
    integrations : int, default=0
        Number of integration steps for a diffeomorphic transform. If zero, no integration is
        applied.
    voxsize : float or int, default=1
        Voxel size used to scale `scales` and `magnitude`.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid of shape `(ndim, *spatial)` used for integration. If None and
        `integrations > 0`, the grid is computed internally.
    non_spatial_dims : Sequence[int] or None, default=None
        Indices of non-spatial dimensions. At most batch and channel dimensions are supported.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal-noise generation method:
        - `'blur'`: generate noise and apply Gaussian smoothing (higher quality)
        - `'upsample'`: generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Displacement field in channels-first format:
        - `(ndim, *spatial)` if `non_spatial_dims=None`
        - `(B, ndim, *spatial)` if `non_spatial_dims=(0,)`

    Examples
    --------
    >>> # Pure spatial 2D displacement field
    >>> disp = random_disp(shape=(64, 64), scales=5.0, magnitude=3.0)
    >>> disp.shape
    torch.Size([2, 64, 64])

    >>> # 3D displacement with integration
    >>> disp = random_disp(shape=(32, 32, 32), integrations=5)
    >>> disp.shape
    torch.Size([3, 32, 32, 32])

    >>> # Batched displacement field
    >>> disp = random_disp(shape=(4, 64, 64), non_spatial_dims=(0,))
    >>> disp.shape
    torch.Size([4, 2, 64, 64])
    """
    return random_field(
        shape=shape,
        scales=scales,
        magnitude=magnitude,
        integrations=integrations,
        voxsize=voxsize,
        meshgrid=meshgrid,
        non_spatial_dims=non_spatial_dims,
        device=device,
        fractal_mode=fractal_mode,
    )


def random_transform(
    shape: Sequence[int],
    affine_probability: float = 1.0,
    max_translation: float = 5.0,
    max_rotation: float = 5.0,
    max_scaling: float = 1.1,
    warp_probability: float = 1.0,
    warp_integrations: int = 5,
    warp_scales_range: Sequence[float] = (10, 20),
    warp_magnitude_range: Sequence[float] = (1, 2),
    voxsize: Union[float, int] = 1,
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample',
    sampling: bool = True,
) -> torch.Tensor:
    """
    Generate a random spatial transformation combining affine and nonlinear warps.

    Creates a displacement field by optionally combining:
    1. Random affine transformation (translation, rotation, scaling)
    2. Random nonlinear warp using fractal noise

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the transformation field. Interpretation depends on non_spatial_dims:
        - non_spatial_dims=None: (*spatial,) pure spatial, output is (ndim, *spatial)
        - non_spatial_dims=(0,): (B, *spatial), output is (B, ndim, *spatial)
    affine_probability : float, default=1.0
        Probability of applying an affine transformation.
    max_translation : float, default=5.0
        Maximum translation in voxel coordinates (before dividing by voxsize).
    max_rotation : float, default=5.0
        Maximum rotation in degrees.
    max_scaling : float, default=1.1
        Maximum scaling factor (min is 1/max_scaling).
    warp_probability : float, default=1.0
        Probability of applying a nonlinear warp.
    warp_integrations : int, default=5
        Number of integration steps for diffeomorphic warp.
    warp_scales_range : Sequence[float], default=(10, 20)
        Range (min, max) to sample smoothing scales for fractal noise.
    warp_magnitude_range : Sequence[float], default=(1, 2)
        Range (min, max) to sample displacement magnitude.
    voxsize : float or int, default=1
        Voxel size for scaling translation, smoothing, and magnitude parameters.
    non_spatial_dims : Sequence of int or None, default=None
        Indices of non-spatial dimensions (only batch dimension supported):
        - None: tensor is pure spatial (*spatial,)
        - (0,): first dim is batch (B, *spatial)
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method for nonlinear warp.
    sampling : bool, default=True
        If True, sample random parameters. If False, use maximum values directly.

    Returns
    -------
    torch.Tensor
        Displacement field in channels-first format:
        - (ndim, *spatial) if non_spatial_dims=None
        - (B, ndim, *spatial) if non_spatial_dims=(0,)
        Returns None if both affine and warp probabilities result in no transform.

    Examples
    --------
    >>> # Pure spatial 2D transform
    >>> trf = random_transform(shape=(64, 64))
    >>> trf.shape
    torch.Size([2, 64, 64])

    >>> # 3D transform with custom parameters
    >>> trf = random_transform(
    ...     shape=(32, 32, 32),
    ...     max_rotation=10.0,
    ...     warp_magnitude_range=(2, 5)
    ... )
    >>> trf.shape
    torch.Size([3, 32, 32, 32])

    >>> # Batched transform
    >>> trf = random_transform(shape=(4, 64, 64), non_spatial_dims=(0,))
    >>> trf.shape
    torch.Size([4, 2, 64, 64])
    """
    return vxf.random_transform(
        shape=shape,
        affine_probability=affine_probability,
        max_translation=max_translation,
        max_rotation=max_rotation,
        max_scaling=max_scaling,
        warp_probability=warp_probability,
        warp_integrations=warp_integrations,
        warp_scales_range=warp_scales_range,
        warp_magnitude_range=warp_magnitude_range,
        voxsize=voxsize,
        non_spatial_dims=non_spatial_dims,
        device=device,
        fractal_mode=fractal_mode,
        sampling=sampling,
    )
