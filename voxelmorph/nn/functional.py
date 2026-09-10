"""
Functions containing the core operations and logic of for image registration for `voxelmorph`
written in PyTorch.
"""

# Core library imports
from functools import partial
from typing import Callable, List, Union, Sequence, Tuple, Literal, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import neurite as ne




def _is_affine_shape(shape: Tuple[int, ...]) -> bool:
    """
    Return whether a shape represents a compact or square 2D/3D affine matrix.
    """
    if len(shape) < 2:
        return False

    rows, cols = shape[-2], shape[-1]
    ndim = cols - 1
    if ndim not in (2, 3):
        return False

    if rows not in (ndim, ndim + 1):
        return False

    return True


def _make_square_affine(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert a compact affine matrix to square homogeneous form.
    """
    assert _is_affine_shape(mat.shape), f'Invalid affine shape: {mat.shape}'

    if mat.shape[-2] == mat.shape[-1]:
        return mat

    *batch_dims, rows, cols = mat.shape
    bottom_row = torch.zeros(*batch_dims, 1, cols, dtype=mat.dtype, device=mat.device)
    bottom_row[..., 0, -1] = 1.0
    return torch.cat([mat, bottom_row], dim=-2)


def _angles_to_rotation_matrix(rotation: torch.Tensor, degrees: bool = True) -> torch.Tensor:
    """
    Compute a 2D or 3D rotation matrix from rotation angle parameters.
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
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ])

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
    return rx @ ry @ rz


def _params_to_affine(
    ndim: int,
    translation: Union[torch.Tensor, None] = None,
    rotation: Union[torch.Tensor, None] = None,
    scale: Union[torch.Tensor, None] = None,
    shear: Union[torch.Tensor, None] = None,
    degrees: bool = True,
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    """
    Make an affine matrix from translation, rotation, scale, and shear parameters.
    """
    assert ndim in (2, 3), f'affine transform must be 2D or 3D, got ndim {ndim}'
    n_rotation_angles = 3 if ndim == 3 else 1

    translation = torch.zeros(ndim) if translation is None else torch.as_tensor(translation)
    assert len(translation) == ndim, f'Translation must be of shape ({ndim},)'

    rotation = torch.zeros(n_rotation_angles) if rotation is None else torch.as_tensor(rotation)
    rotation = torch.atleast_1d(rotation)
    assert rotation.shape[0] == n_rotation_angles, f'Rotation must be shape ({n_rotation_angles},)'

    scale = torch.ones(ndim) if scale is None else torch.as_tensor(scale)
    if scale.ndim == 0:
        scale = scale.repeat(ndim)
    assert scale.shape[0] == ndim, f'scale must be of size {ndim}'

    shear = torch.zeros(n_rotation_angles) if shear is None else torch.as_tensor(shear)
    shear = torch.atleast_1d(shear)
    assert shear.shape[0] == n_rotation_angles, f'shear must be of shape ({n_rotation_angles},)'

    t_mat = torch.eye(ndim + 1, dtype=torch.float64)
    t_mat[:ndim, -1] = translation

    r_mat = torch.eye(ndim + 1, dtype=torch.float64)
    r_mat[:ndim, :ndim] = _angles_to_rotation_matrix(rotation, degrees=degrees)

    z_mat = torch.diag(torch.cat([scale, torch.ones(1, dtype=torch.float64)]))

    s_mat = torch.eye(ndim + 1, dtype=torch.float64)
    s_mat[0][1] = shear[0]
    if ndim == 3:
        s_mat[0][2] = shear[1]
        s_mat[1][2] = shear[2]

    return torch.as_tensor(t_mat @ r_mat @ z_mat @ s_mat, dtype=torch.float32, device=device)


def _random_affine(
    ndim: int,
    max_translation: float = 0,
    max_rotation: float = 0,
    max_scaling: float = 1,
    device: Union[torch.device, None] = None,
    sampling: bool = True,
) -> torch.Tensor:
    """
    Generate a random affine transformation matrix.
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

    return _params_to_affine(
        ndim=ndim,
        translation=translation,
        rotation=rotation,
        scale=scale,
        device=device,
    )


def _affine_to_disp(
    affine: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None,
    warp_right: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Convert an affine transformation matrix to a displacement field.
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

    grid = meshgrid
    if origin_at_center:
        center_offsets = [(s - 1) / 2 for s in spatial_shape]
        center_offsets = torch.tensor(center_offsets, device=meshgrid.device).view(
            -1, *[1] * ndim
        )
        grid = meshgrid - center_offsets

    coords = grid.reshape(ndim, -1)

    if warp_right is not None:
        assert warp_right.shape[-ndim:] == spatial_shape, (
            f"warp_right shape {warp_right.shape[-ndim:]} != meshgrid {spatial_shape}"
        )
        coords = coords + warp_right.reshape(*warp_right.shape[:-ndim], -1)

    transformed = affine[..., :ndim, :ndim] @ coords + affine[..., :ndim, -1:]
    disp_flat = transformed - grid.reshape(ndim, -1)

    output_shape = (affine.shape[0], ndim, *spatial_shape) if is_batched else (ndim, *spatial_shape)
    return disp_flat.reshape(*output_shape)


def spatial_transform(
    image: torch.Tensor,
    trf: Union[torch.Tensor, None],
    method: Literal['nearest', 'linear'] = 'linear',
    isdisp: bool = True,
    meshgrid: Union[torch.Tensor, None] = None,
    origin_at_center: bool = True,
    non_spatial_dims: Union[Tuple[int, ...], None] = (0, 1),
    align_corners: bool = True,
    padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros'
) -> torch.Tensor:
    """
    Apply spatial transformation to image in (B, C, *spatial) format.

    Canonical implementation for tensors in (B, C, *spatial) format.

    Parameters
    ----------
    image : torch.Tensor
        Input image with shape (B, C, *spatial).
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Displacement field: shape (N, *spatial) - channels-first format
        - None: returns image unchanged
    method : str, default='linear'
        Interpolation mode ('linear' or 'nearest').
    isdisp : bool, default=True
        If True, treat trf as displacement field (N, *spatial). If False, treat as
        coordinates (*spatial, N) ready for grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial).
    origin_at_center : bool, default=True
        Place origin at image center for affine transformations.

    Returns
    -------
    torch.Tensor
        Transformed image with shape (B, C, *spatial).

    Examples
    --------
    >>> # 2D image with batch and channel
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(2, 64, 64)  # (ndim, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # 3D image with batch and channel
    >>> image = torch.randn(1, 1, 64, 64, 64)
    >>> disp = torch.randn(3, 64, 64, 64)  # (ndim, D, H, W)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([1, 1, 64, 64, 64])
    """
    if trf is None:
        return image

    num_non_spatial, num_spatial = ne.parse_non_spatial_dims(non_spatial_dims, image.ndim)
    spatial_shape = image.shape[num_non_spatial:]

    is_affine = False
    if trf.ndim == 2 and _is_affine_shape(trf.shape):
        is_affine = True
    elif trf.ndim == 3 and _is_affine_shape(trf.shape):
        trf_spatial_like = trf.shape[1:]
        if trf_spatial_like == spatial_shape:
            is_affine = False
        else:
            rows, cols = trf.shape[-2], trf.shape[-1]
            if rows <= 4 and cols <= 5:
                is_affine = True

    if is_affine:
        trf = torch.linalg.inv(trf)
        trf = _affine_to_disp(trf, meshgrid, shape=spatial_shape, origin_at_center=origin_at_center)
        isdisp = True

    trf_has_batch_dim = trf.ndim > (num_spatial + 1)
    if isdisp:
        trf_non_spatial = (0,) if trf_has_batch_dim else None
        trf = disp_to_coords(trf, meshgrid=meshgrid, non_spatial_dims=trf_non_spatial)

    ndim_dim = 1 if trf_has_batch_dim else 0
    trf = trf.movedim(ndim_dim, -1).flip(-1)

    mode = method
    if mode == 'linear':
        mode = ne.utils.infer_linear_interpolation_mode(num_spatial)
    if mode == 'trilinear':
        mode = 'bilinear'

    original_dtype = None
    if not torch.is_floating_point(image):
        if mode == 'nearest':
            original_dtype = image.dtype
        image = image.type(torch.float32)

    dims_added = 2 - num_non_spatial
    for _ in range(dims_added):
        image = image.unsqueeze(0)

    trf_has_batch_dim = trf.ndim > (num_spatial + 1)
    if not trf_has_batch_dim:
        trf = trf.unsqueeze(0)

    transformed = F.grid_sample(
        image, trf, align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )

    for _ in range(dims_added):
        transformed = transformed.squeeze(0)
    if original_dtype is not None:
        transformed = transformed.type(original_dtype)
    return transformed


def disp_to_coords(
    disp: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = (0,)
) -> torch.Tensor:
    """
    Convert displacement field to normalized coordinates for (B, ndim, *spatial) format.

    Adds displacement to base meshgrid coordinates and normalizes to [-1, 1] range.

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (B, ndim, *spatial).
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from displacement field spatial shape.

    Returns
    -------
    torch.Tensor
        Normalized coordinates in range [-1, 1] with shape (B, ndim, *spatial).

    Examples
    --------
    >>> # 2D displacement field with batch
    >>> disp = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([2, 2, 64, 64])

    >>> # 3D displacement field
    >>> disp = torch.randn(1, 3, 32, 32, 32)  # (B, ndim, D, H, W)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([1, 3, 32, 32, 32])
    """
    num_non_spatial, num_spatial = ne.parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=disp.ndim - 1,
    )
    has_batch = num_non_spatial == 1
    ndim_axis = 1 if has_batch else 0
    spatial_shape = disp.shape[ndim_axis + 1:]

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(
            size=spatial_shape,
            device=disp.device,
            dtype=disp.dtype,
            stack=True,
        )

    coords = meshgrid + disp
    sizes = torch.tensor(spatial_shape, device=disp.device, dtype=disp.dtype)
    scales = 2.0 / (sizes - 1).clamp(min=1)
    broadcast_shape = ((1,) if has_batch else ()) + (len(spatial_shape),) + (1,) * num_spatial
    scales = scales.view(broadcast_shape)
    return coords * scales - 1.0


def coords_to_disp(
    coords: torch.Tensor,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = (0,)
) -> torch.Tensor:
    """
    Convert normalized coordinates to displacement field for (B, ndim, *spatial) format.

    This inverse operation is not implemented yet and currently raises NotImplementedError,
    matching the previous top-level VoxelMorph behavior.

    Parameters
    ----------
    coords : torch.Tensor
        Normalized coordinates in range [-1, 1] with shape (B, *spatial, ndim).
        Channels-last format as output by grid_sample.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from coordinate field spatial shape.

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) once implemented.

    Raises
    ------
    NotImplementedError
        Always raised by the current implementation.
    """
    raise NotImplementedError(
        'coords_to_disp is not yet implemented. '
        'The inverse operations from disp_to_coords need to be applied: '
        'Contact andrew if you need this... or implement it :)'
    )


def integrate_vec(
    vec: torch.Tensor,
    steps: int,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = (0,),
    scale: Union[float, None] = None,
    transformer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Integrate velocity field via scaling and squaring for (B, ndim, *spatial) format.

    Converts a stationary velocity field into a displacement field through iterative
    composition. By default, the input is scaled by 1/2^steps, then composed with itself
    `steps` times. An explicit `scale` overrides the initial scaling.

    Parameters
    ----------
    vec : torch.Tensor
        Velocity field with shape (B, ndim, *spatial).
    steps : int
        Number of integration steps. If 0 and scale is None, returns vec unchanged.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        internally from vec spatial shape.
    scale : float or None, default=None
        Optional initial multiplier, replacing division by 2^steps. Applies even when steps is zero.
    transformer : callable or None, default=None
        Callable accepting (image, displacement) and returning the warped image.
        Called once per squaring step. If supplied, meshgrid and non_spatial_dims are ignored.

    Returns
    -------
    torch.Tensor
        Integrated displacement field with shape (B, ndim, *spatial).

    Examples
    --------
    >>> # 2D velocity field with batch
    >>> vel = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> disp = integrate_vec(vel, steps=7)
    >>> disp.shape
    torch.Size([2, 2, 64, 64])

    >>> # 3D velocity field
    >>> vel = torch.randn(1, 3, 32, 32, 32)  # (B, ndim, D, H, W)
    >>> disp = integrate_vec(vel, steps=5)
    >>> disp.shape
    torch.Size([1, 3, 32, 32, 32])
    """
    if steps == 0:
        return vec if scale is None else vec * scale

    if transformer is None:
        num_non_spatial, num_spatial = ne.parse_non_spatial_dims(
            non_spatial_dims=non_spatial_dims,
            tensor_ndim=vec.ndim - 1,
        )
        has_batch = num_non_spatial == 1
        if has_batch:
            spatial_shape = vec.shape[2:]
            st_non_spatial_dims = (0, 1)
        else:
            spatial_shape = vec.shape[1:]
            st_non_spatial_dims = (0,)

        if meshgrid is None:
            meshgrid = ne.volshape_to_ndgrid(
                size=spatial_shape,
                device=vec.device,
                dtype=vec.dtype,
                stack=True,
            )
        transformer = partial(
            spatial_transform,
            meshgrid=meshgrid,
            non_spatial_dims=st_non_spatial_dims,
        )

    disp = vec / (2 ** steps) if scale is None else vec * scale
    for _ in range(steps):
        disp = disp + transformer(disp, disp)
    return disp


def compose(
    transforms: Sequence[torch.Tensor],
    interpolation_mode: str = 'linear',
    origin_at_center: bool = True,
    shape: Union[Sequence[int], None] = None
) -> torch.Tensor:
    """
    Compose transforms for (B, ndim, *spatial) format displacement fields.

    Composes a sequence of transforms into a single transform. For transforms [A, B, C],
    the composed transform T satisfies T(x) = A(B(C(x))), meaning C is applied first,
    then B, then A.

    Parameters
    ----------
    transforms : Sequence[Tensor]
        List of transforms to compose. Each transform should be:
        - Displacement field: shape (B, ndim, *spatial)
        - Affine matrix: shape (N, N+1) or (N+1, N+1) or batched (B, N, N+1)
    interpolation_mode : str, default='linear'
        Interpolation method for composing displacement fields. Options are {'linear', 'nearest'}.
    origin_at_center : bool, default=True
        Place origin at image center when converting affine matrices to displacement.
    shape : Sequence[int] or None, default=None
        Spatial shape for converting affine matrices to displacement fields.
        Required if rightmost transform is an affine matrix.

    Returns
    -------
    torch.Tensor
        Composed transform as either:
        - Affine matrix if all inputs are affine
        - Displacement field with shape (B, ndim, *spatial) otherwise

    Examples
    --------
    >>> import voxelmorph.nn.functional as vxf
    >>> # Compose two batched displacement fields
    >>> disp1 = torch.randn(2, 2, 64, 64)  # (B, ndim, H, W)
    >>> disp2 = torch.randn(2, 2, 64, 64)
    >>> composed = vxf.compose([disp1, disp2])
    >>> composed.shape
    torch.Size([2, 2, 64, 64])

    >>> # Compose affine with batched displacement
    >>> affine = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (B, 3, 3)
    >>> disp = torch.randn(2, 2, 64, 64)
    >>> composed = vxf.compose([affine, disp])
    >>> composed.shape
    torch.Size([2, 2, 64, 64])
    """
    assert len(transforms) > 0, 'Cannot compose empty list of transforms'

    if len(transforms) == 1:
        return transforms[0]

    curr = transforms[-1]

    for next_trf in reversed(transforms[:-1]):
        curr_is_affine = _is_affine_shape(curr.shape)
        next_is_affine = _is_affine_shape(next_trf.shape)

        if curr_is_affine and next_is_affine:
            curr = (_make_square_affine(next_trf) @ _make_square_affine(curr))[..., :-1, :]
            continue

        if next_is_affine and not curr_is_affine:
            curr = _affine_to_disp(
                next_trf,
                shape=curr.shape[1:],
                origin_at_center=origin_at_center,
                warp_right=curr,
            )
            continue

        if curr_is_affine:
            curr = _affine_to_disp(
                affine=curr,
                shape=shape if shape is not None else next_trf.shape[1:],
                origin_at_center=origin_at_center,
            )

        ndim_if_unbatched = curr.shape[0]
        num_spatial_if_unbatched = curr.ndim - 1
        has_batch = ndim_if_unbatched != num_spatial_if_unbatched
        non_spatial_dims = (0, 1) if has_batch else (0,)

        warped = spatial_transform(
            image=next_trf,
            trf=curr,
            method=interpolation_mode,
            isdisp=True,
            non_spatial_dims=non_spatial_dims,
        )
        curr = curr + warped

    return curr


def random_disp(
    shape: Sequence[int],
    scales: Union[float, int, List[float]] = 10,
    magnitude: float = 10,
    integrations: int = 0,
    voxsize: float = 1,
    meshgrid: Union[torch.Tensor, None] = None,
    non_spatial_dims: Union[Sequence[int], None] = (0, 1),
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample'
) -> torch.Tensor:
    """
    Generate random displacement field for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) and outputs
    displacement field in (B, ndim, *spatial) format - channels-first format.
    The channel dimension is ignored since displacement is per-voxel, not per-channel.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
    scales : float, int, or List[float], default=10
        Smoothing scale(s) for fractal noise, divided by voxsize. Interpretation depends
        on fractal_mode:
        - fractal_mode='blur': sigma values for Gaussian smoothing
        - fractal_mode='upsample': downsampling factors for upsampled noise
    magnitude : float, default=10
        Standard deviation of displacement in voxel coordinates, divided by voxsize.
    integrations : int, default=0
        Number of integration steps for diffeomorphic transform. If 0, no integration.
    voxsize : float, default=1
        Voxel size for scaling smoothing and magnitude parameters.
    meshgrid : torch.Tensor or None, default=None
        Coordinate grid of shape (ndim, *spatial) for integration. If None and
        integrations > 0, computed internally.
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method:
        - 'blur': Generate noise and apply Gaussian smoothing (higher quality)
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) - channels-first format.

    Examples
    --------
    >>> # Generate displacement for 2D image with shape (B, C, H, W)
    >>> disp = random_disp(shape=(1, 1, 64, 64), scales=5.0, magnitude=3.0)
    >>> disp.shape
    torch.Size([1, 2, 64, 64])

    >>> # Generate displacement for 3D image with shape (B, C, D, H, W)
    >>> disp = random_disp(shape=(2, 3, 32, 32, 32), integrations=5)
    >>> disp.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    num_non_spatial, num_spatial = ne.parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape),
    )
    assert num_non_spatial <= 2, (
        "random_disp supports at most batch and channel non-spatial dims, "
        f"got non_spatial_dims={non_spatial_dims}"
    )
    has_batch = num_non_spatial >= 1
    batch_size = shape[0] if has_batch else 1
    spatial_shape = shape[num_non_spatial:]

    if np.isscalar(scales):
        scales = scales / voxsize
    else:
        scales = [s / voxsize for s in scales]
    magnitude = magnitude / voxsize

    disp_components = []
    noise_shape = (batch_size, *spatial_shape) if has_batch else spatial_shape
    noise_non_spatial = (0,) if has_batch else None
    for _ in range(num_spatial):
        noise = ne.fractal_noise(
            shape=noise_shape,
            scales=scales,
            magnitude=magnitude,
            non_spatial_dims=noise_non_spatial,
            device=device,
            method=fractal_mode,
        )
        disp_components.append(noise)

    stack_dim = 1 if has_batch else 0
    disp = torch.stack(disp_components, dim=stack_dim)
    if integrations > 0:
        disp = integrate_vec(
            disp,
            integrations,
            meshgrid,
            non_spatial_dims=(0,) if has_batch else None,
        )
    return disp


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
    non_spatial_dims: Union[Sequence[int], None] = (0, 1),
    device: Union[torch.device, None] = None,
    fractal_mode: Literal['blur', 'upsample'] = 'upsample',
    sampling: bool = True,
) -> torch.Tensor:
    """
    Generate random spatial transformation for images in (B, C, *spatial) format.

    Takes shape in (B, C, *spatial) format (matching image tensors) and outputs
    displacement field in (B, ndim, *spatial) format - channels-first format.

    Parameters
    ----------
    shape : Sequence[int]
        Shape in (B, C, *spatial) format matching the image to be transformed.
        Examples: (1, 1, 64, 64) for 2D, (2, 3, 64, 64, 64) for 3D.
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
    device : torch.device or None, default=None
        Device for tensor allocation.
    fractal_mode : {'blur', 'upsample'}, default='upsample'
        Fractal noise generation method for nonlinear warp.
    sampling : bool, default=True
        If True, sample random parameters. If False, use maximum values directly.

    Returns
    -------
    torch.Tensor
        Displacement field with shape (B, ndim, *spatial) - channels-first format.

    Examples
    --------
    >>> # Generate transform for 2D image with shape (B, C, H, W)
    >>> trf = random_transform(shape=(1, 1, 64, 64))
    >>> trf.shape
    torch.Size([1, 2, 64, 64])

    >>> # Generate transform for 3D image with shape (B, C, D, H, W)
    >>> trf = random_transform(shape=(2, 3, 32, 32, 32), max_rotation=10.0)
    >>> trf.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    num_non_spatial, num_spatial = ne.parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape),
    )
    assert num_non_spatial <= 2, (
        "random_transform supports at most batch and channel non-spatial dims, "
        f"got non_spatial_dims={non_spatial_dims}"
    )
    has_batch = num_non_spatial >= 1
    batch_size = shape[0] if has_batch else 1
    spatial_shape = shape[num_non_spatial:]
    meshgrid = ne.volshape_to_ndgrid(size=spatial_shape, device=device, stack=True)

    def generate_single_transform():
        trf = None
        if np.random.rand() < affine_probability:
            matrix = _random_affine(
                ndim=num_spatial,
                max_translation=max_translation / voxsize,
                max_rotation=max_rotation,
                max_scaling=max_scaling,
                device=device,
                sampling=sampling,
            )
            trf = _affine_to_disp(matrix, meshgrid)
        if np.random.rand() < warp_probability:
            disp = random_disp(
                shape=spatial_shape,
                scales=np.random.uniform(*warp_scales_range),
                magnitude=np.random.uniform(*warp_magnitude_range),
                integrations=warp_integrations,
                voxsize=voxsize,
                device=device,
                fractal_mode=fractal_mode,
                non_spatial_dims=None,
            )
            if trf is None:
                trf = disp
            else:
                trf = trf + spatial_transform(disp, trf, meshgrid=meshgrid, non_spatial_dims=(0,))
        if trf is None:
            trf = torch.zeros(num_spatial, *spatial_shape, device=device)
        return trf

    transforms = [generate_single_transform() for _ in range(batch_size)]
    return torch.stack(transforms, dim=0) if has_batch else transforms[0]
