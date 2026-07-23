"""
Unit tests for the basic utility functions in voxelmorph.
"""

# Standard library imports
import pytest
import torch

import voxelmorph as vxm
import voxelmorph.nn.functional as vxf
import neurite as ne
import neurite.nn.functional as nef


def test_affine_to_disp_identity():
    """
    Identity affine should produce zero displacement everywhere.
    """
    shape = (3, 4)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Identity affine
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)

    # Get displacement field
    disp = vxm.affine_to_disp(affine, grid)

    # output shape and dtype - now (ndim, *spatial)
    assert disp.shape == (ndim,) + shape
    assert disp.dtype == grid.dtype

    # all zeros
    assert torch.allclose(disp, torch.zeros_like(disp))


def test_affine_to_disp_translation():
    """
    Translation affine should yield a constant field.
    """
    shape = (2, 2)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)
    tx, ty = 2.0, 3.0

    # build a 2D affine with translation in the last column
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)

    # Make the translation
    affine[0, -1] = tx
    affine[1, -1] = ty

    # Get displacement field
    disp = vxm.affine_to_disp(affine, grid)

    # expected a field of shape (ndim, *spatial) = (2, 2, 2) filled with [tx, ty]
    expected = torch.stack(
        [
            torch.full(shape, tx, dtype=grid.dtype, device=grid.device),
            torch.full(shape, ty, dtype=grid.dtype, device=grid.device),
        ],
        dim=0  # Stack along first dim for channels-first format
    )

    assert disp.shape == expected.shape
    assert torch.allclose(disp, expected)


def test_disp_to_coords_zero_disp_2d():
    """
    Zero displacement on a 2x3 grid should produce the normalized mesh in range [-1, 1].

    Input disp is (ndim, *spatial) = (2, 2, 3).
    Output coords is (ndim, *spatial) = (2, 2, 3) - same shape, normalized values.
    """
    # Displacement is (ndim, H, W) = (2, 2, 3)
    disp = torch.zeros(2, 2, 3, dtype=torch.float32)
    coords = vxm.disp_to_coords(disp)

    # Output maintains (ndim, *spatial) = (2, 2, 3)
    assert coords.shape == (2, 2, 3)
    assert coords.dtype == torch.float32

    # For shape=(2, 3):
    #  coords[0] (row indices): shape (2, 3), values [-1, 1] along dim 0
    #  coords[1] (col indices): shape (2, 3), values [-1, 0, 1] along dim 1
    expected_row = torch.tensor([
        [-1., -1., -1.],
        [1., 1., 1.],
    ], dtype=torch.float32)
    expected_col = torch.tensor([
        [-1., 0., 1.],
        [-1., 0., 1.],
    ], dtype=torch.float32)

    assert torch.allclose(coords[0], expected_row)
    assert torch.allclose(coords[1], expected_col)


def test_spatial_transform_none_trf_returns_input():
    """
    If trf is None, spatial_transform should return the input image.
    """
    # Image shape (B, C, H, W) for vxf wrapper
    img = torch.rand(1, 1, 5, 5)
    out = vxf.spatial_transform(img, None)

    assert out.shape == img.shape
    assert torch.allclose(out, img)


def test_spatial_transform_identity_affine():
    """
    An identity affine should yield the same image.
    """
    img = torch.rand(1, 3, 3, dtype=torch.float32)

    # 2D identity affine (3×3)
    affine = torch.eye(3, dtype=torch.float32)
    out = vxm.spatial_transform(img, affine, non_spatial_dims=(0,))

    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-6)


def test_spatial_transform_rotation():
    """
    Test that spatial_transform rotates a horizontal line by 90 degrees and a vertical line by
    90 degrees.
    """

    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    tensor = torch.zeros(1, 5, 5)
    horizontal_line = tensor.clone()
    horizontal_line[0, 2, :] = 1

    vertical_line = tensor.clone()
    vertical_line[0, :, 2] = 1

    transformed_horizontal_line = vxm.spatial_transform(
        horizontal_line, trf=A, mode='linear', non_spatial_dims=(0,)
    )

    transformed_vertical_line = vxm.spatial_transform(
        vertical_line, trf=A, mode='linear', non_spatial_dims=(0,)
    )

    assert torch.allclose(horizontal_line, transformed_vertical_line)
    assert torch.allclose(vertical_line, transformed_horizontal_line)


def test_spatial_transform_batched():

    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # B: translation by (8, 0)
    B = torch.tensor([
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # 2 batches (images), one channel
    batched_images = torch.randn(2, 1, 28, 28)
    batched_transforms = torch.stack([A, B])

    vxm.spatial_transform(
        image=batched_images,
        trf=batched_transforms,
        non_spatial_dims=(0, 1)
    )


def test_spatial_transform_rotation_translation_batched():
    """
    Test that spatial_transform correctly applies different affine transformations to each batch
    element.

    This test verifies batched spatial transformations where:
    - Batch element 0: A 90-degree counter-clockwise rotation matrix transforms a horizontal line
      (at row 2) into a vertical line (at column 2)
    - Batch element 1: A translation matrix (shifts by 1 pixel in x-direction) transforms a 
      horizontal line at row 2 to row 3

    The input is a batch of 2 images, each containing a horizontal line at row 2 (middle row).
    After transformation, batch element 0 should have a vertical line at column 2, and batch 
    element 1 should have a horizontal line at row 3.
    """
    A = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    B = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    tensor = torch.zeros(2, 1, 5, 5)
    horizontal_lines = tensor.clone()
    horizontal_lines[:, 0, 2, :] = 1

    batched_transforms = torch.stack([A, B])

    # Apply batched transformation
    transformed = vxm.spatial_transform(
        horizontal_lines, trf=batched_transforms, mode='linear', non_spatial_dims=(0, 1)
    )

    transformed_batch_element_0 = torch.zeros(1, 5, 5)
    transformed_batch_element_0[0, :, 2] = 1

    transformed_batch_element_1 = torch.zeros(1, 5, 5)
    transformed_batch_element_1[0, 3, :] = 1

    # Verify batch element 0: rotation transforms horizontal line to vertical line
    assert torch.allclose(transformed[0], transformed_batch_element_0), (
        f"Batch element 0 failed: Horizontal line should map to a vertical line after 90 degree "
        f"rotation, but got shape {transformed[0].shape} with max diff "
        f"{torch.max(torch.abs(transformed[0] - transformed_batch_element_0))}"
    )

    # Verify batch element 1: translation shifts horizontal line from row 2 to row 3
    assert torch.allclose(transformed[1], transformed_batch_element_1), (
        f"Batch element 1 failed: Horizontal line at row 2 should map to a horizontal line at "
        f"row 3 after translation by (1, 0), but got shape {transformed[1].shape} with max diff "
        f"{torch.max(torch.abs(transformed[1] - transformed_batch_element_1))}"
    )


def test_angles_to_rotation_matrix_2d_identity():
    """
    A 2D rotation of 0 deg must yield the 2x2 identity matrix.
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(0.0), degrees=True)
    expected = torch.eye(2, dtype=torch.float64)

    assert rotation_matrix.shape == (2, 2)
    assert rotation_matrix.dtype == torch.float64
    assert torch.allclose(rotation_matrix, expected, atol=1e-8)


def test_angles_to_rotation_matrix_2d_90_degrees():
    """
    A 2D rotation of 90 degrees should be [[0, -1], [1, 0]].
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(90.0), degrees=True)

    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0]
        ],
        dtype=torch.float64
    )

    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_angles_to_rotation_matrix_2d_pi_over_2_radians():
    """
    With degrees=False and angle=pi/2, result should match the 90° case.
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor(torch.pi / 2), degrees=False)
    expected = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0]
        ],
        dtype=torch.float64
    )
    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_angles_to_rotation_matrix_3d_90_degrees():
    """
    A 3D rotation of 90 degrees around the z axis should be:
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 1]]
    """
    rotation_matrix = vxm.angles_to_rotation_matrix(torch.tensor((0, 0, 90.0)), degrees=True)

    expected = torch.tensor(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=torch.float64
    )

    assert torch.allclose(rotation_matrix, expected, atol=1e-5)


def test_angles_to_rotation_matrix_3d_preserves_gradients():
    """
    A 3D rotation matrix must preserve autograd history from all input angles.
    """
    rotation = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    rotation_matrix = vxm.angles_to_rotation_matrix(rotation, degrees=False)

    rotation_matrix.sum().backward()

    assert int(torch.count_nonzero(rotation.grad).item()) == 3

def test_params_to_affine_translation_shear():
    """
    Composing two translations should yield the sum of the two translations.
    """

    translation = (1, 2)

    result_affine = vxm.params_to_affine(
        ndim=2,
        translation=translation,
        shear=9,
    ).to(torch.float64)

    expected_affine = torch.tensor([[1, 9, 1], [0, 1, 2], [0, 0, 1]], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_resize_scale_nearest_int():
    """
    Nearest-neighbor upsampling of an integer image should replicate pixels.
    """
    img = torch.tensor(
        [[[1, 2],
          [3, 4]]],
        dtype=torch.int32
    )
    out = nef.resize(img, scale_factor=2.0, nearest=True)

    # Expect each pixel to become a 2×2 block
    expected = torch.tensor(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ],
        dtype=torch.int32
    ).unsqueeze(0)

    assert out.shape == (1, 4, 4)
    assert out.dtype == img.dtype
    assert torch.allclose(out, expected)


def test_affine_to_disp_scaling_2d():
    """
    Test scaling with a known expected displacement field.
    Use a simple 2x2 grid with 2x scaling to make the math tractable.
    """
    # Use a simple 2x2 grid for easier calculation
    shape = (2, 2)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Simple 2x scaling in both directions, centered around origin
    # This should be easier to calculate by hand
    scale_factor = 2.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, 0] = scale_factor  # x scaling
    affine[1, 1] = scale_factor  # y scaling
    # No translation - scale around origin

    disp = vxm.affine_to_disp(affine, meshgrid=grid, origin_at_center=False)

    # Check shape - now (ndim, *spatial)
    assert disp.shape == (ndim,) + shape

    # For a 2x2 grid with coordinates (0,0), (0,1), (1,0), (1,1)
    # With 2x scaling around origin:
    # (0,0) -> (0,0), displacement = (0,0) - (0,0) = (0,0)
    # (0,1) -> (0,2), displacement = (0,2) - (0,1) = (0,1)
    # (1,0) -> (2,0), displacement = (2,0) - (1,0) = (1,0)
    # (1,1) -> (2,2), displacement = (2,2) - (1,1) = (1,1)

    # Expected in channels-first format: (ndim, H, W)
    # disp[0] = x-displacements, disp[1] = y-displacements
    expected_disp = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0]],  # x-displacements
        [[0.0, 1.0], [0.0, 1.0]],  # y-displacements
    ], dtype=grid.dtype, device=grid.device)

    # Check that the displacement field matches exactly
    assert torch.allclose(disp, expected_disp, atol=1e-6), f"Expected {expected_disp}, got {disp}"


def test_params_to_affine_scaling():
    """
    params_to_affine with scaling should produce correct scaling matrix.
    """
    scale_factors = (2.0, 3.0)

    result_affine = vxm.params_to_affine(
        ndim=2,
        scale=scale_factors
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_shearing_2d():
    """
    params_to_affine with shearing should produce correct shear matrix.
    """

    # Make an affine with shear
    shear_value = 0.5
    result_affine = vxm.params_to_affine(
        ndim=2,
        shear=shear_value
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_shearing_3d():
    """
    params_to_affine with 3D shearing should produce correct shear matrix.
    """
    shear_values = (0.5, 0.3, 0.7)

    result_affine = vxm.params_to_affine(
        ndim=3,
        shear=shear_values
    ).to(torch.float64)

    expected_affine = torch.tensor([
        [1.0, 0.5, 0.3, 0.0],
        [0.0, 1.0, 0.7, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    assert torch.allclose(result_affine, expected_affine, atol=1e-5)


def test_params_to_affine_complex_2d():
    """
    Test complex composed affine with known expected result.
    """
    translation = (2.0, 3.0)
    rotation = 45.0  # degrees
    scale = (1.5, 2.0)
    shear = 0.3

    result_affine = vxm.params_to_affine(
        ndim=2,
        translation=translation,
        rotation=rotation,
        scale=scale,
        shear=shear
    ).to(torch.float64)

    # Expected result calculated by hand: T @ R @ Z @ S
    # Where T=translation, R=rotation, Z=scale, S=shear
    expected_affine = torch.tensor([
        [1.0607, -1.0960, 2.0000],
        [1.0607, 1.7324, 3.0000],
        [0.0000, 0.0000, 1.0000]
    ], dtype=torch.float64)

    # Check that the result matches the expected matrix exactly
    assert torch.allclose(result_affine, expected_affine, atol=1e-4), \
        f"Expected {expected_affine}, got {result_affine}"


def test_disp_to_coords_zero_disp():
    """
    disp_to_coords with zero displacement should produce normalized grid coordinates.
    Output maintains (ndim, *spatial) format.
    """
    # Create a zero displacement field - (ndim, *spatial) = (2, 2, 2)
    disp = torch.zeros(2, 2, 2, dtype=torch.float32)
    coords = vxm.disp_to_coords(disp)

    # Output maintains (ndim, *spatial) = (2, 2, 2)
    assert coords.shape == (2, 2, 2)

    # With zero displacement, we should get the normalized grid coordinates
    # coords[0] = row indices normalized to [-1, 1]
    # coords[1] = col indices normalized to [-1, 1]
    expected_row = torch.tensor([
        [-1.0, -1.0],
        [1.0, 1.0]
    ], dtype=torch.float32)
    expected_col = torch.tensor([
        [-1.0, 1.0],
        [-1.0, 1.0]
    ], dtype=torch.float32)

    assert torch.allclose(coords[0], expected_row, atol=1e-6), (
        f"Expected row coords {expected_row}, got {coords[0]}"
    )
    assert torch.allclose(coords[1], expected_col, atol=1e-6), (
        f"Expected col coords {expected_col}, got {coords[1]}"
    )


def test_integrate_disp_zero_steps():
    """
    integrate_disp with zero steps should return the original displacement.
    """
    # Displacement is now (ndim, *spatial) = (2, 2, 3)
    disp = torch.randn(2, 2, 3, dtype=torch.float32)
    integrated = vxm.integrate_disp(disp, steps=0)

    assert torch.allclose(integrated, disp)


def test_integrate_disp_single_step():
    """
    integrate_disp with one step should apply spatial transform once.
    """
    # Displacement is now (ndim, *spatial) = (2, 2, 3)
    disp = torch.randn(2, 2, 3, dtype=torch.float32)
    integrated = vxm.integrate_disp(disp, steps=1)

    # Should have same shape
    assert integrated.shape == disp.shape
    # Should be different from original (unless disp is very small)
    assert not torch.allclose(integrated, disp, atol=1e-6)


@pytest.mark.parametrize('module', (vxm, vxf))
def test_random_disp_matches_random_field(module):
    """random_disp should preserve the exact behavior of random_field on each public surface."""
    kwargs = {
        'shape': (1, 1, 8, 8),
        'scales': 2,
        'magnitude': 1,
        'integrations': 1,
        'non_spatial_dims': (0, 1),
    }

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        field = module.random_field(**kwargs)
        torch.manual_seed(42)
        disp = module.random_disp(**kwargs)

    assert torch.equal(disp, field)
    assert field.shape == (1, 2, 8, 8)


def test_random_transform():
    """
    random_transform should generate valid transforms.

    Note: vxf.random_transform expects (B, C, *spatial) format and returns
    (B, ndim, *spatial) displacement fields - channels-first format.
    """
    # Shape in (B, C, H, W) format
    shape = (1, 1, 3, 3)
    spatial_shape = shape[2:]  # (3, 3)
    ndim = len(spatial_shape)  # 2

    # Test affine-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=1.0,
        warp_probability=0.0
    )

    assert trf is not None
    # Output shape is (B, ndim, *spatial) - channels-first
    assert trf.shape == (shape[0], ndim) + spatial_shape

    # Test warp-only transform
    trf = vxf.random_transform(
        shape=shape,
        affine_probability=0.0,
        warp_probability=1.0
    )

    assert trf is not None
    assert trf.shape == (shape[0], ndim) + spatial_shape


def test_affine_to_disp_large_translation():
    """
    affine_to_disp should handle large translations correctly.
    """
    shape = (3, 3)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    ndim = len(shape)

    # Large translation
    large_translation = 100.0
    affine = torch.eye(ndim + 1, dtype=grid.dtype, device=grid.device)
    affine[0, -1] = large_translation
    affine[1, -1] = large_translation

    disp = vxm.affine_to_disp(affine, grid)

    # Check shape - now (ndim, *spatial)
    assert disp.shape == (ndim,) + shape
    # All displacements should be the translation value
    expected_disp = torch.full(
        (ndim,) + shape,
        large_translation,
        dtype=grid.dtype,
        device=grid.device
    )
    assert torch.allclose(disp, expected_disp)


def test_affine_to_disp_origin_at_center_scaling():
    """
    Test that origin_at_center controls the fixed point during scaling.

    With origin_at_center=True: center point should have zero displacement
    With origin_at_center=False: corner (0,0) should have zero displacement
    """
    shape = (3, 3)
    ndim = 2

    # 2x scaling affine
    scale_factor = 2.0
    affine = torch.eye(ndim, ndim + 1)
    affine[0, 0] = scale_factor
    affine[1, 1] = scale_factor

    # Test with origin_at_center=True (scale around center)
    disp_centered = vxm.affine_to_disp(affine, shape=shape, origin_at_center=True)

    # Center point (1,1) should have zero displacement
    # With (ndim, *spatial) format, access as disp[:, 1, 1] to get displacement vector
    assert torch.allclose(disp_centered[:, 1, 1], torch.zeros(2), atol=1e-6), \
        f"Center should be fixed, got displacement {disp_centered[:, 1, 1]}"

    # Test with origin_at_center=False (scale around corner)
    disp_corner = vxm.affine_to_disp(affine, shape=shape, origin_at_center=False)

    # Corner (0,0) should have zero displacement
    assert torch.allclose(disp_corner[:, 0, 0], torch.zeros(2), atol=1e-6), \
        f"Corner should be fixed, got displacement {disp_corner[:, 0, 0]}"

    # The two displacement fields should be different
    assert not torch.allclose(disp_centered, disp_corner), \
        "Displacement fields should differ based on origin_at_center"


def test_affine_to_disp_origin_at_center_rotation():
    """
    Test that origin_at_center controls the fixed point during rotation.

    90-degree rotation makes the difference very obvious.
    With origin_at_center=True: center stays fixed
    With origin_at_center=False: corner (0,0) stays fixed
    """
    shape = (3, 3)

    # 90-degree counter-clockwise rotation
    affine = torch.tensor([[0., -1., 0.],
                           [1., 0., 0.]])

    # Test with origin_at_center=True (rotate around center)
    disp_centered = vxm.affine_to_disp(affine, shape=shape, origin_at_center=True)

    # Center point (1,1) should have zero displacement (stays in place)
    # With (ndim, *spatial) format, access as disp[:, 1, 1] to get displacement vector
    assert torch.allclose(disp_centered[:, 1, 1], torch.zeros(2), atol=1e-6), \
        f"Center should be fixed during rotation, got {disp_centered[:, 1, 1]}"

    # Test with origin_at_center=False (rotate around corner)
    disp_corner = vxm.affine_to_disp(affine, shape=shape, origin_at_center=False)

    # Corner (0,0) should have zero displacement
    assert torch.allclose(disp_corner[:, 0, 0], torch.zeros(2), atol=1e-6), \
        f"Corner should be fixed during rotation, got {disp_corner[:, 0, 0]}"

    # The two should be dramatically different for rotation
    assert not torch.allclose(disp_centered, disp_corner), \
        "Rotation around center vs corner should produce very different results"


def test_params_to_affine_analytical():
    """
    Test compose() validates T(x) = A(B(x)) behavior with non-commuting transforms.

    Uses rotation and translation which don't commute to clearly test order.
    For compose([A, B]), should apply B first, then A, giving matrix A @ B.
    """
    # A: 90-degree counter-clockwise rotation
    A = torch.tensor([[0., -1., 0.],
                      [1., 0., 0.]])

    # B: translation by (2, 0)
    B = torch.tensor([[1., 0., 2.],
                      [0., 1., 0.]])

    # Compose A and B
    composed = vxm.compose([A, B])

    # Expected: A @ B (apply B first, then A)
    expected_affine = torch.tensor([[0., -1., 0.],
                                    [1., 0., 2.]])

    assert torch.allclose(composed, expected_affine, atol=1e-6), \
        f"Expected {expected_affine}, got {composed}"

    # Verify this is NOT B @ A (opposite order)
    wrong_order = torch.tensor([[0., -1., 2.],
                                [1., 0., 0.]])
    assert not torch.allclose(composed, wrong_order), \
        "compose([A, B]) should not be B @ A"

    # Test point transformation: apply B first (translate), then A (rotate)
    test_point = torch.tensor([1., 0., 1.])
    result = composed @ test_point
    expected_result = torch.tensor([0., 3.])

    assert torch.allclose(result, expected_result, atol=1e-6), \
        f"Point transformation failed: expected {expected_result}, got {result}"


def test_disp_to_coords_with_displacement():
    """
    Test that disp_to_coords correctly adds displacement to meshgrid then normalizes.
    Output maintains (ndim, *spatial) format.
    """
    # Create a simple displacement field - (ndim, H, W) = (2, 2, 2)
    disp = torch.zeros(2, 2, 2, dtype=torch.float32)

    # Add some displacement to specific locations
    disp[0, 0, 0] = 1.0  # row-displacement at (0, 0)
    disp[1, 1, 1] = 2.0  # col-displacement at (1, 1)

    # Convert to coordinates
    coords = vxm.disp_to_coords(disp)

    # Output maintains (ndim, *spatial) = (2, 2, 2)
    assert coords.shape == (2, 2, 2)

    # Calculation: coords = (meshgrid + disp) * 2 / (size - 1) - 1
    # For 2x2 grid: formula is (index + disp) * 2 - 1
    # meshgrid[0, 0, 0] = 0, disp[0, 0, 0] = 1 -> (0 + 1) * 2 - 1 = 1
    # meshgrid[1, 1, 1] = 1, disp[1, 1, 1] = 2 -> (1 + 2) * 2 - 1 = 5
    assert torch.isclose(coords[0, 0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(coords[1, 1, 1], torch.tensor(5.0), atol=1e-6)


def test_random_affine_shape_2d():
    """
    random_affine should return valid 2D affine matrix with correct shape and structure.
    """
    affine = vxm.random_affine(
        ndim=2,
        max_translation=10.0,
        max_rotation=15.0,
        max_scaling=1.2
    )

    # Check shape: (ndim+1, ndim+1) = (3, 3)
    assert affine.shape == (3, 3)
    assert affine.dtype == torch.float32

    # Affine should be invertible (non-zero determinant)
    det = torch.linalg.det(affine)
    assert det.abs() > 1e-6


def test_random_affine_shape_3d():
    """
    random_affine should return valid 3D affine matrix with correct shape and structure.
    """
    affine = vxm.random_affine(
        ndim=3,
        max_translation=5.0,
        max_rotation=10.0,
        max_scaling=1.1
    )

    # Check shape: (ndim+1, ndim+1) = (4, 4)
    assert affine.shape == (4, 4)
    assert affine.dtype == torch.float32

    # Affine should be invertible (non-zero determinant)
    det = torch.linalg.det(affine)
    assert det.abs() > 1e-6


def test_random_affine_deterministic_translation():
    """
    random_affine with sampling=False should use max values directly.

    With sampling=False:
    - translation = [max_translation] * ndim
    - rotation = [max_rotation] * ndim (1 for 2D, 3 for 3D)
    - scale = [max_scaling] * ndim
    """
    max_translation = 5.0
    max_rotation = 0.0  # No rotation for simpler verification
    max_scaling = 1.0   # No scaling for simpler verification

    affine = vxm.random_affine(
        ndim=2,
        max_translation=max_translation,
        max_rotation=max_rotation,
        max_scaling=max_scaling,
        sampling=False
    )

    # With no rotation and no scaling, affine should be identity + translation
    expected_translation = torch.tensor([max_translation, max_translation], dtype=torch.float32)
    assert torch.allclose(affine[:2, -1], expected_translation)

    # Linear part should be identity (no rotation, no scaling)
    expected_linear = torch.eye(2, dtype=torch.float32)
    assert torch.allclose(affine[:2, :2], expected_linear)


def test_compose_two_affines():
    """
    Composing two affine matrices should return an affine via matrix multiplication.

    For transforms [A, B], compose returns A @ B (B applied first, then A).
    """
    # Translation by (10, 5)
    translate = torch.tensor([
        [1., 0., 10.],
        [0., 1., 5.]
    ])

    # Scale by 2x
    scale = torch.tensor([
        [2., 0., 0.],
        [0., 2., 0.]
    ])

    # Compose: scale first, then translate
    composed = vxm.compose([translate, scale])

    # Result should be affine shape (2, 3)
    assert composed.shape == (2, 3)
    assert vxm.is_affine_shape(composed.shape)

    # Manual computation: translate @ scale (after making both square)
    # scale maps x -> 2x, then translate maps 2x -> 2x + t
    # So composed should be [[2, 0, 10], [0, 2, 5]]
    expected = torch.tensor([
        [2., 0., 10.],
        [0., 2., 5.]
    ])
    assert torch.allclose(composed, expected)


def test_compose_two_constant_displacements():
    """
    Composing two constant displacement fields should sum them in the interior.

    For compose([disp1, disp2]), the math is:
        composed(x) = disp2(x) + disp1(x + disp2(x))

    When both displacements are spatially constant, disp1(x + disp2(x)) = disp1(x),
    so composed(x) = disp1(x) + disp2(x) (simple addition).

    Note: Boundary pixels may differ due to grid_sample padding_mode='zeros'.
    We test the interior region where sampling stays within bounds.
    """
    shape = (16, 16)
    ndim = len(shape)

    # Constant displacement: shift by (1, 0) everywhere
    # Now (ndim, *spatial) = (2, 16, 16)
    disp1 = torch.zeros(ndim, *shape)
    disp1[0, ...] = 1.0  # x-displacement

    # Constant displacement: shift by (0, 1) everywhere
    disp2 = torch.zeros(ndim, *shape)
    disp2[1, ...] = 1.0  # y-displacement

    composed = vxm.compose([disp1, disp2])

    # Expected: (1, 1) everywhere in the interior
    expected = torch.zeros(ndim, *shape)
    expected[0, ...] = 1.0
    expected[1, ...] = 1.0

    assert composed.shape == (ndim, *shape)

    # Check interior region (exclude boundary pixels affected by padding)
    interior = composed[:, 2:-2, 2:-2]
    expected_interior = expected[:, 2:-2, 2:-2]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_compose_translation_affine_with_displacement():
    """
    Composing [translation_affine, displacement] adds translation to displacement.

    For compose([affine, disp]):
        composed(x) = disp(x) + affine_disp(x + disp(x))

    With a pure translation affine (constant displacement), the affine contribution
    is constant everywhere, so the result is disp + translation.
    """
    shape = (8, 8)
    ndim = len(shape)

    # Translation affine: shift by (5, 3)
    translation = torch.tensor([
        [1., 0., 5.],
        [0., 1., 3.]
    ])

    # Constant displacement field: shift by (1, 2)
    # Now (ndim, *spatial) = (2, 8, 8)
    disp = torch.zeros(ndim, *shape)
    disp[0, ...] = 1.0  # x-displacement
    disp[1, ...] = 2.0  # y-displacement

    # Compose: disp first, then translation
    composed = vxm.compose([translation, disp])

    # Expected: (1+5, 2+3) = (6, 5) everywhere
    expected = torch.zeros(ndim, *shape)
    expected[0, ...] = 6.0
    expected[1, ...] = 5.0

    assert composed.shape == (ndim, *shape)
    assert torch.allclose(composed, expected, atol=1e-5)


def test_compose_displacement_with_translation_affine():
    """
    Composing [displacement, translation_affine] adds translation to displacement.

    For compose([disp, affine]):
        composed(x) = affine_disp(x) + disp(affine(x))

    With constant displacement and pure translation:
        composed(x) = translation + disp (since disp is constant, disp(affine(x)) = disp)

    Note: Boundary pixels may differ due to grid_sample padding_mode='zeros'.
    We test the interior region where sampling stays within bounds.
    """
    shape = (16, 16)
    ndim = len(shape)

    # Constant displacement field: shift by (1, 1)
    # Now (ndim, *spatial) = (2, 16, 16)
    disp = torch.zeros(ndim, *shape)
    disp[0, ...] = 1.0  # x-displacement
    disp[1, ...] = 1.0  # y-displacement

    # Translation affine: shift by (2, 2)
    translation = torch.tensor([
        [1., 0., 2.],
        [0., 1., 2.]
    ])

    # Compose: affine first, then disp
    composed = vxm.compose([disp, translation])

    # Expected: (2+1, 2+1) = (3, 3) everywhere in the interior
    expected = torch.zeros(ndim, *shape)
    expected[0, ...] = 3.0
    expected[1, ...] = 3.0

    assert composed.shape == (ndim, *shape)

    # Check interior region (exclude boundary pixels affected by padding)
    interior = composed[:, 4:-4, 4:-4]
    expected_interior = expected[:, 4:-4, 4:-4]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_compose_batched_displacements():
    """
    Composing batched displacement fields (B, ndim, *spatial) should work correctly.

    This tests the automatic batch dimension detection in compose().
    """
    batch_size = 4
    shape = (16, 16)
    ndim = len(shape)

    # Batched constant displacement: shift by (1, 0) everywhere
    disp1 = torch.zeros(batch_size, ndim, *shape)
    disp1[:, 0, ...] = 1.0  # x-displacement

    # Batched constant displacement: shift by (0, 1) everywhere
    disp2 = torch.zeros(batch_size, ndim, *shape)
    disp2[:, 1, ...] = 1.0  # y-displacement

    composed = vxm.compose([disp1, disp2])

    # Output should be batched with same shape
    assert composed.shape == (batch_size, ndim, *shape)

    # Expected: (1, 1) everywhere in the interior (constant displacements add)
    expected = torch.zeros(batch_size, ndim, *shape)
    expected[:, 0, ...] = 1.0
    expected[:, 1, ...] = 1.0

    # Check interior region (exclude boundary pixels affected by padding)
    interior = composed[:, :, 2:-2, 2:-2]
    expected_interior = expected[:, :, 2:-2, 2:-2]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_compose_scale_affine_with_zero_displacement():
    """
    Composing [scale_affine, zero_disp] should produce the affine's displacement field.

    For compose([affine, disp]) with disp=0:
        composed(x) = 0 + affine_disp(x) = affine_disp(x)

    A 2x scale centered at origin maps x -> 2x, so displacement is x (moves each
    point away from center by its distance from center).
    """
    shape = (5, 5)
    ndim = len(shape)

    # Scale by 2x (centered at image center due to origin_at_center=True)
    scale_affine = torch.tensor([
        [2., 0., 0.],
        [0., 2., 0.]
    ])

    # Zero displacement - now (ndim, *spatial)
    disp = torch.zeros(ndim, *shape)

    composed = vxm.compose([scale_affine, disp])

    # With origin_at_center=True and shape (5,5), center is at (2, 2)
    # Scale 2x maps: x_centered -> 2 * x_centered
    # Displacement = new_pos - old_pos = 2*x_centered - x_centered = x_centered
    # At corners: displacement equals distance from center
    # Grid is now (ndim, *spatial)
    grid = ne.volshape_to_ndgrid(size=shape, stack=True)
    center = torch.tensor([(s - 1) / 2 for s in shape]).view(ndim, *([1] * ndim))
    expected = grid - center  # x_centered

    assert composed.shape == (ndim, *shape)
    assert torch.allclose(composed, expected, atol=1e-5)


@pytest.mark.parametrize("shape,expected", [
    # Valid 2d affine shapes
    ((2, 3), True),   # compact 2d: (N, N+1)
    ((3, 3), True),   # square 2d: (N+1, N+1)
    # Valid 3d affine shapes
    ((3, 4), True),   # compact 3d: (N, N+1)
    ((4, 4), True),   # square 3d: (N+1, N+1)
    # Valid with batch dimensions
    ((5, 2, 3), True),        # batched compact 2d
    ((2, 10, 4, 4), True),    # multi-batch square 3d
    # Invalid: wrong column count (cols must be 3 or 4 for N=2 or N=3)
    ((2, 2), False),   # cols=2 -> N=1, not supported
    ((3, 5), False),   # cols=5 -> N=4, not supported
    ((2, 4), False),   # cols=4 -> N=3, but rows=2 != 3 or 4
    # Invalid: wrong row count
    ((1, 3), False),   # cols=3 -> N=2, but rows=1 != 2 or 3
    ((5, 4), False),   # cols=4 -> N=3, but rows=5 != 3 or 4
    # Invalid: too few dimensions
    ((3,), False),
    ((), False),
])
def test_is_affine_shape(shape, expected):
    """is_affine_shape should correctly identify valid affine matrix shapes."""
    assert vxm.is_affine_shape(shape) == expected


def test_vxf_integrate_disp_zero_steps_2d():
    """
    vxf.integrate_disp with zero steps should return the original displacement.
    """
    disp = torch.randn(2, 2, 8, 8)  # (B, ndim, H, W)
    integrated = vxf.integrate_disp(disp, steps=0)

    assert integrated.shape == disp.shape
    assert torch.allclose(integrated, disp)


def test_vxf_integrate_disp_preserves_batch_2d():
    """
    vxf.integrate_disp should preserve batch dimension and process each sample independently.
    """
    batch_size = 3
    disp = torch.randn(batch_size, 2, 16, 16)  # (B, ndim, H, W)
    integrated = vxf.integrate_disp(disp, steps=5)

    assert integrated.shape == (batch_size, 2, 16, 16)

    # Verify each batch element matches independent integration
    for i in range(batch_size):
        single_integrated = vxm.integrate_disp(disp[i], steps=5)
        assert torch.allclose(integrated[i], single_integrated, atol=1e-6)


def test_vxf_integrate_disp_3d():
    """
    vxf.integrate_disp should work with 3d spatial data.
    """
    disp = torch.randn(2, 3, 8, 8, 8)  # (B, ndim, D, H, W)
    integrated = vxf.integrate_disp(disp, steps=3)

    assert integrated.shape == (2, 3, 8, 8, 8)


def test_vxf_disp_to_coords_zero_disp_2d():
    """
    vxf.disp_to_coords with zero displacement should produce normalized grid coordinates.
    """
    disp = torch.zeros(2, 2, 3, 3)  # (B, ndim, H, W)
    coords = vxf.disp_to_coords(disp)

    assert coords.shape == (2, 2, 3, 3)

    # With zero displacement, coords should be normalized meshgrid in [-1, 1]
    # For 3x3 grid: [-1, 0, 1] along each axis
    expected_row = torch.tensor([
        [-1., -1., -1.],
        [0., 0., 0.],
        [1., 1., 1.],
    ])
    expected_col = torch.tensor([
        [-1., 0., 1.],
        [-1., 0., 1.],
        [-1., 0., 1.],
    ])

    for b in range(2):
        assert torch.allclose(coords[b, 0], expected_row, atol=1e-6)
        assert torch.allclose(coords[b, 1], expected_col, atol=1e-6)


def test_vxf_disp_to_coords_preserves_batch_2d():
    """
    vxf.disp_to_coords should preserve batch dimension and match independent calls.
    """
    batch_size = 3
    disp = torch.randn(batch_size, 2, 8, 8)  # (B, ndim, H, W)
    coords = vxf.disp_to_coords(disp)

    assert coords.shape == (batch_size, 2, 8, 8)

    # Verify each batch element matches independent conversion
    for i in range(batch_size):
        single_coords = vxm.disp_to_coords(disp[i])
        assert torch.allclose(coords[i], single_coords, atol=1e-6)


def test_vxf_disp_to_coords_3d():
    """
    vxf.disp_to_coords should work with 3d spatial data.
    """
    disp = torch.randn(2, 3, 8, 8, 8)  # (B, ndim, D, H, W)
    coords = vxf.disp_to_coords(disp)

    assert coords.shape == (2, 3, 8, 8, 8)


def test_vxf_coords_to_disp_not_implemented():
    """
    vxf.coords_to_disp should raise NotImplementedError until underlying function is implemented.
    """
    coords = torch.randn(2, 8, 8, 2)  # (B, *spatial, ndim)

    with pytest.raises(NotImplementedError):
        vxf.coords_to_disp(coords)


def test_vxf_compose_two_batched_displacements():
    """
    vxf.compose should compose two batched displacement fields.
    """
    batch_size = 2
    shape = (16, 16)
    ndim = len(shape)

    # Constant displacement: shift by (1, 0)
    disp1 = torch.zeros(batch_size, ndim, *shape)
    disp1[:, 0, ...] = 1.0

    # Constant displacement: shift by (0, 1)
    disp2 = torch.zeros(batch_size, ndim, *shape)
    disp2[:, 1, ...] = 1.0

    composed = vxf.compose([disp1, disp2])

    assert composed.shape == (batch_size, ndim, *shape)

    # Interior should sum to (1, 1)
    expected = torch.zeros(batch_size, ndim, *shape)
    expected[:, 0, ...] = 1.0
    expected[:, 1, ...] = 1.0

    interior = composed[:, :, 2:-2, 2:-2]
    expected_interior = expected[:, :, 2:-2, 2:-2]
    assert torch.allclose(interior, expected_interior, atol=1e-5)


def test_vxf_compose_matches_vxm():
    """
    vxf.compose should produce same results as vxm.compose for batched inputs.
    """
    batch_size = 2
    disp1 = torch.randn(batch_size, 2, 16, 16)
    disp2 = torch.randn(batch_size, 2, 16, 16)

    vxf_result = vxf.compose([disp1, disp2])
    vxm_result = vxm.compose([disp1, disp2])

    assert torch.allclose(vxf_result, vxm_result, atol=1e-6)


def test_vxf_compose_3d():
    """
    vxf.compose should work with 3d spatial data.
    """
    batch_size = 2
    disp1 = torch.randn(batch_size, 3, 8, 8, 8)  # (B, ndim, D, H, W)
    disp2 = torch.randn(batch_size, 3, 8, 8, 8)

    composed = vxf.compose([disp1, disp2])

    assert composed.shape == (batch_size, 3, 8, 8, 8)


def test_spatial_transform_non_square_single_pixel_2d():
    """
    Verify coordinate normalization is correct for non-square shapes by tracking a single pixel.

    This test catches bugs where spatial dimensions are swapped during normalization.

    Note: spatial_transform uses backward warping semantics:
        output[y, x] = input[y + disp[0], x + disp[1]]
    So positive displacement causes content to shift in the OPPOSITE direction visually.
    With disp[0] = +2, content shifts UP (output[0,8] samples from input[2,8]).
    """
    # Asymmetric: H=8, W=16
    H, W = 8, 16
    image = torch.zeros(1, H, W, dtype=torch.float32)
    image[0, 4, 8] = 1.0  # single bright pixel at row=4, col=8

    # Displacement that samples from 2 rows down (content shifts UP visually)
    # disp[0] is row displacement, disp[1] is col displacement
    disp = torch.zeros(2, H, W, dtype=torch.float32)
    disp[0, :, :] = 2.0  # sample from y+2
    disp[1, :, :] = 0.0  # no shift in col direction

    warped = vxm.spatial_transform(image, disp, non_spatial_dims=(0,))

    # With backward warping: output[2, 8] = input[2+2, 8] = input[4, 8] = 1.0
    # The pixel visually moves from (4, 8) to (2, 8)
    assert warped.shape == image.shape
    assert warped[0, 2, 8] > 0.5, f"Pixel should be at (2, 8), got max at {warped[0].argmax()}"
    assert warped[0, 4, 8] < 0.1, "Original position should be empty"
    # Verify column didn't change (would happen if normalization is wrong)
    assert warped[0, 2, 10] < 0.1, "Pixel should not have moved horizontally"


def test_spatial_transform_non_square_asymmetric_shift_2d():
    """
    Test asymmetric shifts on non-square image to verify each axis is normalized independently.

    Uses different shift amounts in each direction on a highly asymmetric image.

    Note: spatial_transform uses backward warping semantics:
        output[y, x] = input[y + disp[0], x + disp[1]]
    So positive displacement causes content to shift in the OPPOSITE direction visually.
    """
    # Very asymmetric: H=10, W=40
    H, W = 10, 40
    image = torch.zeros(1, H, W, dtype=torch.float32)
    image[0, 4, 24] = 1.0  # bright pixel at (4, 24)

    # Displacement: sample from (y+1, x+4)
    # Content visually shifts UP by 1 and LEFT by 4
    disp = torch.zeros(2, H, W, dtype=torch.float32)
    disp[0, :, :] = 1.0  # sample from y+1
    disp[1, :, :] = 4.0  # sample from x+4

    warped = vxm.spatial_transform(image, disp, non_spatial_dims=(0,))

    # With backward warping: output[3, 20] = input[3+1, 20+4] = input[4, 24] = 1.0
    # Pixel visually moves from (4, 24) to (3, 20)
    assert warped.shape == image.shape
    assert warped[0, 3, 20] > 0.5, f"Pixel should be at (3, 20)"
    assert warped[0, 4, 24] < 0.1, "Original position should be empty"


def _measure_shift_magnitude(H: int, W: int, shift_dim: int, shift_amount: float) -> float:
    """
    Helper to measure actual shift magnitude on a non-square image.

    Creates a gradient image in the shift direction, applies a constant displacement,
    and measures the actual shift by comparing values at the center pixel.

    Parameters
    ----------
    H : int
        Image height.
    W : int
        Image width.
    shift_dim : int
        Dimension to shift (0 for y/rows, 1 for x/cols).
    shift_amount : float
        Displacement in pixels.

    Returns
    -------
    float
        Measured shift magnitude (should equal shift_amount if implementation is correct).
    """
    # Create image with linear gradient in the shift direction
    image = torch.zeros(1, 1, H, W)
    if shift_dim == 0:  # y-gradient
        image[0, 0] = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    else:  # x-gradient
        image[0, 0] = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)

    # Create displacement field with shift in specified dimension
    disp = torch.zeros(1, 2, H, W)
    disp[0, shift_dim, :, :] = shift_amount

    # Apply transform with border padding to avoid edge effects
    output = vxm.spatial_transform(
        image=image,
        trf=disp,
        mode='linear',
        isdisp=True,
        non_spatial_dims=(0, 1),
        align_corners=True,
        padding_mode='border'
    )

    # Measure shift at center pixel
    center_y, center_x = H // 2, W // 2
    input_val = image[0, 0, center_y, center_x].item()
    output_val = output[0, 0, center_y, center_x].item()

    return output_val - input_val


@pytest.mark.parametrize("shift_dim,shift_amount", [
    (0, 1.0),   # y-shift by 1 pixel
    (0, 5.0),   # y-shift by 5 pixels
    (1, 1.0),   # x-shift by 1 pixel
    (1, 5.0),   # x-shift by 5 pixels
])
def test_spatial_transform_non_square_shift_magnitude_8to1(shift_dim, shift_amount):
    """
    Verify exact shift magnitude on 8:1 aspect ratio image (H=200, W=25).

    Extreme aspect ratio makes normalization bugs very obvious.
    A buggy implementation would scale shifts by 8x or 0.125x.
    """
    H, W = 200, 25
    actual_shift = _measure_shift_magnitude(H, W, shift_dim, shift_amount)

    assert abs(actual_shift - shift_amount) < 0.01, (
        f"Shift in dim {shift_dim} by {shift_amount}: expected {shift_amount}, "
        f"got {actual_shift:.4f} (error: {abs(actual_shift - shift_amount):.4f})"
    )


@pytest.mark.parametrize("shift_dim,shift_amount", [
    (0, 1.0),   # y-shift by 1 pixel
    (0, 5.0),   # y-shift by 5 pixels
    (1, 1.0),   # x-shift by 1 pixel
    (1, 5.0),   # x-shift by 5 pixels
])
def test_spatial_transform_non_square_shift_magnitude_1to8(shift_dim, shift_amount):
    """
    Verify exact shift magnitude on 1:8 aspect ratio image (H=25, W=200).

    Tests the opposite extreme from 8:1 to ensure both tall and wide images work.
    """
    H, W = 25, 200
    actual_shift = _measure_shift_magnitude(H, W, shift_dim, shift_amount)

    assert abs(actual_shift - shift_amount) < 0.01, (
        f"Shift in dim {shift_dim} by {shift_amount}: expected {shift_amount}, "
        f"got {actual_shift:.4f} (error: {abs(actual_shift - shift_amount):.4f})"
    )


@pytest.mark.parametrize("shape,non_spatial_dims", [
    ((2, 32, 32), None),           # Unbatched 2D
    ((3, 16, 16, 16), None),       # Unbatched 3D
    ((4, 2, 32, 32), (0,)),        # Batched 2D
    ((4, 3, 16, 16, 16), (0,)),    # Batched 3D
])
def test_disp_trf_round_trip(shape, non_spatial_dims):
    """Test that disp -> trf -> disp round-trip preserves values."""
    torch.manual_seed(42)
    original = torch.randn(shape)
    trf = vxm.disp_to_trf(original, non_spatial_dims=non_spatial_dims)
    recovered = vxm.trf_to_disp(trf, non_spatial_dims=non_spatial_dims)
    assert torch.allclose(original, recovered, atol=1e-5)
