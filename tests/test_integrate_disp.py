"""
Check backward compatibility of the deprecated velocity integration name.
"""

from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple
import warnings

import neurite as ne
import pytest
import torch

import voxelmorph as vxm
import voxelmorph.functional as functional
import voxelmorph.nn.functional as vxf


@pytest.mark.parametrize("module", [vxm, functional, vxf])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("keyword", [False, True])
def test_integrate_disp_defaults(module: ModuleType, steps: int, ndim: int, keyword: bool) -> None:
    """
    Preserve positional and legacy keyword calls, outputs, gradients, and caller warning location.
    """
    shape = (ndim,) + (5,) * ndim
    if module is vxf:
        shape = (2,) + shape
    velocity = (torch.randn(shape, dtype=torch.float64) * 0.1).requires_grad_()
    with pytest.warns(DeprecationWarning, match="Use integrate_vec") as captured:
        if keyword:
            actual = module.integrate_disp(disp=velocity, steps=steps)
        else:
            actual = module.integrate_disp(velocity, steps)
    assert len(captured) == 1
    assert Path(captured[0].filename) == Path(__file__)
    with warnings.catch_warnings(record=True) as current_warnings:
        warnings.simplefilter("always")
        expected = module.integrate_vec(velocity, steps)
    assert not current_warnings
    assert torch.equal(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), velocity)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), velocity)[0]
    assert torch.equal(actual_gradient, expected_gradient)
    if steps == 0:
        assert actual is velocity


@pytest.mark.parametrize("module", [vxm, functional, vxf])
@pytest.mark.parametrize("non_spatial_dims", [None, (0,)])
def test_integrate_disp_grid(module: ModuleType, non_spatial_dims: Optional[Tuple[int, ...]]) -> None:
    """
    Preserve explicit grid and batch settings passed through the original four-argument signature.
    """
    shape = (2, 5, 7) if non_spatial_dims is None else (2, 2, 5, 7)
    velocity = torch.randn(shape) * 0.1
    grid = ne.volshape_to_ndgrid(size=(5, 7), stack=True)
    with pytest.warns(DeprecationWarning, match="Use integrate_vec"):
        actual = module.integrate_disp(velocity, 3, grid, non_spatial_dims)
    expected = module.integrate_vec(velocity, 3, grid, non_spatial_dims)
    assert torch.equal(actual, expected)
