"""
Check integration outputs, gradients, and checkpoints against the original module calculation.
"""

from copy import deepcopy
from typing import Tuple
from unittest.mock import Mock, patch

import pytest
import torch

import voxelmorph as vxm
import voxelmorph.nn.functional as vxf
from voxelmorph.nn.modules import IntegrateVelocityField


def original_forward(module: IntegrateVelocityField, velocity_field: torch.Tensor) -> torch.Tensor:
    """
    Preserve the original module calculation as an independent compatibility reference.
    """
    velocity_field = velocity_field * module.scale
    for _ in range(module.steps):
        velocity_field = velocity_field + module.transformer(velocity_field, velocity_field)
    return velocity_field


@pytest.mark.parametrize("shape", [(2, 2, 7, 9), (2, 3, 5, 7, 9)])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_outputs_and_gradients(
    shape: Tuple[int, ...], steps: int, mode: str, align_corners: bool, dtype: torch.dtype
) -> None:
    """
    Require exact CPU agreement with the original module for outputs and input gradients.
    """
    generator = torch.Generator().manual_seed(18)
    velocity = (torch.randn(shape, generator=generator, dtype=dtype) * 0.3).requires_grad_()
    reference_velocity = velocity.detach().clone().requires_grad_()
    module = IntegrateVelocityField(
        steps=steps, interpolation_mode=mode, align_corners=align_corners
    )
    reference_module = deepcopy(module)

    expected = original_forward(reference_module, reference_velocity)
    actual = module(velocity)
    weights = torch.randn(shape, generator=generator, dtype=dtype)
    expected_gradient = torch.autograd.grad(expected, reference_velocity, weights)[0]
    actual_gradient = torch.autograd.grad(actual, velocity, weights)[0]

    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, expected_gradient)
    assert actual is not velocity
    assert module.state_dict().keys() == reference_module.state_dict().keys()


@pytest.mark.parametrize("steps", [0, 3])
def test_module_attributes_and_grid_reuse(steps: int) -> None:
    """
    Preserve stored scale, transformer settings, grid reuse, and shape-triggered grid replacement.
    """
    module = IntegrateVelocityField(steps=steps)
    module.scale = 0.25
    module.transformer.interpolation_mode = "nearest"
    module.transformer.align_corners = False
    reference = deepcopy(module)
    velocity = torch.randn(2, 2, 7, 9)

    assert torch.equal(module(velocity), original_forward(reference, velocity))
    if steps == 0:
        assert not hasattr(module.transformer, "meshgrid")
        return

    grid = module.transformer.meshgrid
    module(velocity)
    assert module.transformer.meshgrid is grid
    changed_shape = torch.randn(2, 2, 9, 11)
    assert torch.equal(module(changed_shape), original_forward(reference, changed_shape))
    assert module.transformer.meshgrid is not grid
    assert module.transformer.meshgrid.shape == (2, 9, 11)


@pytest.mark.parametrize("steps", [0, 3])
def test_module_calls_function(steps: int) -> None:
    """
    Ensure the module delegates integration while preserving its public child-module structure.
    """
    module = IntegrateVelocityField(steps=steps)
    velocity = torch.randn(2, 2, 7, 9)
    with patch.object(vxf, "integrate_vec", wraps=vxf.integrate_vec) as integrate:
        module(velocity)
    integrate.assert_called_once()
    assert dict(module.named_children()).keys() == {"transformer"}
    assert module.state_dict() == {}


def scale_warp_output(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """
    Change a warp through a forward hook to detect bypassed transformer calls.
    """
    return output * 0.75


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_transformer_forward_and_hooks(steps: int) -> None:
    """
    Preserve transformer forward calls, hook effects, and gradients from the original loop.
    """
    module = IntegrateVelocityField(steps=steps)
    reference = deepcopy(module)
    hook = Mock(side_effect=scale_warp_output)
    module.transformer.register_forward_hook(hook)
    reference.transformer.register_forward_hook(scale_warp_output)
    velocity = torch.randn(2, 2, 7, 9, requires_grad=True)
    reference_velocity = velocity.detach().clone().requires_grad_()
    expected = original_forward(reference, reference_velocity)
    with patch.object(module.transformer, "forward", wraps=module.transformer.forward) as forward:
        actual = module(velocity)
    assert forward.call_count == steps
    assert hook.call_count == steps
    assert torch.equal(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), velocity)[0]
    reference_gradient = torch.autograd.grad(expected.sum(), reference_velocity)[0]
    assert torch.equal(gradient, reference_gradient)


def test_integration_keeps_batched_sampling() -> None:
    """
    Retain one batched spatial sampling call per integration step.
    """
    module = IntegrateVelocityField(steps=3)
    velocity = torch.randn(4, 2, 7, 9)
    with patch("torch.nn.functional.grid_sample", wraps=torch.nn.functional.grid_sample) as sample:
        module(velocity)
    assert sample.call_count == module.steps
    for call in sample.call_args_list:
        assert call.args[0].shape[0] == velocity.shape[0]


@pytest.mark.parametrize("ndim", [2, 3])
def test_model_checkpoint_outputs_and_gradients(ndim: int) -> None:
    """
    Load an old-format model state and preserve bidirectional outputs and parameter gradients.
    """
    torch.manual_seed(19)
    model = vxm.nn.models.VxmPairwise(
        ndim=ndim, source_channels=1, target_channels=1, integration_steps=3, device="cpu"
    )
    reference_model = deepcopy(model)
    with patch.object(IntegrateVelocityField, "forward", original_forward):
        checkpoint = deepcopy(reference_model.state_dict())
    model.load_state_dict(checkpoint, strict=True)
    shape = (1, 1) + (32,) * ndim
    source = torch.randn(shape)
    target = torch.randn(shape)
    options = dict(
        return_warped_source=True, return_warped_target=True, return_field_type="displacement"
    )

    with patch.object(IntegrateVelocityField, "forward", original_forward):
        expected = reference_model(source, target, **options)
        expected_loss = sum(output.square().mean() for output in expected)
        expected_loss.backward()
    actual = model(source, target, **options)
    actual_loss = sum(output.square().mean() for output in actual)
    actual_loss.backward()

    for output, reference_output in zip(actual, expected):
        assert torch.equal(output, reference_output)
    for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
        assert (parameter.grad is None) == (reference_parameter.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, reference_parameter.grad)
    assert model.state_dict().keys() == checkpoint.keys()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference_model.parameters(), lr=0.01).step()
    for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
        assert torch.equal(parameter, reference_parameter)
