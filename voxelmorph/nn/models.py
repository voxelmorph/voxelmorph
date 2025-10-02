"""
Core VoxelMorph models for unsupervised and supervised learning.
"""

# Core library imports
from typing import List, Union, Callable, Tuple, Optional

# Third-party imports
import torch
import torch.nn as nn
import neurite as ne

# Local imports
import voxelmorph as vxm

__all__ = [
    "VxmPairwise",
]


class VxmPairwise(nn.Module):
    """
    A network architecture built on `BasicUNet` to perform nD image registration using a flow
    field.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
    source_channels : int
        Number of channels in the source image.
    target_channels : int
        Number of channels in the target image.
    out_channels : int
        Number of output channels in the displacement field.
    *args : list
        Additional positional arguments for the `BasicUNet` constructor.
    nb_features : List[int], optional
        List of integers specifying the number of features in each
        level of the UNet architecture. Default is `[16, 16, 16, 16, 16]`.
    normalizations : Union[List[str], str], optional
        Normalization layers for the UNet. Can be a list of normalization
        types or a single normalization type. Default is `None`.
    activations : Union[List[str], str], optional
        Activation functions for the UNet layers. Can be a list of
        activation functions or a single function. Default is `nn.ReLU`.
    order : str, optional
        The order of operations in each UNet block. Default is `'ncaca'`.
    final_activation : Union[str, nn.Module, None], optional
        The activation applied to the final output of the network. Default is `None`.
    flow_initializer : ne.random.Sampler, optional
        A custom sampler for initializing the weights of the flow layer.
        If not provided, it defaults to a normal distribution
        with mean 0 and standard deviation `1e-5`.
    integration_steps : int, optional
        Number of steps to take in integrating the flow field. Default is 1.
    **kwargs : dict
        Additional keyword arguments passed to the `BasicUNet` constructor.

    Attributes
    ----------
    flow_layer : nn.Module
        A custom convolutional block used to generate the flow field
        from the combined source and target features.

    Methods
    -------
    forward(source, target)
        Combines source and target images, processes them through the
        UNet and the flow layer, and returns the resulting flow field.
    """

    def __init__(
        self,
        ndim: int,
        source_channels: int,
        target_channels: int,
        nb_features: List[int] = (16, 16, 16, 16, 16),
        normalizations: Union[List[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[List[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = None,
        flow_initializer: Union[float, ne.samplers.Sampler] = ne.samplers.Normal(0, 1e-5),
        bidirectional_cost: bool = False,
        integration_steps: int = 0,
        device: str = "cpu",
    ):

        """
        Initialize the `VxmPairwise`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        source_channels : int
            Number of channels in the `source_tensor` input to the forward method of this class.
        target_channels : int
            Number of channels in the `target_tensor` input to the forward method of this class.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        normalizations : Union[List[str], str, None], optional
            Normalization layers to use in each block. Can be a string or a list
            of strings specifying normalizations for each layer, or `None` for no norm.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        order : str, optional
            The order of operations in each convolutional block. Default is 'cna'
            (normalization -> convolution -> activation). Each character in the string represents
            one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        bidirectional_cost : bool, optional
            Enable calculation of the cost-function bidirectionally. Default is False
        integration_steps : int, optional
            Number of scaling and squaring steps for integrating the flow field.
            Default is 0 (no integration).
        device : str, optional
            Device identifier (e.g., 'cpu' or 'cuda') to place/run the model on.
        """

        # Initialize the Module
        super().__init__()

        # Set constant attrs
        self.ndim = ndim
        self.integration_steps = integration_steps
        self.bidirectional_cost = bidirectional_cost
        self.device = device
        self.out_channels = ndim

        # Set derived attrs
        self._init_flow_layer(ndim, self.out_channels, flow_initializer)
        self.model = ne.nn.models.BasicUNet(
            ndim=ndim, in_channels=(source_channels + target_channels),
            out_channels=self.out_channels,
            nb_features=nb_features,
            normalizations=normalizations, activations=activations, order=order,
            final_activation=final_activation
        )

        # lazy-initialized, shape/device-bound ops (parameter-free)
        self.velocity_field_integrator: Optional[nn.Module] = None
        self.spatial_transformer: Optional[nn.Module] = None
        self._ops_shape: Optional[Tuple[int, ...]] = None
        self._ops_device: Optional[torch.device] = None

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        reg_field: str = 'warp',
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of `VxmPairwise`.

        This forward pass concatenates the `source` and `target` images, processes them with a
        `BasicUNet` backbone, and uses a flow layer to predict a velocity field (source -> target).

        By default, this method returns warped_source and pos_flow. If `reg_field='svf'`,
        it will return pos_vel instead of pos_flow. The pos_flow is obtained by integrating
        the pos_vel using scaling and squaring if `integration_steps > 0`; otherwise, the
        pos_vel is used directly as the pos_flow for warping.

        Parameters
        ----------
        source : torch.Tensor
            Source image tensor with batch and channel dimensions.
        target : torch.Tensor
            Target image tensor. Must have the same shape as `source`.
        reg_field : str, optional
            Which regularization field to return: 'svf' (velocity) or 'warp' (deformation).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            - warped_source: Source image warped by the pos_flow.
            - warped_target: Target image warped by the neg_flow (if bidirectional).
            - reg_field: The requested regularization field (velocity or deformation).
        """
        # Validate input shapes and devices
        if source.shape != target.shape:
            raise ValueError(f"Source and target must have the same shape, got {source.shape} and {target.shape}.")
        if source.device != target.device:
            raise ValueError(f"Source and target must be on the same device, got {source.device} and {target.device}.")

        # Pass combined features through the model's backbone & flow layer
        combined_features = torch.cat([source, target], dim=1)
        combined_features = self.model(combined_features)
        pos_vel = self.flow_layer(combined_features)

        neg_vel = -pos_vel if self.bidirectional_cost else None

        # Integrate velocity fields if needed
        if self.integration_steps > 0:
            pos_flow, neg_flow = self._integrate_velocity_fields(pos_vel, neg_vel)
        else:
            pos_flow = pos_vel
            neg_flow = neg_vel

        # Warp the source image with the deformation field
        warped_source = self._spatial_transform(source, pos_flow)
        warped_target = self._spatial_transform(target, neg_flow) if self.bidirectional_cost and neg_flow is not None else None

        # Prepare the outputs
        output_list = [warped_source]
        if self.bidirectional_cost and warped_target is not None:
            output_list.append(warped_target)

        reg_field = reg_field.lower()
        if reg_field == 'svf':
            output_list.append(pos_vel)
        elif reg_field == 'warp':
            output_list.append(pos_flow)

        return tuple(output_list)

    def _ensure_ops(self, ref_tensor: torch.Tensor) -> None:
        """
        Ensure VoxelMorph operators exist and match the current shape and device.

        This method **lazily constructs or re-constructs** the two parameter-free operators
        used by this model that depend on the *spatial size* of the data and the *compute device*:

        - ``velocity_field_integrator``: an instance of
        :class:`vxm.nn.modules.IntegrateVelocityField` used to integrate a stationary
        velocity field (SVF) into a displacement field via scaling-and-squaring.
        - ``spatial_transformer``: an instance of
        :class:`vxm.nn.modules.SpatialTransformer` used to warp an image with a
        displacement field (backed by ``grid_sample``).

        The method inspects ``ref_tensor`` to infer the required spatial shape and device,
        compares them with the cached values (``self._ops_shape`` and ``self._ops_device``),
        and if a mismatch is detected—or the operators have not yet been created—it
        (re)creates the two operators and registers them as submodules of this class.

        Parameters
        ----------
        ref_tensor : torch.Tensor
            A tensor whose trailing ``self.ndim`` dimensions define the spatial size
            required by the operators (e.g., ``(Z, Y, X)`` for 3D), and whose
            ``.device`` defines the target device on which the operators should live.
            Typical choices are the predicted flow (velocity) or the deformation field
            tensors passed into ``_integrate_velocity_fields`` and ``_spatial_transform``.

        Returns
        -------
        None
            This method updates internal state in-place by assigning
            ``self.velocity_field_integrator`` and ``self.spatial_transformer``, and by
            caching the active shape/device in ``self._ops_shape`` and ``self._ops_device``.
        """
        dev = ref_tensor.device
        spatial = tuple(ref_tensor.shape[-self.ndim:])

        need_new = (
            self.velocity_field_integrator is None
            or self.spatial_transformer is None
            or self._ops_shape != spatial
            or self._ops_device != dev
        )
        if not need_new:
            return

        # Create new instances bound to current shape/device
        vfi = vxm.nn.modules.IntegrateVelocityField(shape=spatial, steps=self.integration_steps, device=dev)
        stf = vxm.nn.modules.SpatialTransformer(size=spatial, device=dev)

        # Register/replace as submodules (they’re parameter-free so optimizer is unaffected)
        self.velocity_field_integrator = vfi
        self.spatial_transformer = stf
        self.add_module("velocity_field_integrator", self.velocity_field_integrator)
        self.add_module("spatial_transformer", self.spatial_transformer)

        self._ops_shape = spatial
        self._ops_device = dev

    def _init_flow_layer(
        self,
        ndim: int,
        features: int,
        flow_initializer: Union[float, ne.samplers.Sampler] = ne.samplers.Normal(0, 1e-5)
    ):
        """
        Initialize the flow layer with custom weight initialization (by sampling
        `flow_initializer`).

        This layer is a convolutional block that produces a displacement (flow)
        field. The weights of its initial convolution are sampled using the
        provided flow_initializer, and biases are set to zero.

        Parameters
        ----------
        ndim : int
            **Spatial** dimensionality of the input (1, 2, or 3).
        features : int
            Number of input and output features for the flow layer.
        flow_initializer :  Union[float, ne.random.Sampler], optional
            Sampler for initializing the *weights* of the flow layer. Default is
            `ne.random.Normal(0, 1e-5)`.
        """

        # Initialize the conv ("flow") layer with congruent in and out features
        flow_layer = ne.nn.modules.ConvBlock(ndim, features, features).to(self.device)

        # Optionally, apply custom initialization if `flow_initializer`` is provided
        if flow_initializer is not None:

            # Make the distribution to sample the flow parameters
            flow_initializer = ne.samplers.Fixed.make(flow_initializer)

            # Sample the weight parameters from the distribution for first (and only) conv
            with torch.no_grad():
                flow_layer.conv0.weight.copy_(
                    flow_initializer(flow_layer.conv0.weight.shape)
                    .to(flow_layer.conv0.weight.device)
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if flow_layer.conv0.bias is not None:
                    flow_layer.conv0.bias.zero_()
        # Register the flow layer as a submodule
        self.add_module("flow_layer", flow_layer)

    def _integrate_velocity_fields(
        self,
        pos_flow: torch.Tensor,
        neg_flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate the velocity fields to obtain diffeomorphic warp (displacement) fields.

        Derive a smooth and invertable displacement field by integrating a velocity field via the
        scaling and squaring method. If no `IntegrateVelocityField` object exists in the model's
        state dictionary, instantiate it with the correct size of the input and insert it. This will
        only happen once upon initial call of this method.

        Parameters
        ----------
        pos_flow : torch.Tensor
            Positive flow (velocity) field (source -> target).
        neg_flow : torch.Tensor
            Negative flow (velocity) field (target -> source).

        Returns
        -------
        torch.Tensor
            Displacement field obtained by integrating the velocity field via scaling and squaring.
        """

        # Ensure ops based on the flow’s current shape/device
        self._ensure_ops(pos_flow)

        # Integrate the positive flow
        pos_flow = self.velocity_field_integrator(pos_flow)

        # Integrate the negative velocity field if bidirectional cost is enabled
        neg_flow = self.velocity_field_integrator(neg_flow) if self.bidirectional_cost else None

        return pos_flow, neg_flow

    def _spatial_transform(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp an image tensor using a deformation/displacement field.

        This method applies a spatial transformation to the provided image tensor based on the
        deformation field using a SpatialTransformer. If no `IntegrateVelocityField` object exists
        in the model's state dictionary, instantiate one with the correct size of the input and
        register it as a submodule. This will only happen once upon the initial call of this method.

        Parameters
        ----------
        moving_image : torch.Tensor
            Image tensor to be warped, with shape (B, C, ...).
        deformation_field : torch.Tensor
            Displacement field used for warping, with shape matching the spatial dimensions of
            `moving_image`.

        Returns
        -------
        torch.Tensor
            The warped image tensor.
        """

        # Ensure ops (use deformation field for shape/device)
        self._ensure_ops(deformation_field)
        
        # Warp the moving image with the deformation field
        warped_image = self.spatial_transformer(moving_image, deformation_field)

        return warped_image
