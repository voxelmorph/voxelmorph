# ---- voxelmorph ----
# unsupervised learning for image registration


# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
# import neurite
# minv = '0.2'
# curv = getattr(neurite, '__version__', None)
# if curv is None or version.parse(curv) < version.parse(minv):
#     raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
#                       f'but found version {curv}')

# move on the actual voxelmorph imports
from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
reg = "groupwise"

if reg == 'pairwise':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import torch
    from .pairwise import layers
    from .pairwise import networks
    from .pairwise import losses

elif reg == 'groupwise':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import torch
    from .groupwise import layers
    from .groupwise import networks
    from .groupwise import losses
