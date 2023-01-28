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

try:
    import torch
except ImportError:
    raise ImportError('Please install pytorch to use this voxelmorph backend')

from . import groupwise
from .groupwise import layers
from .groupwise import networks
from .groupwise import losses
