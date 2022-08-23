# ---- voxelmorph ----
# unsupervised learning for image registration


# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
import neurite
minv = '0.2'
curv = getattr(neurite, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
                      f'but found version {curv}')

# move on the actual voxelmorph imports
from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()

if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import torch
    from .torch import layers
    from .torch import networks
    from .torch import losses

else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this voxelmorph backend')

    # ensure valid tensorflow version is available
    minv = '2.4'
    curv = getattr(tensorflow, '__version__', None)
    if curv is None or version.parse(curv) < version.parse(minv):
        raise ImportError(f'voxelmorph requires tensorflow version {minv} or greater, '
                          f'but found version {curv}')

    from . import tf
    from .tf import layers
    from .tf import networks
    from .tf import losses
    from .tf import utils
