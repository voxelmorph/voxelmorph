# ---- voxelmorph ----
# unsupervised learning for image registration

from . import utils
from . import generators

# import backend-dependent submodules
backend = utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import torch
    from .torch import layers
    from .torch import networks
    from .torch import losses
else:
    # tensorflow is default backend
    from . import tf
    from .tf import layers
    from .tf import networks
    from .tf import losses
