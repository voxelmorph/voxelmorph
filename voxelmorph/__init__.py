# ---- voxelmorph ----
# unsupervised learning for image registration

from . import utils
from . import generators

# import backend-dependent submodules
if utils.get_backend() == 'pytorch':
    # the pytorch backend can be enabled by setting the
    # VXM_BACKEND environment var to "pytorch"
    from . import torch_backend as backend
    from .torch_backend import layers
    from .torch_backend import networks
    from .torch_backend import losses
else:
    # tensorflow is default backend
    from . import tf_backend as backend
    from .tf_backend import layers
    from .tf_backend import networks
    from .tf_backend import losses
