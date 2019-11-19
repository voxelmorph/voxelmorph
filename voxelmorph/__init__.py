# ---- voxelmorph ----
# unsupervised learning for image registration

from . import utils
from . import generators

# import backend-dependent submodules
backend = utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import pytorch_backend
    from .pytorch_backend import layers
    from .pytorch_backend import networks
    from .pytorch_backend import losses
else:
    # tensorflow is default backend
    from . import tensorflow_backend
    from .tensorflow_backend import layers
    from .tensorflow_backend import networks
    from .tensorflow_backend import losses
