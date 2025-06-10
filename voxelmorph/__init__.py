# ---- voxelmorph ----
# unsupervised learning for image registration

import os

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

import tensorflow
# ensure valid tensorflow version is available
minv = '2.4'
curv = getattr(tensorflow, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires tensorflow version {minv} or greater, '
                        f'but found version {curv}')


from . import py
from . import utils
from . import generators
from . import layers
from . import losses
from . import networks
