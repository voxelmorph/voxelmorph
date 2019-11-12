import os
import sys

import voxelmorph as vxm
module = os.path.dirname(os.path.abspath(vxm.__file__))
extern = os.path.join(module, 'external')
sys.path.insert(0, os.path.join(extern, 'pytools-lib'))
sys.path.insert(0, os.path.join(extern, 'pynd-lib'))
sys.path.insert(0, os.path.join(extern, 'neuron'))

from . import layers
from . import networks
from . import losses
