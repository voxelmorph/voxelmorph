import os
import sys

# The tensorflow backend relies on the neuron, pytools, and pynd
# packages, which aren't available via pip and are included directly
# in the 'external' subdirectory. We also want to keep these packages independent
# of one another, so we don't import them as submodules but instead as
# individual modules by modifying sys.path.

subdirs = ['pytools-lib', 'neuron']

# add external subdirs to sys.path
basedir = os.path.join(os.path.dirname(__file__), 'external')
paths = [os.path.join(basedir, package) for package in subdirs]
sys.path = paths + sys.path

from . import layers
from . import networks
from . import losses

# We don't want the modified sys.path to potentially interfere with a
# user's other module imports (if they share the same name), so we'll
# revert back to the orig sys.path after import. This isn't the cleanest
# method, as differed imports might throw errors, but it's a workable
# solution for now.

sys.path = sys.path[len(subdirs):]
