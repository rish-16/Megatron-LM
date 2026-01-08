# Hypercore neural network layers
# Import all layers from submodules
from .conv import *
from .linear import *
from .attention import *

# Graph convolution layers are optional (require torch_scatter)
from .graph_conv import *

# PEFT layers
from .peft import *

# Create a reference to this module itself so code can do:
# from megatron.core.hypercore.nn import nn
# nn.LorentzRMSNorm(...)
import sys
nn = sys.modules[__name__]
