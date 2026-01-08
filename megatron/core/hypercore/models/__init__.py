# Hypercore models
# Core models that don't require graph dependencies
from .lorentz_feedforward import LorentzFeedForward
from .tokenizer import Tokenizer

# Models with optional dependencies
try:
    from .lorentz_resnet import Lorentz_ResNet
    from .lorentz_resnet import Lorentz_resnet18, Lorentz_resnet34, Lorentz_resnet50, Lorentz_resnet101, Lorentz_resnet152
    from .LViT import LViT
    from .Transformer_encoder import LTransformerEncoder
except ImportError:
    pass

# Graph-based models require torch_scatter/torch_geometric
try:
    from .graph_models import BaseModel
    from .graph_models import LPModel
    from .graph_models import NCModel
    from .graph_models import MDModel
    from .graph_retriever import GRetriever
    from .LCLIP import LCLIP
    _GRAPH_MODELS_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Graph-based models not available: {e}. Install torch_scatter and torch_geometric for graph operations.")
    _GRAPH_MODELS_AVAILABLE = False
