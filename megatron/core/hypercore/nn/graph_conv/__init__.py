# Graph convolution layers - requires torch_scatter and torch_geometric
# These are optional dependencies for graph-based operations

try:
    from .CentroidDistance import CentroidDistance
    from .att_layers import DenseAtt
    from .att_layers import SpecialSpmmFunction
    from .att_layers import SpecialSpmm
    from .att_layers import SpGraphAttentionLayer
    from .att_layers import GraphAttentionLayer
    from .att_layers import GraphConvolution
    from .hgcn_conv import HGCNConv
    from .hgcn_conv import HypAgg
    from .lgcn_conv import LGCNConv
    from .lgcn_conv import LGCNAgg
    from .lgcn_conv import LGCNLinear
    from .hyobnet_conv import HybonetConv
    from .hyobnet_conv import LorentzAgg
    from .qgcn_conv import QGCNConv
    from .qgcn_conv import PseudoHypAgg
    from .hgat_conv import HGATConv
    from .gat_conv import GATConv
    from .gcn_conv import GCNConv
    from .hgnn_conv import HGNNConv
    from .gil_conv import GILConv
    from .gil_conv import EFusion
    from .gil_conv import HFusion
    _GRAPH_CONV_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Graph convolution layers not available: {e}. Install torch_scatter and torch_geometric for graph operations.")
    _GRAPH_CONV_AVAILABLE = False
