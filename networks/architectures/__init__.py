from .mlp import MLP
from .siren import SIREN, SirenLayer
from .fourier import FourierNet
from .kan_pinn import KANPINN, ScaledCPIKAN, ChebyKANLayer
from .enrichment import CornerEnrichment, build_corner_enrichment, extract_corners

__all__ = [
    "MLP",
    "FourierNet",
    "SIREN",
    "SirenLayer",
    "KANPINN",
    "ScaledCPIKAN",
    "ChebyKANLayer",
    "CornerEnrichment",
    "build_corner_enrichment",
    "extract_corners",
    "CURRENT_ARCHITECTURE_CONFIG"
]

KAN_CONFIG = {
    "architecture": "kan",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 64,
    "n_layers": 4,
    "activation": "tanh",
    "use_corner_enrichment": True, 
    "kan_degree": 3,
    "n_fourier": 2,
    "freq_min": 1.0,
    "freq_max": 2.0,
}

MLP_CONFIG = {
    "architecture": "mlp",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 64,
    "n_layers": 4,
    "activation": "tanh",
    "use_corner_enrichment": False,
}

SIREN_CONFIG = {
    "architecture": "siren",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 64,
    "n_layers": 4,
    "use_corner_enrichment": False,
    "siren_w0": 30.0,
}

FOURIER_CONFIG = {
    "architecture": "fourier",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 64,
    "n_layers": 4,
    "activation": "tanh",
    "use_corner_enrichment": False,
    "fourier_features": 256,
    "fourier_sigma": 10.0,
}

CURRENT_ARCHITECTURE_CONFIG = MLP_CONFIG