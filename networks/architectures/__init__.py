from .mlp import MLP
from .siren import SIREN, SirenLayer
from .fourier import FourierNet
from .kan_pinn import ScaledCPIKAN, ChebyKANLayer
from .pi_dbsn import PIDBSN
from .rbf_kan import RBFKAN
from .wav_kan import WavKAN

from networks.configs import (
    NetworkConfig,
    DEFAULT_CONFIG,
    PRESET_CONFIGS,
    get_config,
)

__all__ = [

    "MLP",
    "FourierNet",
    "SIREN",
    "SirenLayer",
    "ScaledCPIKAN",
    "ChebyKANLayer",
    "PIDBSN",
    "RBFKAN",
    "WavKAN",

    "NetworkConfig",
    "DEFAULT_CONFIG",
    "PRESET_CONFIGS",
    "get_config",
]

MLP_CONFIG = {
    "architecture": "mlp",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 32,
    "n_layers": 3,
    "activation": "tanh",
}

SIREN_CONFIG = {
    "architecture": "siren",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 32,
    "n_layers": 3,
    "siren_w0": 20.0,
}

FOURIER_CONFIG = {
    "architecture": "fourier",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 8,
    "n_layers": 2,
    "activation": "tanh",
    "fourier_features": 32,
    "fourier_sigma": 10.0,
    "freq_min": 1.0,
    "freq_max": 2.0,
    "trainable_freqs": False
}

KAN_CONFIG = {
    "architecture": "kan",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 18,
    "n_layers": 2,
    "kan_degree": 5, 
}

PI_DBSN_CONFIG = {
    "architecture": "pi-dbsn",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 15,
    "n_layers": 2,
    "dbsn_grid_size": 5,
    "dbsn_spline_order": 3,
}

RBF_KAN_CONFIG = {
    "architecture": "rbf-kan",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 18,
    "n_layers": 2,
    "num_rbf_centers": 5,
}

WAV_KAN_CONFIG = {
    "architecture": "wav-kan",
    "in_dim": 2,
    "out_dim": 1,
    "hidden_dim": 18,
    "n_layers": 2,
    "num_wavelets": 5,
}

CURRENT_ARCHITECTURE_CONFIG = MLP_CONFIG

__all__.extend([
    "MLP_CONFIG",
    "SIREN_CONFIG", 
    "FOURIER_CONFIG",
    "KAN_CONFIG",
    "PI_DBSN_CONFIG",
    "RBF_KAN_CONFIG",
    "WAV_KAN_CONFIG",
    "CURRENT_ARCHITECTURE_CONFIG",
])
