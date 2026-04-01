from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NetworkConfig:

    architecture: str = "mlp"
    in_dim: int = 2
    out_dim: int = 1
    hidden_dim: int = 64
    grid_size: int = 5
    spline_order: int = 3
    n_layers: int = 4
    activation: str = "tanh"

    siren_w0: float = 30.0

    fourier_features: int = 256
    fourier_sigma: float = 10.0
    freq_min: float = 0.5
    freq_max: float = 10.0
    trainable_freqs: bool = False

    kan_degree: int = 5

    dbsn_grid_size: int = 5
    dbsn_spline_order: int = 3

    num_rbf_centers: int = 5

    num_wavelets: int = 5

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

    @classmethod
    def mlp(cls, hidden_dim: int = 28, n_layers: int = 3, 
            activation: str = "tanh", **kwargs) -> "NetworkConfig":
        return cls(
            architecture="mlp",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            **kwargs
        )

    @classmethod
    def siren(cls, hidden_dim: int = 28, n_layers: int = 3,
              w0: float = 30.0, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="siren",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            siren_w0=w0,
            **kwargs
        )

    @classmethod
    def fourier(cls, hidden_dim: int = 16, n_layers: int = 3,
                features: int = 8, sigma: float = 10.0,
                trainable: bool = True, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="fourier",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            fourier_features=features,
            fourier_sigma=sigma,
            trainable_freqs=trainable,
            **kwargs
        )

    @classmethod
    def kan(cls, hidden_dim: int = 16, n_layers: int = 2,
            grid_size: int = 5, spline_order: int = 3, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="kan",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            **kwargs
        )

    @classmethod
    def cheby_kan(cls, hidden_dim: int = 16, n_layers: int = 2,
            kan_degree: int = 5, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="cheby_kan",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kan_degree=kan_degree,
            **kwargs
        )

    @classmethod
    def pi_dbsn(cls, hidden_dim: int = 12, n_layers: int = 2,
                grid_size: int = 5, spline_order: int = 3, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="pi-dbsn",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dbsn_grid_size=grid_size,
            dbsn_spline_order=spline_order,
            **kwargs
        )

    @classmethod
    def rbf_kan(cls, hidden_dim: int = 16, n_layers: int = 2,
                num_centers: int = 5, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="rbf-kan",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_rbf_centers=num_centers,
            **kwargs
        )

    @classmethod
    def wav_kan(cls, hidden_dim: int = 16, n_layers: int = 2,
                num_wavelets: int = 5, **kwargs) -> "NetworkConfig":
        return cls(
            architecture="wav-kan",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_wavelets=num_wavelets,
            **kwargs
        )

DEFAULT_CONFIG = NetworkConfig.mlp()

PRESET_CONFIGS = {
    "mlp": NetworkConfig.mlp(),
    "fourier": NetworkConfig.fourier(),
    "siren": NetworkConfig.siren(),
    "pi-dbsn": NetworkConfig.pi_dbsn(),
    "kan": NetworkConfig.kan(),
    "cheby_kan": NetworkConfig.cheby_kan(),
    "rbf-kan": NetworkConfig.rbf_kan(),
    "wav-kan": NetworkConfig.wav_kan(),
}

def get_config(name: Optional[str] = None, **overrides) -> NetworkConfig:

    if name is None:
        config = DEFAULT_CONFIG
    else:
        if name not in PRESET_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Available: {list(PRESET_CONFIGS)}")
        config = PRESET_CONFIGS[name]

    if overrides:
        data = {k: v for k, v in config.__dict__.items()}
        data.update(overrides)
        return NetworkConfig(**data)

    return config