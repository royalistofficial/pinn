from __future__ import annotations
import os
import copy
from typing import Dict, Type, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from networks.configs import NetworkConfig
from networks.architectures import (
    ScaledCPIKAN,
    PIDBSN,
    RBFKAN,
    WavKAN,
    MLP,
    SIREN,
    FourierNet,
    ScaledCPIKAN
)

class PINNFactory:
    _registry: Dict[str, Type[nn.Module]] = {
        "kan": ScaledCPIKAN,
        "mlp": MLP,
        "siren": SIREN,
        "fourier": FourierNet,
        "pi-dbsn": PIDBSN,
        "cheby_kan": ScaledCPIKAN,
        "rbf-kan": RBFKAN,
        "wav-kan": WavKAN,
    }

    @classmethod
    def create(cls, config: NetworkConfig) -> nn.Module:
        arch_name = config.architecture.lower()

        if arch_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Архитектура '{arch_name}' не поддерживается. "
                f"Доступны: {available}"
            )

        ModelClass = cls._registry[arch_name]

        config_dict = config.__dict__.copy()
        config_dict.pop("architecture", None)  

        return ModelClass(config)

    @classmethod
    def available_architectures(cls) -> list:
        return list(cls._registry.keys())

class PINN(nn.Module):
    def __init__(self, config: NetworkConfig | dict):
        super().__init__()

        self.config = copy.deepcopy(config)
        self.model = PINNFactory.create(self.config)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[PINN | {self.config.architecture.upper()}] "
            f"{total_params:,} trainable parameters"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
