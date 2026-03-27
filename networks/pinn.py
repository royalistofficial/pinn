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
    FourierNet
)

class PINNFactory:
    _registry: Dict[str, Type[nn.Module]] = {
        "kan": ScaledCPIKAN,
        "mlp": MLP,
        "siren": SIREN,
        "fourier": FourierNet,
        "pi-dbsn": PIDBSN,
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

        if isinstance(config, dict):
            config = NetworkConfig(**config)

        self.config = copy.deepcopy(config)
        self.model = PINNFactory.create(self.config)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[PINN | {self.config.architecture.upper()}] "
            f"{total_params:,} trainable parameters"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def plot_model_summary(self, save_path: str = "data/model_summary.png") -> None:
        total_params = self.count_parameters()
        layers_info = []

        for name, module in self.named_modules():
            if module is self:
                continue

            params = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )
            depth = name.count('.')
            indent = "    " * depth
            branch = "├── " if depth > 0 else "■ "

            layer_short_name = name.split('.')[-1] if name else "core"
            class_name = module.__class__.__name__

            extras = []
            if params > 0:
                extras.append(f"Weights: {params:,}")

            if hasattr(module, 'residual') and module.residual:
                extras.append("Residual")
            if hasattr(module, 'res_scale'):
                extras.append("Skip-Conn")
            if hasattr(module, 'w0'):
                extras.append(f"w0={module.w0}")
            if isinstance(module, (nn.Tanh, nn.SiLU, nn.GELU, nn.ReLU)):
                extras.append("Activation")

            is_container = isinstance(module, (nn.Sequential, nn.ModuleList))

            if params > 0 or extras or is_container:
                extra_str = f" [{', '.join(extras)}]" if extras else ""
                line = f"{indent}{branch}{layer_short_name} ({class_name}){extra_str}"
                layers_info.append(line)

        fig_height = max(5.0, len(layers_info) * 0.28 + 3.0)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        text_str = f"[*] PINN Architecture\n"
        text_str += "═" * 60 + "\n"
        text_str += f" Architecture    : {self.config.architecture.upper()}\n"
        text_str += f" Hidden Dim      : {self.config.hidden_dim}\n"
        text_str += f" Layers          : {self.config.n_layers}\n"
        text_str += f" Total Params    : {total_params:,}\n"
        text_str += "═" * 60 + "\n\n"
        text_str += " Structure:\n"
        text_str += "\n".join(layers_info)

        ax.text(
            0.02, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle='round4,pad=1',
                facecolor='#1e1e2e',
                edgecolor='#89b4fa',
                linewidth=2,
                alpha=0.95
            ),
            color='#cdd6f4'
        )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#f1f3f5')
        plt.close(fig)

        print(f"[PINN] Model summary saved: {save_path}")
