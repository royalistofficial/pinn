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

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layers_info = []

        modules = list(self.named_modules())

        for i, (name, module) in enumerate(modules):
            if module is self:
                continue

            params = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            parts = name.split('.')
            depth = len(parts) - 1
            layer_short_name = parts[-1] if name else "core"

            if depth == 0:
                prefix = "■ "
            else:
                prefix = "│   " * (depth - 1) + "├── "

            class_name = module.__class__.__name__

            extras = []

            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                extras.append(f"{module.in_features}→{module.out_features}")
            elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                extras.append(f"{module.in_channels}→{module.out_channels}")

            if params > 0:
                extras.append(f"W:{params:,}")

            is_activation = isinstance(module, (
                nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, 
                nn.Sigmoid, nn.ELU, nn.Softplus, nn.Mish
            )) or "activation" in module.__module__.lower()

            if is_activation:
                extras.append("ƒ(x) Act")

            has_skip = any(
                hasattr(module, attr) and getattr(module, attr) for attr in 
                ['residual', 'use_skip', 'res_scale', 'shortcut', 'skip_connection', 'has_residual']
            )
            if has_skip:
                extras.append("⤿ Skip-Conn")

            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                extras.append("± Norm")
            if isinstance(module, nn.Dropout):
                extras.append(f"✗ Drop(p={module.p})")

            if hasattr(module, 'w0'):
                extras.append(f"w0={getattr(module, 'w0')}")
            if hasattr(module, 'sigma'):
                extras.append(f"σ={getattr(module, 'sigma')}")
            if hasattr(module, 'grid_size'):
                extras.append(f"grid={getattr(module, 'grid_size')}")
            if hasattr(module, 'degree'):
                extras.append(f"deg={getattr(module, 'degree')}")

            extra_str = f" [{', '.join(extras)}]" if extras else ""
            line = f"{prefix}{layer_short_name} ({class_name}){extra_str}"
            layers_info.append(line)

        line_height = 0.22
        fig_height = max(6.0, len(layers_info) * line_height + 4.0)

        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')

        text_str = f"[*] PINN ARCHITECTURE SUMMARY\n"
        text_str += "═" * 70 + "\n"

        arch = getattr(self.config, 'architecture', 'unknown').upper() if hasattr(self, 'config') else 'N/A'
        h_dim = getattr(self.config, 'hidden_dim', 'N/A') if hasattr(self, 'config') else 'N/A'
        n_lay = getattr(self.config, 'n_layers', 'N/A') if hasattr(self, 'config') else 'N/A'

        text_str += f" Architecture    : {arch}\n"
        text_str += f" Hidden Dim      : {h_dim}\n"
        text_str += f" Layers          : {n_lay}\n"
        text_str += f" Total Params    : {total_params:,}\n"
        text_str += "═" * 70 + "\n\n"
        text_str += " Model Structure:\n"
        text_str += "\n".join(layers_info)

        ax.text(
            0.02, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle='round4,pad=1.5',
                facecolor='#1e1e2e',    
                edgecolor='#89b4fa',    
                linewidth=2,
                alpha=0.95
            ),
            color='#cdd6f4'             
        )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f1f3f5')
        plt.close(fig)

        print(f"[PINN] Подробная структура модели сохранена: {save_path}")