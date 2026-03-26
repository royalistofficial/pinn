from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from networks.architectures import (
    CornerEnrichment, 
    build_corner_enrichment,
    ScaledCPIKAN,
    MLP,
    SIREN,
    FourierNet
)

@dataclass
class PINNConfig:
    architecture: str = "kan"
    in_dim: int = 2
    out_dim: int = 1
    hidden_dim: int = 64
    n_layers: int = 4
    activation: str = "tanh"
    use_corner_enrichment: bool = True

    siren_w0: float = 30.0
    fourier_features: int = 256
    fourier_sigma: float = 10.0
    freq_min: float = 0.5
    freq_max: float = 10.0
    kan_degree: int = 5
    n_fourier: int = 0

class PINNFactory:
    def __init__(self, config: PINNConfig):
        self.config = config

        self._registry: Dict[str, Type[nn.Module]] = {
            "kan": ScaledCPIKAN,
            "mlp": MLP,
            "siren": SIREN,
            "fourier": FourierNet,
        }

    def build_core_model(self, corner_enrichment: Optional[CornerEnrichment] = None) -> nn.Module:
        arch_name = self.config.architecture.lower()
        if arch_name not in self._registry:
            raise ValueError(f"Архитектура '{arch_name}' не поддерживается. Доступны: {list(self._registry.keys())}")

        ModelClass = self._registry[arch_name]

        if arch_name == "kan":
            return ModelClass(
                out_dim=self.config.out_dim,
                hidden=self.config.hidden_dim,
                n_layers=self.config.n_layers,
                degree=self.config.kan_degree,
                n_fourier=self.config.n_fourier,
                freq_min=self.config.freq_min,
                freq_max=self.config.freq_max,
                corner_enrichment=corner_enrichment
            )
        else:

            return ModelClass(self.config)

class PINN(nn.Module):
    def __init__(
        self,
        config: PINNConfig,
        domain=None,
        corner_enrichment: Union[bool, CornerEnrichment] = False,
    ):
        super().__init__()
        self.config = config

        if isinstance(corner_enrichment, bool):
            if corner_enrichment:
                if domain is None:
                    raise ValueError("Для автоматической сборки CornerEnrichment необходимо передать объект domain.")
                self.corner_enrichment = build_corner_enrichment(domain, torch.device("cpu"))
            else:
                self.corner_enrichment = None
        else:
            self.corner_enrichment = corner_enrichment

        self.factory = PINNFactory(self.config)
        self.model = self.factory.build_core_model(corner_enrichment=self.corner_enrichment)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enrichment_status = 'Active' if self.corner_enrichment is not None else 'Disabled'
        print(
            f"[{self.__class__.__name__} | {self.config.architecture.upper()}] "
            f"{total_params:,} trainable params | "
            f"Corner Enrichment: {enrichment_status}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.config.architecture != "kan" and self.corner_enrichment is not None:
            c_feat = self.corner_enrichment(x)
            x = torch.cat([x, c_feat], dim=-1)

        return self.model(x)

    def plot_model_summary(self, save_path: str = "data/model_summary.png"):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layers_info = []

        for name, module in self.named_modules():
            if module is self or isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue

            params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if params > 0 or "enrichment" in name.lower() or "activation" in module.__class__.__name__.lower():
                layer_name = name if name else "core_layer"
                param_str = f"({params:,} params)" if params > 0 else "(0 params)"
                layers_info.append(f" ├─ {layer_name}: {module.__class__.__name__} {param_str}")

        fig_height = max(4.0, len(layers_info) * 0.35 + 3.0)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.axis('off')

        text_str =  f"[*] PINN Model Summary\n"
        text_str += f"══════════════════════════════════════════════════\n"
        text_str += f" Architecture   : {self.config.architecture.upper()}\n"
        text_str += f" Output Dim     : {self.config.out_dim}\n"
        text_str += f" Enrichment     : {'Active' if self.corner_enrichment else 'Disabled'}\n"
        text_str += f" Total Params   : {total_params:,}\n"
        text_str += f"══════════════════════════════════════════════════\n\n"
        text_str += " Layer Structure:\n"
        text_str += "\n".join(layers_info)

        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle='round4,pad=1', facecolor='#f8f9fa', 
                          edgecolor='#ced4da', linewidth=2, alpha=0.95))

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[{self.__class__.__name__}] Информативная сводка модели сохранена в: {save_path}")