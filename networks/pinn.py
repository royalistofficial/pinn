from __future__ import annotations
import os
import copy
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from networks.architectures import (
    GeometryEnrichment, 
    build_enrichment,
    ScaledCPIKAN,
    PIDBSN,
    RBFKAN,
    WavKAN,
    MLP,
    SIREN,
    FourierNet
)

@dataclass
class PINNConfig:
    architecture: str = "mlp"
    in_dim: int = 2
    out_dim: int = 1
    hidden_dim: int = 64
    n_layers: int = 4
    activation: str = "tanh"
    use_corner_enrichment: bool = False

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

class PINNFactory:
    def __init__(self, config: PINNConfig):
        self.config = config

        self._registry: Dict[str, Type[nn.Module]] = {
            "kan": ScaledCPIKAN,
            "mlp": MLP,
            "siren": SIREN,
            "fourier": FourierNet,
            "pi-dbsn": PIDBSN,
            "rbf-kan": RBFKAN,
            "wav-kan": WavKAN,
        }

    def build_core_model(self, corner_enrichment: Optional[GeometryEnrichment] = None) -> nn.Module:
        arch_name = self.config.architecture.lower()
        if arch_name not in self._registry:
            raise ValueError(f"Архитектура '{arch_name}' не поддерживается. Доступны: {list(self._registry.keys())}")

        ModelClass = self._registry[arch_name]

        return ModelClass(self.config)

class PINN(nn.Module):
    def __init__(
        self,
        config: PINNConfig,
        domain=None,
        corner_enrichment: Union[bool, GeometryEnrichment] = False,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)

        if isinstance(corner_enrichment, bool):
            if corner_enrichment:
                if domain is None:
                    raise ValueError("Для автоматической сборки GeometryEnrichment необходимо передать объект domain.")
                self.corner_enrichment = build_enrichment(domain, torch.device("cpu"))
            else:
                self.corner_enrichment = None
        else:
            self.corner_enrichment = corner_enrichment

        actual_in_dim = self.config.in_dim
        if self.config.architecture != "kan" and self.corner_enrichment is not None:
            dummy_x = torch.zeros(1, self.config.in_dim)
            with torch.no_grad():
                c_feat = self.corner_enrichment(dummy_x)
            actual_in_dim += c_feat.shape[-1]

        self.config.in_dim = actual_in_dim

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
        import os
        import matplotlib.pyplot as plt
        import torch.nn as nn

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layers_info = []

        for name, module in self.named_modules():
            if module is self:
                continue

            params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            depth = name.count('.')
            indent = "    " * depth
            branch = "├── " if depth > 0 else "■ "

            layer_short_name = name.split('.')[-1] if name else "core"
            class_name = module.__class__.__name__

            extras = []
            if params > 0:
                extras.append(f"Weights: {params:,}")

            if hasattr(module, 'residual') and module.residual:
                extras.append("(+) Residual (+x)")
            if hasattr(module, 'res_scale'): 
                extras.append("(+) Learnable Skip-Conn")

            if hasattr(module, 'base_activation'):
                act_name = module.base_activation.__class__.__name__
                extras.append(f"Base Act: {act_name}")

            if hasattr(module, 'w0'):
                extras.append(f"w0={module.w0}")
                if getattr(module, 'is_first', False):
                    extras.append("First Layer (Sine)")
                else:
                    extras.append("Sine Act")

            if isinstance(module, (nn.Tanh, nn.SiLU, nn.GELU, nn.ReLU)):
                extras.append("f(x) Activation")

            is_container = isinstance(module, (nn.Sequential, nn.ModuleList))

            if params > 0 or extras or is_container or "enrichment" in name.lower():
                extra_str = f" [{', '.join(extras)}]" if extras else ""

                line = f"{indent}{branch}{layer_short_name} ({class_name}){extra_str}"
                layers_info.append(line)

        fig_height = max(5.0, len(layers_info) * 0.28 + 3.0)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        text_str =  f"[*] PINN Architecture & Data Flow\n"
        text_str += f"════════════════════════════════════════════════════════════════\n"
        text_str += f" Model Type     : {self.config.architecture.upper()}\n"
        text_str += f" Total Params   : {total_params:,}\n"
        text_str += f" Corner Features: {'Enabled' if getattr(self, 'corner_enrichment', None) else 'Disabled'}\n"
        text_str += f"════════════════════════════════════════════════════════════════\n\n"
        text_str += " Structure:\n"
        text_str += "\n".join(layers_info)

        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle='round4,pad=1', facecolor='#1e1e2e', edgecolor='#89b4fa', linewidth=2, alpha=0.95), color='#cdd6f4')

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#f1f3f5')
        plt.close()
        print(f"[{self.__class__.__name__}] Расширенная сводка модели сохранена в: {save_path}")