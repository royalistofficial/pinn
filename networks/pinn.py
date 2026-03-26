from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from config import PINN_ARCH, PINN_MLP, MLP
from networks.blocks import ScaledCPIKAN
from networks.corners import CornerEnrichment

class PINN(nn.Module):
    def __init__(
        self,
        corner_enrichment: CornerEnrichment | None = None,
    ):
        super().__init__()
        if MLP:
            self.model = self.build_mlp()
        else:
            self.model = ScaledCPIKAN(
                out_dim=1,
                corner_enrichment=corner_enrichment,
                **PINN_ARCH,
            )

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[{self.__class__.__name__}] {total_params:,} trainable params | "
        )

    def build_mlp(self):
        layers = []

        in_dim = 2    
        hidden = PINN_MLP["hidden"]
        n_blocks = PINN_MLP["n_blocks"]

        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Sigmoid())

        for _ in range(n_blocks):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
