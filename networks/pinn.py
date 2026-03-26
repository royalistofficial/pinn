from __future__ import annotations
from typing import Optional
import torch

from config import PINN_ARCH
from networks.blocks import FourierNet
from networks.corners import CornerEnrichment

class PINN(FourierNet):
    def __init__(
        self,
        corner_enrichment: CornerEnrichment | None = None,
        init_freqs: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            out_dim=1,
            corner_enrichment=corner_enrichment,
            init_freqs=init_freqs,
            **PINN_ARCH,
        )
