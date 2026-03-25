from config import PINN_ARCH
from networks.blocks import FourierCnNet
from networks.corners import CornerEnrichment

class PINN(FourierCnNet):
    def __init__(self, corner_enrichment: CornerEnrichment | None = None):
        super().__init__(out_dim=1, n_smooth=2, corner_enrichment=corner_enrichment, **PINN_ARCH)
