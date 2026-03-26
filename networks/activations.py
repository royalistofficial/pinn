from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnActivation(nn.Module):
    def __init__(self, n: int = 1, alpha: float = 0.01):
        super().__init__()
        self.n = n
        self.alpha = alpha
        self.norm_factor = math.factorial(n + 1)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = F.relu(x)
        x_neg = F.relu(-x)
        out = (x_pos.pow(self.n + 1) - self.alpha * x_neg.pow(self.n + 1)) / self.norm_factor
        return self.scale * out

    def extra_repr(self) -> str:
        return f"n={self.n}, alpha={self.alpha}"

class SoftplusSmooth(nn.Module):
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta
        self._shift = math.log(2.0) / beta  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) - self._shift

def make_activation(name: str = "gelu", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    elif name == "softplus":
        return SoftplusSmooth(beta=kwargs.get("beta", 2.0))
    elif name == "cn":
        return CnActivation(
            n=kwargs.get("n", 2),
            alpha=kwargs.get("alpha", 0.01),
        )
    else:
        raise ValueError(f"Unknown activation: {name}. Use: gelu, silu, softplus, cn")
