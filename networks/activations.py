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
