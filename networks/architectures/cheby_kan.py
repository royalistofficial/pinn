import math
import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, degree: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree

        std = 1.0 / math.sqrt(in_dim * (degree + 1))
        self.coeffs = nn.Parameter(torch.empty(in_dim, out_dim, degree + 1))
        nn.init.normal_(self.coeffs, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = x.shape

        x_norm = torch.tanh(x)

        x_norm = x_norm.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(x_norm)                                      # (batch, in_dim)
        T_list = [torch.cos(k * theta) for k in range(self.degree + 1)]
        T = torch.stack(T_list, dim=2)                                  # (batch, in_dim, degree+1)

        y = torch.einsum('bij,ioj->bo', T, self.coeffs)

        return y


class ScaledCPIKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden = getattr(config, 'hidden_dim', 25)
        n_layers = getattr(config, 'n_layers', 3)
        degree = getattr(config, 'kan_degree', 10)

        self.layers = nn.ModuleList()
        self.layers.append(ChebyKANLayer(in_dim, hidden, degree))

        for _ in range(n_layers - 1):
            self.layers.append(ChebyKANLayer(hidden, hidden, degree))

        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)