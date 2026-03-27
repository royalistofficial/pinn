import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_centers: int = 5, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_centers = num_centers

        self.residual = residual and (in_dim == out_dim)

        grid = torch.linspace(-1.0, 1.0, num_centers).unsqueeze(0).repeat(in_dim, 1)
        self.centers = nn.Parameter(grid) 

        self.widths = nn.Parameter(torch.ones(in_dim, num_centers) * (2.0 / num_centers))

        self.weights = nn.Parameter(torch.empty(out_dim, in_dim, num_centers))
        nn.init.normal_(self.weights, std=0.1)

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_u = x.unsqueeze(-1) 

        rbf = torch.exp(-((x_u - self.centers) ** 2) / (self.widths ** 2 + 1e-8)) 

        y_rbf = torch.einsum("bic,oic->bo", rbf, self.weights)

        y_base = F.linear(self.base_activation(x), self.base_weight)

        y = y_rbf + y_base

        if self.residual:
            y = y + x

        return y

class RBFKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)
        num_centers = getattr(config, 'num_rbf_centers', 5)

        self.layers = nn.ModuleList()

        self.layers.append(RBFKANLayer(in_dim, hidden_dim, num_centers, residual=False))

        for _ in range(n_layers - 1):
            self.layers.append(RBFKANLayer(hidden_dim, hidden_dim, num_centers, residual=True))

        self.head = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.head(h)