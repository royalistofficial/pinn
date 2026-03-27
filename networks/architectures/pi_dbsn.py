import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BSplineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.residual = residual and (in_dim == out_dim)

        self.n_coeffs = grid_size + spline_order

        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim, self.n_coeffs))
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_activation = nn.SiLU()

        step = 2.0 / grid_size
        grid = torch.arange(
            -grid_size - spline_order, 
            grid_size + spline_order + 1, 
            dtype=torch.float32
        ) * step

        self.register_buffer("grid", grid.repeat(in_dim, 1))

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) 

        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):

            denom1 = self.grid[:, k:-1] - self.grid[:, :-(k + 1)] + 1e-8
            left = (x - self.grid[:, :-(k + 1)]) / denom1

            denom2 = self.grid[:, k + 1:] - self.grid[:, 1:-k] + 1e-8
            right = (self.grid[:, k + 1:] - x) / denom2

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_clamped = x.clamp(-1.0 + 1e-4, 1.0 - 1e-4)

        spline_basis = self.b_spline(x_clamped) 

        spline_out = torch.einsum("oic,bic->bo", self.coeffs, spline_basis)

        base_out = F.linear(self.base_activation(x), self.base_weight)

        y = spline_out + base_out

        if self.residual:
            y = y + x

        return y

class PIDBSN(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 1, hidden_dim: int = 64, n_layers: int = 4, grid_size: int = 5, spline_order: int = 3):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(BSplineLayer(in_dim, hidden_dim, grid_size, spline_order, residual=False))

        for _ in range(n_layers - 1):
            self.layers.append(BSplineLayer(hidden_dim, hidden_dim, grid_size, spline_order, residual=True))

        self.head = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = 0.95 * x 
        for layer in self.layers:
            h = layer(h)
        return self.head(h)