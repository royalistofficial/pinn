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
        n_knots = grid_size + 2 * spline_order + 1

        self.grid_steps_log = nn.Parameter(
            torch.log(torch.ones(in_dim, n_knots - 1) * step)
        )

        self.grid_start = nn.Parameter(
            torch.ones(in_dim, 1) * (-1.0 - step * spline_order)
        )

        if self.residual:
            self.res_scale = nn.Parameter(torch.zeros(1))

    def get_grid(self) -> torch.Tensor:

        steps = F.softplus(self.grid_steps_log) 

        grid_offsets = torch.cumsum(steps, dim=1)

        grid = torch.cat([self.grid_start, self.grid_start + grid_offsets], dim=1)
        return grid

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:

        grid = self.get_grid()

        x = x.unsqueeze(-1) 

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):

            denom1 = grid[:, k:-1] - grid[:, :-(k + 1)]
            left = (x - grid[:, :-(k + 1)]) / torch.where(denom1 == 0, torch.ones_like(denom1), denom1)

            denom2 = grid[:, k + 1:] - grid[:, 1:-k]
            right = (grid[:, k + 1:] - x) / torch.where(denom2 == 0, torch.ones_like(denom2), denom2)

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        spline_basis = self.b_spline(x) 

        spline_out = torch.einsum("oic,bic->bo", self.coeffs, spline_basis)

        base_out = F.linear(self.base_activation(x), self.base_weight)

        y = spline_out + base_out

        if self.residual:
            y = y + self.res_scale * x

        y = torch.tanh(y)

        return y

class PIDBSN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)
        grid_size = getattr(config, 'dbsn_grid_size', 5)
        spline_order = getattr(config, 'dbsn_spline_order', 3)

        self.layers = nn.ModuleList()

        self.layers.append(BSplineLayer(in_dim, hidden_dim, grid_size, spline_order, residual=False))

        for _ in range(n_layers - 1):
            self.layers.append(BSplineLayer(hidden_dim, hidden_dim, grid_size, spline_order, residual=True))

        self.head = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = x 
        for layer in self.layers:
            h = layer(h)
        return self.head(h)