import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.residual = residual and (in_dim == out_dim)

        self.n_coeffs = grid_size + spline_order

        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim * self.n_coeffs))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1)

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_activation = nn.SiLU()

        step = 2.0 / grid_size
        n_knots = grid_size + 2 * spline_order + 1

        self.grid_steps_log = nn.Parameter(torch.log(torch.ones(in_dim, n_knots - 1) * step))
        self.grid_start = nn.Parameter(torch.ones(in_dim, 1) * (-1.0 - step * spline_order))

        if self.residual:
            self.res_scale = nn.Parameter(torch.zeros(1))

    def get_grid(self):
        steps = F.softplus(self.grid_steps_log)
        grid_offsets = torch.cumsum(steps, dim=1)
        return torch.cat([self.grid_start, self.grid_start + grid_offsets], dim=1)

    def b_spline(self, x, grid):
        x = x.unsqueeze(-1)
        left = grid[:, :-1]
        right = grid[:, 1:]

        bases = ((x >= left) & (x < right)).to(x.dtype)

        eps = 1e-8
        denom = grid[:, 1:] - grid[:, :-1]

        for k in range(self.spline_order):
            left_den = denom[:, :-1 - k]
            right_den = denom[:, 1 + k:]

            left_num = x - grid[:, :-(2 + k)]
            right_num = grid[:, 2 + k:] - x

            left = left_num / (left_den + eps)
            right = right_num / (right_den + eps)

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases

    def forward(self, x):
        grid = self.get_grid()

        spline_basis = self.b_spline(x, grid)

        B = spline_basis.shape[0]
        
        spline_flat = spline_basis.view(B, -1)
        spline_out = F.linear(spline_flat, self.coeffs)

        base_out = F.linear(self.base_activation(x), self.base_weight)

        y = spline_out + base_out

        if self.residual:
            y = y + self.res_scale * x

        return torch.tanh(y)


class BSplineKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)

        grid_size = getattr(config, 'kan_grid_size', 5)
        spline_order = getattr(config, 'kan_spline_order', 3)

        self.layers = nn.ModuleList()

        self.layers.append(BSplineKANLayer(in_dim, hidden, grid_size, spline_order, residual=False))

        for _ in range(n_layers - 1):
            self.layers.append(BSplineKANLayer(hidden, hidden, grid_size, spline_order, residual=True))

        self.head = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.head(h)