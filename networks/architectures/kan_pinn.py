import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.n_coeffs = grid_size + spline_order

        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim * self.n_coeffs))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1)

        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.constant_(self.spline_weight, 1.0)

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.base_weight)

        self.base_activation = nn.SiLU()

        step = 2.0 / grid_size
        n_knots = grid_size + 2 * spline_order + 1
        init_val = math.log(math.exp(step) - 1.0)
        self.grid_steps_log = nn.Parameter(torch.ones(in_dim, n_knots - 1) * init_val)
        self.grid_start = nn.Parameter(torch.ones(in_dim, 1) * (-1.0 - spline_order * step))

    def get_grid(self):
        steps = F.softplus(self.grid_steps_log)
        grid = torch.cat([self.grid_start, steps], dim=-1)
        grid = torch.cumsum(grid, dim=-1)
        return grid

    def b_spline(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x_u = x.unsqueeze(-1)
        grid_exp = grid.unsqueeze(0)

        bases = ((x_u >= grid_exp[:, :, :-1]) & (x_u < grid_exp[:, :, 1:])).to(x.dtype)

        eps = 1e-8

        for d in range(1, self.spline_order + 1):
            left_num = x_u - grid_exp[:, :, :-(d + 1)]
            left_den = grid_exp[:, :, d:-1] - grid_exp[:, :, :-(d + 1)] + eps
            left = left_num / left_den

            right_num = grid_exp[:, :, d + 1:] - x_u
            right_den = grid_exp[:, :, d + 1:] - grid_exp[:, :, 1:-d] + eps
            right = right_num / right_den

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases

    def forward(self, x):
        grid = self.get_grid()
        spline_basis = self.b_spline(x, grid)  # B x in_dim x n_coeffs

        # Коэффициенты переформатированы: out x in x n_coeffs
        coeffs_reshaped = self.coeffs.view(self.out_dim, self.in_dim, self.n_coeffs)

        spline_per_conn = torch.einsum('b i k, o i k -> b o i', spline_basis, coeffs_reshaped)

        # Применяем w_s
        spline_per_conn = spline_per_conn * self.spline_weight.unsqueeze(0)

        # Суммируем по входным измерениям
        spline_out = spline_per_conn.sum(dim=-1)  # B x out_dim

        # Базовая часть
        base_out = F.linear(self.base_activation(x), self.base_weight)

        y = spline_out + base_out
        return y


class BSplineKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 15)
        n_layers = getattr(config, 'n_layers', 3)

        grid_size = getattr(config, 'kan_grid_size', 10)
        spline_order = getattr(config, 'kan_spline_order', 3)

        self.layers = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            self.layers.append(
                BSplineKANLayer(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x