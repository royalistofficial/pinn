import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WavKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_wavelets: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_wavelets = num_wavelets

        self.translation = nn.Parameter(torch.zeros(out_dim, in_dim, num_wavelets))
        self.scale = nn.Parameter(torch.zeros(out_dim, in_dim, num_wavelets))
        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim, num_wavelets))
        nn.init.normal_(self.coeffs, std=0.1)

    def mexican_hat(self, x):
        return (1.0 - x ** 2) * torch.exp(-0.5 * x ** 2)

    def forward(self, x):
        # x: (B, in_dim)
        B = x.shape[0]

        x_exp = x.unsqueeze(1).unsqueeze(3)

        trans = self.translation.unsqueeze(0)
        scale = F.softplus(self.scale.unsqueeze(0)) + 1e-4

        x_norm = (x_exp - trans) / scale

        wav = self.mexican_hat(x_norm)

        coeffs = self.coeffs.unsqueeze(0)
        y = torch.sum(wav * coeffs, dim=(2, 3))  # (B, out_dim)

        return y


class WavKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 8)
        n_layers = getattr(config, 'n_layers', 2)
        num_wavelets = getattr(config, 'num_wavelets', 5)

        self.layers = nn.ModuleList()

        self.layers.append(WavKANLayer(in_dim, hidden_dim, num_wavelets))
        for _ in range(n_layers - 1):
            self.layers.append(WavKANLayer(hidden_dim, hidden_dim, num_wavelets))

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)