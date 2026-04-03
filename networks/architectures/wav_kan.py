import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WavKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_wavelets: int = 5, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_wavelets = num_wavelets
        self.residual = residual and (in_dim == out_dim)

        self.translation = nn.Parameter(torch.zeros(in_dim, num_wavelets))
        self.scale = nn.Parameter(torch.zeros(in_dim, num_wavelets))

        self.weights = nn.Parameter(torch.empty(out_dim, in_dim * num_wavelets))
        nn.init.normal_(self.weights, std=0.1)

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_activation = nn.SiLU()

        if self.residual:
            self.res_scale = nn.Parameter(torch.zeros(1))

    def mexican_hat(self, x):
        return (1.0 - x ** 2) * torch.exp(-0.5 * x ** 2)

    def forward(self, x):
        x_u = x.unsqueeze(-1)

        scale = F.softplus(self.scale) + 1e-4
        x_norm = (x_u - self.translation) / scale

        wav = self.mexican_hat(x_norm)

        B = wav.shape[0]

        wav_flat = wav.view(B, -1)
        y_wav = F.linear(wav_flat, self.weights)

        y_base = F.linear(self.base_activation(x), self.base_weight)

        y = y_wav + y_base

        # [ИСПРАВЛЕНИЕ] Мягкое сложение с входом
        if self.residual:
            y = y + self.res_scale * x

        return y

class WavKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)
        num_wavelets = getattr(config, 'num_wavelets', 5)

        self.layers = nn.ModuleList()

        self.layers.append(WavKANLayer(in_dim, hidden_dim, num_wavelets, residual=False))
        for _ in range(n_layers - 1):
            self.layers.append(WavKANLayer(hidden_dim, hidden_dim, num_wavelets, residual=True))

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)