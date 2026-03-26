import math
import torch
import torch.nn as nn

class FourierNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_fourier = getattr(config, 'fourier_features', 256)
        freq_min = getattr(config, 'freq_min', 0.5)
        freq_max = getattr(config, 'freq_max', 5.0)
        trainable_freqs = getattr(config, 'trainable_freqs', False)

        freqs = torch.linspace(math.log(freq_min), math.log(freq_max), self.n_fourier)
        self.w_x = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)
        self.w_y = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)

        self.fourier_dim = 2 + 8 * self.n_fourier
        self.norm = math.sqrt(self.fourier_dim / 2)  

        layers = []
        in_dim = self.fourier_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.Tanh()) 
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, config.out_dim))
        self.net = nn.Sequential(*layers)

    def fourier_mapping(self, xy: torch.Tensor) -> torch.Tensor:
        Bx = self.w_x.exp()
        By = self.w_y.exp()

        px = math.pi * xy[:, 0:1] * Bx
        py = math.pi * xy[:, 1:2] * By

        raw = torch.cat([
            xy,
            px.sin(), px.cos(),
            py.sin(), py.cos(),
            (px + py).sin(), (px + py).cos(),
            (px - py).sin(), (px - py).cos(),
        ], dim=-1)

        return raw / self.norm

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        f = self.fourier_mapping(xy)
        return self.net(f)