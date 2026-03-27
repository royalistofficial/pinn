import math
import torch
import torch.nn as nn

class FourierNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = getattr(config, 'in_dim', 2)
        self.n_fourier = getattr(config, 'fourier_features', 256)
        freq_min = getattr(config, 'freq_min', 0.5)
        freq_max = getattr(config, 'freq_max', 5.0)
        trainable_freqs = getattr(config, 'trainable_freqs', False)

        freqs = torch.linspace(math.log(freq_min), math.log(freq_max), self.n_fourier)

        self.is_2d = (self.in_dim == 2)
        if self.is_2d:
            self.w_x = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)
            self.w_y = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)
            self.fourier_dim = 2 + 8 * self.n_fourier
        else:

            self.B = nn.Parameter(
                torch.stack([freqs.clone() for _ in range(self.in_dim)]),
                requires_grad=trainable_freqs
            )
            self.fourier_dim = self.in_dim + 2 * self.in_dim * self.n_fourier

        self.norm = math.sqrt(self.fourier_dim / 2)  

        layers = []
        in_dim = self.fourier_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.Tanh()) 
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, config.out_dim))
        self.net = nn.Sequential(*layers)

    def fourier_mapping(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_2d and x.shape[-1] == 2:
            Bx = self.w_x.exp()
            By = self.w_y.exp()

            px = math.pi * x[:, 0:1] * Bx
            py = math.pi * x[:, 1:2] * By

            raw = torch.cat([
                x,
                px.sin(), px.cos(),
                py.sin(), py.cos(),
                (px + py).sin(), (px + py).cos(),
                (px - py).sin(), (px - py).cos(),
            ], dim=-1)
        else:
            B_exp = self.B.exp() 

            p = math.pi * x.unsqueeze(-1) * B_exp
            p = p.view(x.shape[0], -1) 

            raw = torch.cat([x, p.sin(), p.cos()], dim=-1)

        return raw / self.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fourier_mapping(x)
        return self.net(f)