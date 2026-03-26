import math
import torch
import torch.nn as nn

class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.w0 = w0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:

                bound = 1 / math.sqrt(self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class FourierNet(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden: int,
        n_blocks: int,
        n_fourier: int,
        freq_min: float = 0.5,
        freq_max: float = 5.0,  
        trainable_freqs: bool = False,
        corner_enrichment=None,
        init_freqs: torch.Tensor | None = None,
    ):
        super().__init__()

        self.corner_enrichment = corner_enrichment
        corner_dim = corner_enrichment.out_dim if corner_enrichment else 0

        if init_freqs is not None:
            freqs = init_freqs
        else:
            freq_max = min(freq_max, 5.0)  
            freqs = torch.linspace(math.log(freq_min), math.log(freq_max), n_fourier)

        self.w_x = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)
        self.w_y = nn.Parameter(freqs.clone(), requires_grad=trainable_freqs)
        self.n_fourier = n_fourier

        self.fourier_dim = 2 + 8 * n_fourier
        self.norm = math.sqrt(self.fourier_dim / 2)  

        in_dim = self.fourier_dim + corner_dim

        layers = [SirenLayer(in_dim, hidden, w0=30.0, is_first=True)]
        for _ in range(max(n_blocks - 1, 1)):
            layers.append(SirenLayer(hidden, hidden, w0=1.0))  
        self.siren = nn.Sequential(*layers)

        self.head = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.head.weight)

    def fourier(self, xy):
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

    def forward(self, xy):
        cf = self.corner_enrichment(xy) if self.corner_enrichment else None
        f = self.fourier(xy)
        if cf is not None:
            f = torch.cat([f, cf], dim=-1)
        h = self.siren(f)
        return self.head(h)