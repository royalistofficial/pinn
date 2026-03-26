import math
import torch
import torch.nn as nn

class ChebyshevBasis(nn.Module):
    def __init__(self, degree: int = 5):
        super().__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(torch.ones(degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape
        device = x.device
        dtype = x.dtype

        T = torch.ones(batch_shape + (self.degree + 1,), device=device, dtype=dtype)
        if self.degree > 0:
            T[..., 1] = x.squeeze(-1) if x.ndim > 1 else x

        for k in range(2, self.degree + 1):
            T[..., k] = 2 * (x.squeeze(-1)) * T[..., k-1] - T[..., k-2]

        return torch.sum(T * self.coeffs, dim=-1, keepdim=True)  

class ChebyKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, degree: int = 5, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.residual = residual

        self.coeffs = nn.Parameter(torch.ones(out_dim, in_dim, degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        x_exp = x.unsqueeze(-1)  

        T = [torch.ones_like(x_exp)]  
        if self.degree > 0:
            T.append(x_exp)  
        for k in range(2, self.degree + 1):
            T.append(2 * x_exp * T[-1] - T[-2])

        T_stack = torch.cat(T, dim=-1)
        T_stack = T_stack.reshape(batch, self.in_dim, self.degree + 1)

        y = torch.einsum('oij,bij->bo', self.coeffs, T_stack)

        if self.residual and x.shape[-1] == y.shape[-1]:
            y = y + x

        return torch.tanh(y)

class ScaledCPIKAN(nn.Module):
    def __init__(self, out_dim: int, **config):
        super().__init__()
        self.hidden = config.get("hidden", 64)
        self.n_layers = config.get("n_layers", 4)
        self.degree = config.get("degree", 5)
        self.n_fourier = config.get("n_fourier", 0)
        self.freq_min = config.get("freq_min", 0.5)
        self.freq_max = config.get("freq_max", 10.0)
        self.trainable_freqs = config.get("trainable_freqs", False)
        self.corner_enrichment = config.get("corner_enrichment", None)   

        corner_dim = self.corner_enrichment.out_dim if self.corner_enrichment is not None else 0
        self.use_fourier = self.n_fourier > 0

        if self.use_fourier:
            freqs = torch.linspace(math.log(self.freq_min), math.log(self.freq_max), self.n_fourier)
            self.w_x = nn.Parameter(freqs.clone(), requires_grad=self.trainable_freqs)
            self.w_y = nn.Parameter(freqs.clone(), requires_grad=self.trainable_freqs)
            self.fourier_dim = 2 + 8 * self.n_fourier
            self.norm = math.sqrt(self.fourier_dim / 2.0)
        else:
            self.fourier_dim = 0

        in_dim = 2 + self.fourier_dim + corner_dim

        self.layers = nn.ModuleList()
        self.layers.append(ChebyKANLayer(in_dim, self.hidden, self.degree, residual=False))
        for _ in range(self.n_layers - 1):
            self.layers.append(ChebyKANLayer(self.hidden, self.hidden, self.degree, residual=True))

        self.head = nn.Linear(self.hidden, out_dim)
        nn.init.orthogonal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def fourier(self, xy: torch.Tensor) -> torch.Tensor:
        if not self.use_fourier:
            return torch.empty((xy.shape[0], 0), device=xy.device)

        Bx, By = self.w_x.exp(), self.w_y.exp()
        px = math.pi * xy[:, 0:1] * Bx
        py = math.pi * xy[:, 1:2] * By

        stacked = torch.cat([px, py, px + py, px - py], dim=-1)  
        sin_cos = torch.cat([stacked.sin(), stacked.cos()], dim=-1)
        return torch.cat([xy, sin_cos], dim=-1) / self.norm

    def forward(self, xy: torch.Tensor):
        cf = self.corner_enrichment(xy) if self.corner_enrichment is not None else None
        f = self.fourier(xy)

        x = xy
        if f.numel() > 0:
            x = torch.cat([x, f], dim=-1)
        if cf is not None:
            x = torch.cat([x, cf], dim=-1)

        h = x
        for layer in self.layers:
            h = layer(h)

        return self.head(h)