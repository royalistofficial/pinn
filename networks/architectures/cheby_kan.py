import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # Импортируем F для использования F.linear

class ChebyKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, degree: int = 5, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree

        self.residual = residual and (in_dim == out_dim)

        std = 1.0 / math.sqrt(in_dim * (degree + 1))
        
        self.coeffs = nn.Parameter(torch.empty(out_dim, in_dim * (degree + 1)))
        nn.init.normal_(self.coeffs, mean=0.0, std=std)

        self.register_buffer("_scale", torch.tensor(1.0 / math.sqrt(in_dim)))

        if self.residual:
            self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        xc = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)   

        T = [torch.ones_like(xc), xc]
        for k in range(2, self.degree + 1):
            T.append(2.0 * xc * T[-1] - T[-2])

        T_stack = torch.stack(T, dim=-1).view(batch, -1)   
        y = F.linear(T_stack, self.coeffs)
        
        y = y * self._scale

        if self.residual:
            y = y + self.res_scale * x

        y = torch.tanh(y)

        return y

class ScaledCPIKAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)
        degree = getattr(config, 'kan_degree', 5)

        self.layers = nn.ModuleList()
        self.layers.append(ChebyKANLayer(in_dim, hidden, degree, residual=False))

        for _ in range(n_layers - 1):
            self.layers.append(ChebyKANLayer(hidden, hidden, degree, residual=True))

        self.head = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = 0.95 * x  
        for layer in self.layers:
            h = layer(h)
        return self.head(h)