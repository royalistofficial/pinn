import math
import torch
import torch.nn as nn
import torch.nn.functional as F  

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

        xc = x.clamp(-0.9999, 0.9999)   

        theta = torch.acos(xc) # Размерность: (batch, in_dim)
        
        T_list = [torch.cos(k * theta) for k in range(self.degree + 1)]
        T_stack = torch.stack(T_list, dim=-1).view(batch, -1)   
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 0.95
        for layer in self.layers:
            x = layer(x)
        return self.head(x)