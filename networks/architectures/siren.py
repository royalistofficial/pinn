import math
import torch
import torch.nn as nn

class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float = 30.0, is_first: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential()
        self.w0 = getattr(config, 'siren_w0', 30.0)

        in_dim = config.in_dim
        for i in range(config.n_layers):
            is_first = (i == 0)
            w0 = self.w0 if is_first else 1.0 

            self.net.add_module(f"siren_{i}", SirenLayer(in_dim, config.hidden_dim, w0=w0, is_first=is_first))
            in_dim = config.hidden_dim

        final_layer = nn.Linear(in_dim, config.out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / in_dim) / 1.0
            final_layer.weight.uniform_(-bound, bound)

        self.net.add_module("final", final_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)