import torch
import torch.nn as nn

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    elif name == "relu":
        return nn.ReLU()
    else:
        return nn.Tanh()  

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        in_dim = getattr(config, 'in_dim', 2)
        out_dim = getattr(config, 'out_dim', 1)
        hidden_dim = getattr(config, 'hidden_dim', 64)
        n_layers = getattr(config, 'n_layers', 4)
        act_name = getattr(config, 'activation', 'tanh')

        activation = _get_activation(act_name)

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):

                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)