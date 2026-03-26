from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from networks.activations import make_activation, CnActivation

class NTKLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1.0 / math.sqrt(in_features)

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, scale={self.scale:.4f}")

def sn_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    return spectral_norm(nn.Linear(in_f, out_f, bias=bias))

def make_linear(in_f: int, out_f: int, bias: bool = True,
                use_ntk_param: bool = True) -> nn.Module:
    if use_ntk_param:
        return NTKLinear(in_f, out_f, bias=bias)
    else:
        return sn_linear(in_f, out_f, bias=bias)

class ResBlock(nn.Module):
    def __init__(self, hidden: int, expansion: int = 1,
                 activation: str = "gelu", use_ntk_param: bool = True):
        super().__init__()
        if expansion == 1:
            self.net = nn.Sequential(
                make_linear(hidden, hidden, use_ntk_param=use_ntk_param),
                make_activation(activation),
                make_linear(hidden, hidden, use_ntk_param=use_ntk_param),
            )
        else:
            mid = hidden * expansion
            self.net = nn.Sequential(
                make_linear(hidden, mid, use_ntk_param=use_ntk_param),
                make_activation(activation),
                make_linear(mid, hidden, use_ntk_param=use_ntk_param),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class ScaleBranch(nn.Module):
    def __init__(self, f_lo: float, f_hi: float, n_fourier: int,
                 hidden: int, n_blocks: int,
                 expansion: int = 1, activation: str = "gelu",
                 use_ntk_param: bool = True,
                 trainable_freqs: bool = True, extra_in_dim: int = 0,
                 init_freqs: torch.Tensor | None = None):
        super().__init__()

        if init_freqs is not None and len(init_freqs) == n_fourier:

            self.w_x = nn.Parameter(init_freqs.clone(), requires_grad=trainable_freqs)
            self.w_y = nn.Parameter(init_freqs.clone(), requires_grad=trainable_freqs)
        else:
            target_f = torch.linspace(math.log(f_lo), math.log(f_hi), n_fourier)
            self.w_x = nn.Parameter(target_f.clone(), requires_grad=trainable_freqs)
            self.w_y = nn.Parameter(target_f.clone(), requires_grad=trainable_freqs)

        in_f = 2 + 8 * n_fourier + extra_in_dim
        self.proj = nn.Sequential(
            make_linear(in_f, hidden, use_ntk_param=use_ntk_param),
            make_activation(activation),
        )
        self.blocks = nn.Sequential(*[
            ResBlock(hidden, expansion, activation, use_ntk_param)
            for _ in range(max(n_blocks, 1))
        ])
        self._fourier_dim = 2 + 8 * n_fourier

    def _fourier(self, xy: torch.Tensor) -> torch.Tensor:
        Bx = self.w_x.exp()
        By = self.w_y.exp()
        px = math.pi * xy[:, 0:1] * Bx
        py = math.pi * xy[:, 1:2] * By
        return torch.cat([
            xy, px.sin(), px.cos(), py.sin(), py.cos(),
            (px + py).sin(), (px + py).cos(),
            (px - py).sin(), (px - py).cos(),
        ], dim=-1)

    def forward(self, xy: torch.Tensor,
                extra_features: torch.Tensor | None = None):
        f = self._fourier(xy)
        if extra_features is not None:
            f_full = torch.cat([f, extra_features], dim=-1)
        else:
            f_full = f
        h = self.proj(f_full)
        h = self.blocks(h)
        return h, f

class FourierNet(nn.Module):
    def __init__(self, out_dim: int, hidden: int, n_blocks: int,
                 n_fourier: int, n_scales: int,
                 freq_min: float, freq_max: float, expansion: int,
                 shortcut: bool, activation: str = "gelu",
                 use_ntk_param: bool = True,
                 trainable_freqs: bool = True,
                 corner_enrichment=None,
                 init_freqs: torch.Tensor | None = None):
        super().__init__()
        self.corner_enrichment = corner_enrichment
        corner_dim = corner_enrichment.out_dim if corner_enrichment is not None else 0

        log_bounds = torch.linspace(math.log(freq_min), math.log(freq_max), n_scales + 1)
        base_nb = n_blocks // n_scales
        extra = n_blocks % n_scales

        branches = []
        for i in range(n_scales):
            f_lo = math.exp(log_bounds[i].item())
            f_hi = math.exp(log_bounds[i + 1].item())
            nb = base_nb + (1 if i < extra else 0)
            branches.append(ScaleBranch(
                f_lo=f_lo, f_hi=f_hi, n_fourier=n_fourier,
                hidden=hidden, n_blocks=nb,
                expansion=expansion, activation=activation,
                use_ntk_param=use_ntk_param,
                trainable_freqs=trainable_freqs,
                extra_in_dim=corner_dim,
                init_freqs=init_freqs,
            ))
        self.branches = nn.ModuleList(branches)

        self.head = nn.Linear(hidden, out_dim)

        if shortcut:
            fourier_total = (2 + 8 * n_fourier) * n_scales + corner_dim
            self.shortcut_layer = nn.Linear(fourier_total, out_dim, bias=False)
        else:
            self.shortcut_layer = None

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] {total_params:,} trainable params, "
              f"activation={activation}, ntk_param={use_ntk_param}")

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        cf = self.corner_enrichment(xy) if self.corner_enrichment is not None else None

        outputs, fourier_feats = [], []
        for branch in self.branches:
            h, f = branch(xy, extra_features=cf)
            outputs.append(h)
            fourier_feats.append(f)

        combined = torch.stack(outputs, dim=0).sum(dim=0)
        out = self.head(combined)

        if self.shortcut_layer is not None:
            all_f = torch.cat(fourier_feats, dim=-1)
            if cf is not None:
                all_f = torch.cat([all_f, cf], dim=-1)
            out = out + self.shortcut_layer(all_f)

        return out

