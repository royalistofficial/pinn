from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from networks.activations import CnActivation

def sn_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    return spectral_norm(nn.Linear(in_f, out_f, bias=bias))

class CnResBlock(nn.Module):
    def __init__(self, hidden: int, n_smooth: int, expansion: int, alpha: float = 0.01):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        if expansion == 1:
            self.net = nn.Sequential(
                sn_linear(hidden, hidden),
                CnActivation(n=n_smooth, alpha=alpha),
                sn_linear(hidden, hidden),
            )
        else:
            mid = hidden * expansion
            self.net = nn.Sequential(
                sn_linear(hidden, mid),
                CnActivation(n=n_smooth, alpha=alpha),
                sn_linear(mid, hidden),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))

class ScaleBranch(nn.Module):
    def __init__(self, f_lo: float, f_hi: float, n_fourier: int,
                 hidden: int, n_blocks: int, n_smooth: int,
                 expansion: int, alpha: float = 0.01,
                 trainable_freqs: bool = True, extra_in_dim: int = 0):
        super().__init__()
        target_f =torch.linspace(math.log(f_lo), math.log(f_hi), n_fourier)
        self.w_x = nn.Parameter(target_f.clone(), requires_grad=trainable_freqs)
        self.w_y = nn.Parameter(target_f.clone(), requires_grad=trainable_freqs)

        in_f = 2 + 8 * n_fourier + extra_in_dim
        self.proj = nn.Sequential(
            sn_linear(in_f, hidden),
            CnActivation(n=n_smooth, alpha=alpha),
        )
        self.blocks = nn.Sequential(*[
            CnResBlock(hidden, n_smooth, expansion, alpha)
            for _ in range(max(n_blocks, 1))
        ])
        self.norm = nn.LayerNorm(hidden)
        self._fourier_dim = 2 + 8 * n_fourier

    def _fourier(self, xy: torch.Tensor) -> torch.Tensor:
        Bx = self.w_x.exp()
        By = self.w_y.exp()
        px = math.pi * xy[:, 0:1] * Bx
        py = math.pi * xy[:, 1:2] * By
        return torch.cat([
            xy, px.sin(), px.cos(), py.sin(), py.cos(),
            (px+py).sin(), (px+py).cos(), (px-py).sin(), (px-py).cos(),
        ], dim=-1)

    def forward(self, xy, extra_features=None):
        f = self._fourier(xy)
        if extra_features is not None:
            f_full = torch.cat([f, extra_features], dim=-1)
        else:
            f_full = f
        h = self.proj(f_full)
        h = self.blocks(h)
        h = self.norm(h)
        return h, f

class FourierCnNet(nn.Module):
    def __init__(self, out_dim: int, n_smooth: int,
                 hidden: int, n_blocks: int, n_fourier: int, n_scales: int,
                 freq_min: float, freq_max: float, expansion: int,
                 shortcut: bool, alpha: float = 0.01,
                 trainable_freqs: bool = True,
                 corner_enrichment=None):
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
                hidden=hidden, n_blocks=nb, n_smooth=n_smooth,
                expansion=expansion, alpha=alpha,
                trainable_freqs=trainable_freqs, extra_in_dim=corner_dim,
            ))
        self.branches = nn.ModuleList(branches)

        self.head = nn.Linear(hidden, out_dim)

        if shortcut:
            fourier_total = (2 + 8 * n_fourier) * n_scales + corner_dim
            self.shortcut_layer = nn.Linear(fourier_total, out_dim, bias=False)
        else:
            self.shortcut_layer = None

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] {total_params:,} trainable params.")

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
