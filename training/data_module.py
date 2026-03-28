from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import torch
from torch.utils.data import DataLoader, TensorDataset
from geometry.quadrature import QuadratureData
from problems.solutions import AnalyticalSolution

@dataclass
class TrainingSample:
    quad: QuadratureData
    u_exact: torch.Tensor
    f_in: torch.Tensor
    g_D: torch.Tensor
    g_N: torch.Tensor

def prepare_sample(quad: QuadratureData, solution: AnalyticalSolution) -> TrainingSample:
    return TrainingSample(
        quad=quad,
        u_exact=solution.eval(quad.xy_in),
        f_in=solution.rhs(quad.xy_in),
        g_D=solution.eval(quad.xy_bd),
        g_N=solution.neumann_data(quad.xy_bd, quad.normals),
    )

class DataModule:
    def __init__(self, sample: TrainingSample, batch_size: int):
        self.sample = sample
        self.batch_size = batch_size
        q = sample.quad

        idx_in = torch.arange(len(q.xy_in), device=q.xy_in.device)
        self.in_dataset = TensorDataset(
            q.xy_in, sample.f_in, q.vol_w, q.tri_indices, idx_in)
        self.N_in = len(self.in_dataset)

        idx_bd = torch.arange(len(q.xy_bd), device=q.xy_bd.device)
        self.bd_dataset = TensorDataset(
            q.xy_bd, q.normals, sample.g_D, sample.g_N,
            q.bc_mask, q.surf_w, q.tri_indices_bd, idx_bd)
        self.N_bd = len(self.bd_dataset)

        self.in_loader = DataLoader(self.in_dataset, batch_size=batch_size, shuffle=True)
        self.bd_loader = DataLoader(self.bd_dataset, batch_size=batch_size, shuffle=True)

    def boundary_iter(self) -> Iterator:
        while True:
            for b in self.bd_loader:
                yield b