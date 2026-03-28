from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from ntk import (
    compute_jacobian, 
    compute_pde_jacobian, 
    compute_bc_jacobian,
    compute_ntk_from_jacobian,
    compute_spectrum,
    get_all_metrics,
    plot_ntk_master_dashboard,
    plot_ntk_evolution  
)

@dataclass
class NTKResult:
    epoch: int
    eigenvalues_KL: np.ndarray
    metrics_KL: Dict[str, Any]
    eigenvalues_K: np.ndarray
    metrics_K: Dict[str, Any]
    metrics_D: Optional[Dict[str, Any]] = None
    metrics_N: Optional[Dict[str, Any]] = None
    metrics_tot: Optional[Dict[str, Any]] = None
    n_interior: int = 0
    n_dirichlet: int = 0
    n_neumann: int = 0

class NTKAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "data/ntk_plots",
        n_interior: int = 64,
        n_boundary: int = 32,
        learning_rate: float = 1e-3,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.learning_rate = learning_rate
        self.logger = logger or print
        self.history: List[NTKResult] = []
        os.makedirs(output_dir, exist_ok=True)

    def analyze(
        self,
        epoch: int,
        X_interior: torch.Tensor,
        X_boundary: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        bc_mask: Optional[torch.Tensor] = None,
    ) -> NTKResult:
        device = next(self.model.parameters()).device
        self.logger(f"[NTK] Строгий спектральный анализ для эпохи {epoch}...")

        X_in = self._subsample(X_interior.to(device), self.n_interior)
        n_in = len(X_in)

        J_in = compute_jacobian(self.model, X_in)
        J_L_in = compute_pde_jacobian(self.model, X_in)

        K_in = compute_ntk_from_jacobian(J_in).cpu().numpy()
        K_L_in = compute_ntk_from_jacobian(J_L_in).cpu().numpy()

        eig_K = compute_spectrum(K_in)
        eig_KL = compute_spectrum(K_L_in)

        metrics_K = get_all_metrics(K_in, eig_K)
        metrics_KL = get_all_metrics(K_L_in, eig_KL)

        J_total_list = [J_L_in]

        components = {
            "Внутренние (K)": {"eigenvalues": eig_K, "metrics": metrics_K, "color": "#2563EB", "marker": "o"},
            "PDE (K_L)": {"eigenvalues": eig_KL, "metrics": metrics_KL, "color": "#DC2626", "marker": "s"}
        }

        metrics_D, metrics_N = None, None
        n_D, n_N = 0, 0

        if X_boundary is not None and len(X_boundary) > 0 and bc_mask is not None:
            X_bd = self._subsample(X_boundary.to(device), self.n_boundary)
            normals_sub = self._subsample(normals.to(device), self.n_boundary) if normals is not None else None
            bc_mask_sub = self._subsample(bc_mask.to(device), self.n_boundary)

            mask_flat = bc_mask_sub.squeeze(-1) if bc_mask_sub.dim() > 1 else bc_mask_sub
            idx_D = (mask_flat > 0.5).nonzero(as_tuple=True)[0]
            idx_N = (mask_flat <= 0.5).nonzero(as_tuple=True)[0]

            xy_D = X_bd[idx_D]
            xy_N = X_bd[idx_N]
            n_D, n_N = len(xy_D), len(xy_N)

            if n_D > 0:
                normals_D = normals_sub[idx_D] if normals_sub is not None else None
                J_D = compute_bc_jacobian(self.model, xy_D, normals_D, "dirichlet")
                J_total_list.append(J_D)

                K_D = compute_ntk_from_jacobian(J_D).cpu().numpy()
                eig_D = compute_spectrum(K_D)
                metrics_D = get_all_metrics(K_D, eig_D)
                components["Дирихле (K_D)"] = {"eigenvalues": eig_D, "metrics": metrics_D, "color": "#059669", "marker": "^"}

            if n_N > 0:
                normals_N = normals_sub[idx_N] if normals_sub is not None else None
                J_N = compute_bc_jacobian(self.model, xy_N, normals_N, "neumann")
                J_total_list.append(J_N)

                K_N = compute_ntk_from_jacobian(J_N).cpu().numpy()
                eig_N = compute_spectrum(K_N)
                metrics_N = get_all_metrics(K_N, eig_N)
                components["Нейман (K_N)"] = {"eigenvalues": eig_N, "metrics": metrics_N, "color": "#D97706", "marker": "v"}

        J_total = torch.cat(J_total_list, dim=0)
        K_total = compute_ntk_from_jacobian(J_total).cpu().numpy()
        eig_tot = compute_spectrum(K_total)
        metrics_tot = get_all_metrics(K_total, eig_tot)
        components["Полная (K_tot)"] = {"eigenvalues": eig_tot, "metrics": metrics_tot, "color": "#8B5CF6", "marker": "D"}

        result = NTKResult(
            epoch=epoch,
            eigenvalues_KL=eig_KL,
            metrics_KL=metrics_KL,
            eigenvalues_K=eig_K,
            metrics_K=metrics_K,
            metrics_D=metrics_D,
            metrics_N=metrics_N,
            metrics_tot=metrics_tot,
            n_interior=n_in,
            n_dirichlet=n_D,
            n_neumann=n_N
        )

        self.history.append(result)

        self.logger(
            f"[NTK] Epoch {epoch} Спектральная сводка:\n"
            f"       K_L (PDE): κ={metrics_KL['condition_number']:.2e}, rank={metrics_KL['effective_rank']:.1f}\n"
            f"       K_tot (Все): κ={metrics_tot['condition_number']:.2e}, rank={metrics_tot['effective_rank']:.1f}"
        )

        dash_path = plot_ntk_master_dashboard(
            epoch=epoch,
            components=components,
            output_dir=self.output_dir
        )
        self.logger(f"[NTK] Спектральный Дашборд сохранен: {dash_path}")

        return result

    @staticmethod
    def _subsample(X: torch.Tensor, n: int) -> torch.Tensor:
        if len(X) <= n:
            return X
        return X[torch.linspace(0, len(X) - 1, n, device=X.device).long()]

    def plot_evolution(self) -> None:
        if len(self.history) < 2:
            self.logger("[NTK] Недостаточно данных для графика эволюции (нужно минимум 2 замера).")
            return

        self.logger("[NTK] Генерация дашборда эволюции NTK...")
        path = plot_ntk_evolution(self.history, self.output_dir)
        if path:
            self.logger(f"[NTK] Дашборд эволюции сохранен: {path}")