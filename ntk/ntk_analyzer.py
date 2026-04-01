from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from config import NTK_SHOW_BOUNDARY_PAIRS
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

    def _build_dashboard_components(self, J_in, J_D, J_N, w_pde=1.0, w_dir=1.0, w_neu=1.0) -> dict:
        components = {}

        def is_valid(J):
            return J is not None and J.shape[0] > 0

        J_in_scaled = J_in * math.sqrt(w_pde) if is_valid(J_in) else None
        J_D_scaled = J_D * math.sqrt(w_dir) if is_valid(J_D) else None
        J_N_scaled = J_N * math.sqrt(w_neu) if is_valid(J_N) else None

        def add_comp(name: str, J: Optional[torch.Tensor], color: str, marker: str):
            if not is_valid(J): 
                return
            K = compute_ntk_from_jacobian(J).cpu().numpy()
            eig = compute_spectrum(K)
            if len(eig) > 0:
                metrics = get_all_metrics(K, eig)
                components[name] = {
                    "eigenvalues": eig, 
                    "metrics": metrics, 
                    "color": color, 
                    "marker": marker
                }

        add_comp("Внутренние", J_in, "#2563EB", "o")
        add_comp("Дирихле", J_D, "#059669", "^")
        add_comp("Нейман", J_N, "#D97706", "v")

        valid_comps = [j for j in [J_in, J_D, J_N] if is_valid(j)]

        if len(valid_comps) == 3:

            if NTK_SHOW_BOUNDARY_PAIRS:
                add_comp("Внутр + Дир", torch.cat([J_in, J_D], dim=0), "#0891B2", "s")
                add_comp("Внутр + Нейм", torch.cat([J_in, J_N], dim=0), "#7C3AED", "D")

            add_comp("Дир + Нейм", torch.cat([J_D, J_N], dim=0), "#BE123C", "X")

        if len(valid_comps) > 1:
            add_comp("Полная", torch.cat(valid_comps, dim=0), "#E11D48", "P")

        return components

    def analyze(
        self,
        epoch: int,
        X_interior: torch.Tensor,
        X_boundary: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        bc_mask: Optional[torch.Tensor] = None,
        w_pde: float = 1.0,
        w_dirichlet: float = 1.0,
        w_neumann: float = 1.0,
    ) -> NTKResult:
        device = next(self.model.parameters()).device
        self.logger(f"[NTK] Строгий спектральный анализ для эпохи {epoch}...")

        X_in = self._subsample(X_interior.to(device), self.n_interior)
        n_in = len(X_in)

        J_u_in = compute_jacobian(self.model, X_in)

        J_pde_in = compute_pde_jacobian(self.model, X_in, 
        w_pde=w_pde, w_dir=w_dirichlet, w_neu=w_neumann)

        J_u_D, J_u_N = None, None
        J_pde_D, J_pde_N = None, None
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

                J_u_D = compute_jacobian(self.model, xy_D)

                J_pde_D = compute_bc_jacobian(self.model, xy_D, normals_D, "dirichlet")

            if n_N > 0:
                normals_N = normals_sub[idx_N] if normals_sub is not None else None

                J_u_N = compute_jacobian(self.model, xy_N)

                J_pde_N = compute_bc_jacobian(self.model, xy_N, normals_N, "neumann")

        comps_u = self._build_dashboard_components(J_u_in, J_u_D, J_u_N)
        comps_pde = self._build_dashboard_components(J_pde_in, J_pde_D, J_pde_N)

        dash_path_u = plot_ntk_master_dashboard(
            epoch=epoch,
            components=comps_u,
            output_dir=self.output_dir,
            prefix="ntk_standard",
            title_prefix="Стандартный NTK (Выход сети)"
        )

        dash_path_pde = plot_ntk_master_dashboard(
            epoch=epoch,
            components=comps_pde,
            output_dir=self.output_dir,
            prefix="ntk_pde",
            title_prefix="PDE NTK (Функция потерь)"
        )

        self.logger(f"[NTK] Дашборд Стандартного NTK сохранен: {dash_path_u}")
        self.logger(f"[NTK] Дашборд PDE NTK сохранен: {dash_path_pde}")

        result = NTKResult(
            epoch=epoch,
            eigenvalues_KL=comps_pde.get("Внутренние", {}).get("eigenvalues", np.array([])),
            metrics_KL=comps_pde.get("Внутренние", {}).get("metrics", {}),
            eigenvalues_K=comps_u.get("Внутренние", {}).get("eigenvalues", np.array([])),
            metrics_K=comps_u.get("Внутренние", {}).get("metrics", {}),
            metrics_D=comps_pde.get("Дирихле", {}).get("metrics"),
            metrics_N=comps_pde.get("Нейман", {}).get("metrics"),
            metrics_tot=comps_pde.get("Полная", {}).get("metrics"),
            n_interior=n_in,
            n_dirichlet=n_D,
            n_neumann=n_N
        )

        self.history.append(result)
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