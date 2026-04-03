from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from config import NTK_SHOW_BOUNDARY_PAIRS
from ntk import (
    compute_jacobian, 
    compute_pde_jacobian, 
    compute_bc_jacobian,
    compute_ntk_from_jacobian,
    compute_cross_ntk,          
    compute_spectrum,
    get_all_metrics,
    plot_ntk_evolution  
)

from visualization import plot_ntk_master_dashboard

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

    trace_rr: float = 0.0
    trace_bb: float = 0.0
    interference_score: float = 0.0

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

        J_b_list = [j for j in [J_D_scaled, J_N_scaled] if j is not None]
        J_b_scaled = torch.cat(J_b_list, dim=0) if len(J_b_list) > 0 else None

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

        add_comp("Блок Уравнения (K_rr)", J_in_scaled, "#2563EB", "o")
        add_comp("Блок Границы (K_bb)", J_b_scaled, "#059669", "s")

        if is_valid(J_D_scaled) and is_valid(J_N_scaled):
            add_comp("Детали: Дирихле", J_D_scaled, "#10B981", "^")
            add_comp("Детали: Нейман", J_N_scaled, "#D97706", "v")

        valid_main_blocks = [j for j in [J_in_scaled, J_b_scaled] if is_valid(j)]
        if len(valid_main_blocks) > 1:
            J_full = torch.cat(valid_main_blocks, dim=0)
            add_comp("Полная матрица (K_full)", J_full, "#E11D48", "P")

        return components

    def _plot_block_matrix_heatmap(self, K_rr: torch.Tensor, K_bb: torch.Tensor, K_rb: torch.Tensor, 
                                    epoch: int, n_r: int, n_D: int, n_N: int, sort_by_diagonal: bool = True):
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            if sort_by_diagonal:
                if n_r > 0:
                    idx_r = torch.argsort(torch.diag(K_rr), descending=True)
                    K_rr = K_rr[idx_r][:, idx_r]
                    K_rb = K_rb[idx_r, :] 

                idx_D = torch.empty(0, dtype=torch.long, device=K_bb.device)
                if n_D > 0:
                    idx_D = torch.argsort(torch.diag(K_bb[:n_D, :n_D]), descending=True)

                idx_N = torch.empty(0, dtype=torch.long, device=K_bb.device)
                if n_N > 0:
                    idx_N = torch.argsort(torch.diag(K_bb[n_D:, n_D:]), descending=True) + n_D

                idx_b = torch.cat([idx_D, idx_N])
                if len(idx_b) > 0:
                    K_bb = K_bb[idx_b][:, idx_b]
                    K_rb = K_rb[:, idx_b]

            K_top = torch.cat([K_rr, K_rb], dim=1)
            K_bottom = torch.cat([K_rb.T, K_bb], dim=1)
            K_full = torch.cat([K_top, K_bottom], dim=0).cpu().numpy()

            K_vis = np.log10(np.abs(K_full) + 1e-12)

            plt.figure(figsize=(9, 8))
            ax = sns.heatmap(K_vis, cmap="magma", square=True, cbar_kws={'label': 'log10(|K|)'})

            ax.axhline(n_r, color='white', lw=2, linestyle='--')
            ax.axvline(n_r, color='white', lw=2, linestyle='--')

            if n_D > 0 and n_N > 0:
                ax.axhline(n_r + n_D, color='lightgray', lw=1.5, linestyle=':')
                ax.axvline(n_r + n_D, color='lightgray', lw=1.5, linestyle=':')

            def add_block_label(start_idx, size, text):
                if size > 0:
                    center = start_idx + size / 2
                    ax.text(center, center, text, color='white', ha='center', va='center',
                            fontsize=11, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

            add_block_label(0, n_r, "Уравнение\n(K_rr)")
            add_block_label(n_r, n_D, "Гр. условия\n(Дирихле)")
            add_block_label(n_r + n_D, n_N, "Гр. условия\n(Нейман)")

            plt.title(f"Блочная структура PDE NTK (Эпоха {epoch})", fontsize=14, pad=15)
            plt.xlabel(f"Точки (Отсортированы по величине градиента)", fontsize=12)
            plt.ylabel(f"Точки (Отсортированы по величине градиента)", fontsize=12)

            path = os.path.join(self.output_dir, f"ntk_block_heatmap_ep{epoch}.png")
            plt.savefig(path, bbox_inches='tight', dpi=150)
            plt.close()
            return path

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
        J_pde_in = compute_pde_jacobian(self.model, X_in)

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

        trace_rr, trace_bb, interference_score = 0.0, 0.0, 0.0
        n_r = J_pde_in.shape[0] if J_pde_in is not None else 0

        J_b_scaled_list = []
        if n_D > 0: J_b_scaled_list.append(J_pde_D * math.sqrt(w_dirichlet))
        if n_N > 0: J_b_scaled_list.append(J_pde_N * math.sqrt(w_neumann))

        if len(J_b_scaled_list) > 0 and n_r > 0:
            J_r_scaled = J_pde_in * math.sqrt(w_pde)
            J_b_scaled = torch.cat(J_b_scaled_list, dim=0)

            K_rr = compute_ntk_from_jacobian(J_r_scaled)
            K_bb = compute_ntk_from_jacobian(J_b_scaled)
            K_rb = compute_cross_ntk(J_r_scaled, J_b_scaled)

            trace_rr = torch.trace(K_rr).item()
            trace_bb = torch.trace(K_bb).item()

            norm_rr = torch.norm(K_rr, p='fro')
            norm_bb = torch.norm(K_bb, p='fro')
            norm_rb = torch.norm(K_rb, p='fro')

            interference_score = (norm_rb / (norm_rr * norm_bb + 1e-12)**0.5).item()

            self.logger(f"[NTK Block Analysis] Эпоха {epoch}:")
            self.logger(f"  -> След PDE (tr K_rr): {trace_rr:.4e} | След Boundary (tr K_bb): {trace_bb:.4e}")
            self.logger(f"  -> Интерференция ||K_rb||: {interference_score:.4f}")

            hm_path = self._plot_block_matrix_heatmap(K_rr, K_bb, K_rb, epoch, n_r, n_D, n_N)
            self.logger(f"  -> Heatmap сохранен: {hm_path}")

        comps_u = self._build_dashboard_components(J_u_in, J_u_D, J_u_N, 1.0, 1.0, 1.0)
        comps_pde = self._build_dashboard_components(J_pde_in, J_pde_D, J_pde_N, w_pde, w_dirichlet, w_neumann)

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
            title_prefix="PDE NTK (С учетом весов loss-функции)"
        )

        self.logger(f"[NTK] Дашборды спектра сохранены.")

        result = NTKResult(
            epoch=epoch,
            eigenvalues_KL=comps_pde.get("Блок Уравнения (K_rr)", {}).get("eigenvalues", np.array([])),
            metrics_KL=comps_pde.get("Блок Уравнения (K_rr)", {}).get("metrics", {}),
            eigenvalues_K=comps_u.get("Блок Уравнения (K_rr)", {}).get("eigenvalues", np.array([])),
            metrics_K=comps_u.get("Блок Уравнения (K_rr)", {}).get("metrics", {}),
            metrics_D=comps_pde.get("Детали: Дирихле", {}).get("metrics"),
            metrics_N=comps_pde.get("Детали: Нейман", {}).get("metrics"),
            metrics_tot=comps_pde.get("Полная матрица (K_full)", {}).get("metrics"),
            n_interior=n_in,
            n_dirichlet=n_D,
            n_neumann=n_N,
            trace_rr=trace_rr,
            trace_bb=trace_bb,
            interference_score=interference_score
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
            self.logger("[NTK] Недостаточно данных для графика эволюции.")
            return

        self.logger("[NTK] Генерация дашборда эволюции NTK...")
        path = plot_ntk_evolution(self.history, self.output_dir)
        if path:
            self.logger(f"[NTK] Дашборд эволюции сохранен: {path}")