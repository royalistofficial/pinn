from __future__ import annotations
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE, DEFAULT_LR, TRAIN_EPOCHS, BC_PENALTY, LOSS_TYPE, OUTPUT_DIR,
    USE_ADAPTIVE_FREQ, ADAPTIVE_FREQ_POINTS,
    NTK_ANALYSIS_EVERY, NTK_ANALYSIS_POINTS, NTK_NODE_ORDER,
    USE_NTK_WEIGHTS, NTK_WEIGHT_EVERY, NTK_WEIGHT_POINTS, NTK_WEIGHT_MOMENTUM,
    USE_CORNER_WEIGHTING, CORNER_WEIGHT_BETA,
)
from geometry.domains import BaseDomain
from geometry.quadrature import QuadratureData
from problems.solutions import AnalyticalSolution

from networks.pinn import PINN
from networks.corners import build_corner_enrichment
from networks.ntk_utils import (
    extract_learned_frequencies,
    compute_jacobian,
    compute_pde_jacobian,
    compute_ntk_from_jacobian,
)
from networks.freq_init import compute_adaptive_frequencies

from training.data_module import DataModule, prepare_sample
from evaluation.evaluator import Evaluator
from file_io.logger import FileLogger

from functionals.operators import laplacian, gradient
from functionals.integrals import domain_integral, boundary_integral

from visualization.ntk_plotter import (
    plot_ntk_full_analysis,
    plot_spectrum_evolution,
    plot_adaptive_frequencies,
)

_NTK_PLOT_DIR = os.path.join(OUTPUT_DIR, "ntk_plots")

class Trainer:
    def __init__(self, domain, quad, solution, logger,
                 lr=DEFAULT_LR, batch_size=4096):
        self.has_neumann = domain.has_neumann
        self.logger = logger
        self.solution = solution
        self.domain = domain

        sample = prepare_sample(quad, solution)
        self.data = DataModule(sample, batch_size=min(batch_size, len(quad.xy_in)))

        init_freqs = None
        self._init_freqs_np = None
        if USE_ADAPTIVE_FREQ:
            try:
                from config import PINN_ARCH
                init_freqs = compute_adaptive_frequencies(
                    solution, domain,
                    n_fourier=PINN_ARCH["n_fourier"],
                    n_points=ADAPTIVE_FREQ_POINTS,
                )
                self._init_freqs_np = init_freqs.numpy().copy()
                logger(f"  [AdaptiveFreq] Инициализированы {len(init_freqs)} частот")
            except Exception as e:
                logger(f"  [AdaptiveFreq] Ошибка: {e}, используем стандартные")
                init_freqs = None

        corner_enrichment = build_corner_enrichment(domain, DEVICE)
        self.pinn = PINN(
            corner_enrichment=corner_enrichment,
            init_freqs=init_freqs,
        ).to(DEVICE)

        self.opt_pinn = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.opt_pinn, mode="min", factor=0.5,
            patience=50, min_lr=1e-6,
        )

        domain_name = getattr(domain, "name", "Domain")
        self.evaluator = Evaluator(
            pinn=self.pinn, data=self.data, solution=solution,
            logger=logger, domain_name=domain_name,
            has_neumann=self.has_neumann,
        )

        self._ntk_history: list[dict] = []
        self._ntk_epochs:  list[int]  = []

        self._lambda_r = 1.0   
        self._lambda_b = 1.0   

        self._corner_pde_weights: torch.Tensor | None = None
        if USE_CORNER_WEIGHTING:
            self._corner_pde_weights = _compute_corner_weights(
                self.data.sample.quad.xy_in,
                domain,
                beta=CORNER_WEIGHT_BETA,
            )
            logger(f"  [CornerWeight] β={CORNER_WEIGHT_BETA}, "
                   f"min_w={self._corner_pde_weights.min().item():.3f}, "
                   f"max_w={self._corner_pde_weights.max().item():.3f}")

    def train(self) -> None:
        self.logger.section(
            f"Training PINN  (Loss: {LOSS_TYPE},  Epochs: {TRAIN_EPOCHS},  "
            f"AdaptiveFreq: {USE_ADAPTIVE_FREQ})"
        )
        
        self._run_ntk_analysis(0)

        best_loss = float("inf")
        bd_iter   = iter(self.data.boundary_iter())
        callback  = self.evaluator.pretrain_callback()

        self.pinn.train()
        for ep in range(1, TRAIN_EPOCHS + 1):
            ep_loss = ep_pde = ep_dir = ep_neu = 0.0
            n_batches = len(self.data.in_loader)

            for batch_in in self.data.in_loader:
                xy_in, f_in, vol_w, tri_idx, idx_in = batch_in
                batch_bd = next(bd_iter)
                xy_bd, normals, g_D, g_N, bc_mask, surf_w, tri_idx_bd, _ = batch_bd

                vol_w_sc, surf_w_sc = self.data.scale_weights(
                    xy_in, xy_bd, vol_w, surf_w)

                self.opt_pinn.zero_grad(set_to_none=True)
                xq = xy_in.clone().requires_grad_(True)
                xb = xy_bd.clone().requires_grad_(True)

                v = self.pinn(xq)
                _, lap_v = laplacian(v, xq)
                residual = -lap_v - f_in

                v_bd      = self.pinn(xb)
                diff_D    = v_bd - g_D
                loss_neu  = torch.tensor(0.0, device=DEVICE)

                if LOSS_TYPE == "mse":

                    if self._corner_pde_weights is not None:

                        w_corner = self._corner_pde_weights[idx_in].to(DEVICE)
                        loss_pde = (residual.pow(2) * w_corner.unsqueeze(-1)).mean()
                    else:
                        loss_pde = residual.pow(2).mean()

                    loss_pde = loss_pde * self._lambda_r

                    mask_sum = bc_mask.sum()
                    loss_dir = (diff_D ** 2 * bc_mask).sum() / (mask_sum + 1e-8)

                    loss_dir = loss_dir * self._lambda_b
                    if self.has_neumann:
                        grad_v_bd = gradient(v_bd, xb)
                        v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
                        diff_N = v_n - g_N
                        neu_mask = 1.0 - bc_mask
                        loss_neu = (diff_N ** 2 * neu_mask).sum() / (neu_mask.sum() + 1e-8)
                else:
                    loss_pde = domain_integral(residual ** 2, vol_w_sc)
                    loss_dir = boundary_integral(diff_D ** 2, surf_w_sc, mask=bc_mask)
                    if self.has_neumann:
                        grad_v_bd = gradient(v_bd, xb)
                        v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
                        diff_N = v_n - g_N
                        loss_neu = boundary_integral(
                            diff_N ** 2, surf_w_sc, mask=(1.0 - bc_mask))

                loss = loss_pde + BC_PENALTY * (loss_dir + loss_neu)
                loss.backward()
                nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
                self.opt_pinn.step()

                ep_loss += loss.item()
                ep_pde  += loss_pde.item()
                ep_dir  += loss_dir.item()
                ep_neu  += loss_neu.item()

            avg_loss = ep_loss / n_batches
            self.scheduler.step(avg_loss)

            if ep == 1 or ep % 10 == 0:
                current_lr = self.opt_pinn.param_groups[0]["lr"]
                callback.on_epoch_end(
                    ep,
                    loss=avg_loss,
                    pde=ep_pde / n_batches,
                    dir_loss=ep_dir / n_batches,
                    neu_loss=ep_neu / n_batches,
                    lr=current_lr,
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(
                        OUTPUT_DIR,
                        f"{self.evaluator.domain_name}_best_pinn.pth",
                    )
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save(self.pinn.state_dict(), save_path)
                    self.logger(f"  [Model Saved] New best loss: {best_loss:.4e}")

            if ep % NTK_ANALYSIS_EVERY == 0:
                try:
                    self._run_ntk_analysis(ep)
                except Exception as exc:
                    self.logger(f"  [NTK] Ошибка на эпохе {ep}: {exc}")

        callback.on_phase_end()
        self._plot_final_reports()

    def _run_ntk_analysis(self, epoch: int) -> None:
        xy_all = self.data.sample.quad.xy_in.detach()  
        N_all  = len(xy_all)
        n_pts  = min(NTK_ANALYSIS_POINTS, N_all)

        self.logger(
            f"  [NTK] Epoch {epoch}: {n_pts} точек "
            f"(всего обучающих: {N_all})"
        )

        result = plot_ntk_full_analysis(
            model=self.pinn,
            epoch=epoch,
            X_train=xy_all,
            n_pts=n_pts,
            node_order=NTK_NODE_ORDER,
            output_dir=_NTK_PLOT_DIR,
        )

        self._ntk_history.append({
            "eig_K":    result["eig_K"],
            "eig_KL":   result["eig_KL"],
            "kappa_K":  result["kappa_K"],
            "kappa_KL": result["kappa_KL"],
            "rank_K":   float(np.exp(-np.sum(
                _norm_entropy(result["eig_K"])
            ))),
            "rank_KL":  float(np.exp(-np.sum(
                _norm_entropy(result["eig_KL"])
            ))),
        })
        self._ntk_epochs.append(epoch)

        self.logger(
            f"  [NTK] κ(K)={result['kappa_K']:.2e}  "
            f"κ(K_L)={result['kappa_KL']:.2e}"
        )

    def _plot_final_reports(self) -> None:

        if len(self._ntk_history) >= 2:
            try:
                plot_spectrum_evolution(
                    self._ntk_history,
                    self._ntk_epochs,
                    output_dir=_NTK_PLOT_DIR,
                )
                self.logger("  [NTK] Сохранён график эволюции спектра")
            except Exception as exc:
                self.logger(f"  [NTK] Ошибка эволюции спектра: {exc}")

        learned = extract_learned_frequencies(self.pinn)
        if self._init_freqs_np is not None and len(learned) > 0:
            try:
                spectrum_info = None
                if USE_ADAPTIVE_FREQ:
                    from networks.freq_init import estimate_rhs_spectrum
                    spectrum_info = estimate_rhs_spectrum(
                        self.solution, self.domain)
                plot_adaptive_frequencies(
                    self._init_freqs_np, learned,
                    spectrum_info=spectrum_info,
                    output_dir=_NTK_PLOT_DIR,
                )
                self.logger("  [NTK] Сохранён график адаптивных частот")
            except Exception as exc:
                self.logger(f"  [NTK] Ошибка графика частот: {exc}")

def _norm_entropy(eig: np.ndarray) -> np.ndarray:
    ep = eig[eig > 1e-10]
    if len(ep) == 0:
        return np.array([0.0])
    p = ep / ep.sum()
    return p * np.log(p + 1e-30)

def _compute_corner_weights(
        xy_in: torch.Tensor,
        domain,
        beta: float = 0.4,
    ) -> torch.Tensor:
    from networks.corners import extract_corners

    corners, angles = extract_corners(domain)
    if corners.shape[0] == 0 or beta < 1e-6:

        return torch.ones(len(xy_in), dtype=torch.float32, device=xy_in.device)

    xy_np = xy_in.detach().cpu().numpy()        
    c_np  = corners.numpy()                      

    dists = np.full(len(xy_np), np.inf)
    for c in c_np:
        d = np.linalg.norm(xy_np - c[None, :], axis=1)
        dists = np.minimum(dists, d)

    dists = np.clip(dists, 1e-4, None)

    w = dists ** beta                            
    w = w / w.mean()                             

    return torch.tensor(w, dtype=torch.float32, device=xy_in.device)