from __future__ import annotations
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE, DEFAULT_LR, TRAIN_EPOCHS, BC_PENALTY, LOSS_TYPE, OUTPUT_DIR,
    USE_NTK_PRECOND, NTK_PRECOND_EVERY, NTK_PRECOND_POINTS, NTK_PRECOND_REG,
    USE_ADAPTIVE_FREQ, ADAPTIVE_FREQ_POINTS,
    NTK_ANALYSIS_EVERY, NTK_ANALYSIS_POINTS,
)
from geometry.domains import BaseDomain
from geometry.quadrature import QuadratureData
from problems.solutions import AnalyticalSolution

from networks.pinn import PINN
from networks.corners import build_corner_enrichment
from networks.ntk_utils import (
    ntk_spectrum_analysis, ntk_preconditioned_step_batched,
    extract_learned_frequencies,
)
from networks.freq_init import compute_adaptive_frequencies

from training.data_module import DataModule, prepare_sample
from evaluation.evaluator import Evaluator
from file_io.logger import FileLogger

from functionals.operators import laplacian, gradient
from functionals.integrals import domain_integral, boundary_integral

from visualization.ntk_plotter import (
    plot_ntk_analysis, plot_spectrum_evolution, plot_adaptive_frequencies,
)

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
            self.opt_pinn, mode='min', factor=0.5, patience=50, min_lr=1e-6)

        domain_name = getattr(domain, 'name', 'Domain')
        self.evaluator = Evaluator(
            pinn=self.pinn, data=self.data, solution=solution,
            logger=logger, domain_name=domain_name,
            has_neumann=self.has_neumann)

        self._ntk_spectra_history = []
        self._ntk_spectra_epochs = []

    def train(self) -> None:
        self.logger.section(
            f"Training PINN (Loss: {LOSS_TYPE}, Epochs: {TRAIN_EPOCHS}, "
            f"NTK-precond: {USE_NTK_PRECOND}, AdaptiveFreq: {USE_ADAPTIVE_FREQ})"
        )

        best_e = float("inf")
        bd_iter = iter(self.data.boundary_iter())
        callback = self.evaluator.pretrain_callback()

        self.pinn.train()
        for ep in range(1, TRAIN_EPOCHS + 1):
            ep_loss = ep_pde = ep_dir = ep_neu = 0.0
            nb = len(self.data.in_loader)

            for batch_in in self.data.in_loader:
                xy_in, f_in, vol_w, tri_idx, _ = batch_in
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

                v_bd = self.pinn(xb)
                diff_D = v_bd - g_D
                loss_neu = torch.tensor(0.0, device=DEVICE)

                if LOSS_TYPE == "mse":
                    loss_pde = residual.pow(2).mean()
                    mask_sum = bc_mask.sum()
                    loss_dir = (diff_D ** 2 * bc_mask).sum() / (mask_sum + 1e-8)
                    if self.has_neumann:
                        grad_v_bd = gradient(v_bd, xb)
                        v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
                        diff_N = v_n - g_N
                        neu_mask = 1.0 - bc_mask
                        neu_sum = neu_mask.sum()
                        loss_neu = (diff_N ** 2 * neu_mask).sum() / (neu_sum + 1e-8)
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
                ep_pde += loss_pde.item()
                ep_dir += loss_dir.item()
                ep_neu += loss_neu.item()

            avg_loss = ep_loss / nb
            self.scheduler.step(avg_loss)

            if USE_NTK_PRECOND and ep % NTK_PRECOND_EVERY == 0:
                try:
                    self._apply_ntk_preconditioned_step(ep)
                except Exception as e:
                    self.logger(f"  [NTK-Precond] Ошибка на эпохе {ep}: {e}")

            if ep == 1 or ep % 10 == 0:
                current_lr = self.opt_pinn.param_groups[0]['lr']
                callback.on_epoch_end(
                    ep, loss=avg_loss, pde=ep_pde / nb,
                    dir_loss=ep_dir / nb, neu_loss=ep_neu / nb,
                    lr=current_lr)

                if best_e > avg_loss:
                    best_e = avg_loss
                    save_path = os.path.join(
                        OUTPUT_DIR, f"{self.evaluator.domain_name}_best_pinn.pth")
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save(self.pinn.state_dict(), save_path)
                    self.logger(f"  [Model Saved] New best loss:{best_e:.4e}")

            if ep == 1 or ep % NTK_ANALYSIS_EVERY == 0:
                try:
                    self._run_ntk_analysis(ep)
                except Exception as e:
                    self.logger(f"  [NTK] Ошибка на эпохе {ep}: {e}")

        callback.on_phase_end()
        self._plot_final_ntk_reports()

    def _apply_ntk_preconditioned_step(self, epoch: int):
        s = self.data.sample
        xq = s.quad.xy_in.clone().requires_grad_(True)

        with torch.enable_grad():
            v = self.pinn(xq)
            _, lap_v = laplacian(v, xq)

        residual = (-lap_v - s.f_in).detach()
        lr = self.opt_pinn.param_groups[0]['lr']

        update_norm = ntk_preconditioned_step_batched(
            self.pinn, xq.detach(), residual,
            lr=lr * 0.1,  
            n_sample=NTK_PRECOND_POINTS,
            reg=NTK_PRECOND_REG,
        )
        self.logger(f"  [NTK-Precond] Epoch {epoch}: ||δθ|| = {update_norm:.4e}")

    def _run_ntk_analysis(self, epoch: int):
        device = next(self.pinn.parameters()).device
        xy = torch.rand(NTK_ANALYSIS_POINTS, 2, device=device) * 2 - 1

        analysis = ntk_spectrum_analysis(self.pinn, xy)
        self._ntk_spectra_history.append(analysis)
        self._ntk_spectra_epochs.append(epoch)

        self.logger(
            f"  [NTK] Epoch {epoch}: κ={analysis['condition_number']:.2e}, "
            f"ранг_эфф={analysis['effective_rank']:.1f}, "
            f"trace={analysis['trace']:.2e}"
        )

        plot_ntk_analysis(self.pinn, epoch)

    def _plot_final_ntk_reports(self):

        if len(self._ntk_spectra_history) >= 2:
            try:
                plot_spectrum_evolution(
                    self._ntk_spectra_history,
                    self._ntk_spectra_epochs,
                )
                self.logger("  [NTK] Сохранён график эволюции спектра")
            except Exception as e:
                self.logger(f"  [NTK] Ошибка эволюции спектра: {e}")

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
                )
                self.logger("  [NTK] Сохранён график адаптивных частот")
            except Exception as e:
                self.logger(f"  [NTK] Ошибка графика частот: {e}")
