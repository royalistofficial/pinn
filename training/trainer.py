from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DEVICE, DEFAULT_LR, TRAIN_EPOCHS, BC_PENALTY, LOSS_TYPE, OUTPUT_DIR
from geometry.domains import BaseDomain
from geometry.quadrature import QuadratureData
from problems.solutions import AnalyticalSolution

from networks.pinn import PINN
from networks.corners import build_corner_enrichment
from training.data_module import DataModule, prepare_sample
from evaluation.evaluator import Evaluator
from file_io.logger import FileLogger

from functionals.operators import laplacian, gradient
from functionals.integrals import domain_integral, boundary_integral

class Trainer:
    def __init__(
        self,
        domain: BaseDomain,
        quad: QuadratureData,
        solution: AnalyticalSolution,
        logger: FileLogger,
        lr: float = DEFAULT_LR,
        batch_size: int = 4096,
    ):
        self.has_neumann = domain.has_neumann
        self.logger = logger
        
        sample = prepare_sample(quad, solution)
        self.data = DataModule(sample, batch_size=min(batch_size, len(quad.xy_in)))
 
        corner_enrichment = build_corner_enrichment(domain, DEVICE)
        self.pinn = PINN(corner_enrichment=corner_enrichment).to(DEVICE)
        self.opt_pinn = torch.optim.Adam(self.pinn.parameters(), lr=lr)

        self.scheduler = ReduceLROnPlateau(
            self.opt_pinn,
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-6
        )

        # Инициализация Evaluator (передаем имя домена для графиков)
        domain_name = getattr(domain, 'name', 'Domain')
        self.evaluator = Evaluator(
            pinn=self.pinn,
            data=self.data,
            solution=solution,
            logger=logger,
            domain_name=domain_name,
            has_neumann=self.has_neumann,
        )

    def train(self) -> None:
        self.logger.section(f"Training standard PINN (Loss: {LOSS_TYPE}, Epochs: {TRAIN_EPOCHS})")
        
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
                
                vol_w_sc, surf_w_sc = self.data.scale_weights(xy_in, xy_bd, vol_w, surf_w)

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
                    loss_dir = (diff_D**2 * bc_mask).sum() / (mask_sum + 1e-8)
                    
                    if self.has_neumann:
                        grad_v_bd = gradient(v_bd, xb)
                        v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
                        diff_N = v_n - g_N
                        neu_mask = 1.0 - bc_mask
                        neu_sum = neu_mask.sum()
                        loss_neu = (diff_N**2 * neu_mask).sum() / (neu_sum + 1e-8)
                else:
                    loss_pde = domain_integral(residual ** 2, vol_w_sc)
                    loss_dir = boundary_integral(diff_D**2, surf_w_sc, mask=bc_mask)
                    
                    if self.has_neumann:
                        grad_v_bd = gradient(v_bd, xb)
                        v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
                        diff_N = v_n - g_N
                        loss_neu = boundary_integral(diff_N**2, surf_w_sc, mask=(1.0 - bc_mask))

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

            if ep == 1 or ep % 10 == 0:
                current_lr = self.opt_pinn.param_groups[0]['lr']
                callback.on_epoch_end(
                    ep, loss=avg_loss, pde=ep_pde / nb,
                    dir_loss=ep_dir / nb, neu_loss=ep_neu / nb,
                    lr=current_lr
                )
                
                if best_e > avg_loss:
                    best_e = avg_loss

                    save_path = os.path.join(OUTPUT_DIR, f"{self.evaluator.domain_name}_best_pinn.pth")
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save(self.pinn.state_dict(), save_path)
                    self.logger(f"  [Model Saved] New best energy error: E={best_e:.4e}")

        callback.on_phase_end()