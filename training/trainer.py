from __future__ import annotations
import os

import numpy as np
import torch
import torch.nn as nn

from config import (
    DEVICE, ADAM_LR, ADAM_EPOCHS, LBFGS_EPOCHS, LBFGS_LR, 
    LBFGS_MAX_ITER, LBFGS_TOLERANCE_GRAD, LBFGS_TOLERANCE_CHANGE,
    BC_PENALTY, OUTPUT_DIR, NTK_ANALYSIS_EVERY, NTK_ANALYSIS_POINTS, 
    AUTO_BALANCE_ENABLED
)

from networks.configs import NetworkConfig
from networks.pinn import PINN

from training.data_module import DataModule, prepare_sample
from ntk.ntk_analyzer import NTKAnalyzer
from training.weight_balancer import WeightBalancer, WeightConfig

from evaluation.metrics import MetricsCalculator
from evaluation.callbacks import TrainingCallback

class Trainer:
    def __init__(
                self,
                domain,
                quad,
                solution,
                logger,
                batch_size: int = 4096,
                config: NetworkConfig | dict | None = None,
                eval_quad = None, 
            ):
        self.has_neumann = domain.has_neumann
        self.logger = logger
        self.solution = solution
        self.domain = domain

        sample = prepare_sample(quad, solution)
        self.data = DataModule(sample, batch_size=min(batch_size, len(quad.xy_in)))

        if config is None:
            from networks.configs import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        self.pinn = PINN(config).to(DEVICE)

        self.opt_adam = torch.optim.Adam(self.pinn.parameters(), lr=ADAM_LR)

        self.opt_lbfgs = torch.optim.LBFGS(
            self.pinn.parameters(), 
            lr=LBFGS_LR, 
            max_iter=LBFGS_MAX_ITER, 
            tolerance_grad=LBFGS_TOLERANCE_GRAD, 
            tolerance_change=LBFGS_TOLERANCE_CHANGE,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        self.lr = ADAM_LR
        self.weight_balancer = WeightBalancer(WeightConfig(
            enabled=AUTO_BALANCE_ENABLED,
        ))

        self.ntk_analyzer = NTKAnalyzer(
            model=self.pinn,
            output_dir=OUTPUT_DIR,
            n_interior=NTK_ANALYSIS_POINTS,
            n_boundary=NTK_ANALYSIS_POINTS // 2,
            logger=logger,
        )

        self.metrics_calculator = MetricsCalculator(solution)

        domain_name = getattr(domain, "name", "Domain")
        self.callback = TrainingCallback(
            pinn=self.pinn,
            data=self.data,
            solution=solution,
            logger=logger,
            domain_name=domain_name,
            metrics_calculator=self.metrics_calculator,
            weight_balancer=self.weight_balancer,
            ntk_analyzer=self.ntk_analyzer,
            eval_quad=eval_quad,
        )

    def _compute_domain_diameter(self, domain) -> float:
        try:
            if hasattr(domain, 'vertices'):
                verts = domain.vertices

                from scipy.spatial.distance import pdist
                return float(np.max(pdist(verts)))
            elif hasattr(domain, 'Lx') and hasattr(domain, 'Ly'):

                return float(np.sqrt(domain.Lx**2 + domain.Ly**2))
            else:

                return 1.0
        except Exception:
            return 1.0

    def train(self, patience: int = 50) -> None:
        self.logger.section(
            f"Training PINN | Phase 1: Adam ({ADAM_EPOCHS} ep) | Phase 2: L-BFGS ({LBFGS_EPOCHS} ep)"
        )

        alpha = 0.75
        min_delta = 1e-6  

        ep = 0
        ep_adam = None
        best_ema = float("inf")
        ema_loss = None
        patience_counter = 0
        loss_info = None  
        bd_iter = iter(self.data.boundary_iter())

        self.pinn.train()

        self._run_ntk_analysis(0)
        self.callback.plot_fields(0)

        if ADAM_EPOCHS > 0:
            self.logger.section("PHASE 1: ADAM OPTIMIZATION")
            self.lr = ADAM_LR
            patience_counter = 0

            for _ in range(ADAM_EPOCHS):
                ep += 1
                loss_info = self._train_epoch_adam(bd_iter)
                avg_loss = loss_info["loss"]

                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = alpha * ema_loss + (1 - alpha) * avg_loss

                if ep == 1 or ep % 1 == 0:
                    self.callback.on_epoch_end(
                        ep, lr=self.lr, ema_loss=ema_loss, **loss_info
                    )

                if ep % 100 == 0:
                    self.callback.plot_fields(ep)

                if ema_loss < best_ema - min_delta:
                    best_ema = ema_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if ep % NTK_ANALYSIS_EVERY == 0:
                    self._run_ntk_analysis(ep)

                if patience_counter >= patience:
                    self.logger(
                        f"[Early Stopping] Adam остановлен на эпохе {ep} "
                        f"(EMA без улучшений {patience} эпох)."
                    )
                    break

        if LBFGS_EPOCHS > 0:
            self.logger.section("PHASE 2: L-BFGS OPTIMIZATION")
            self.lr = LBFGS_LR
            ep_adam = ep
            patience_counter = 0

            if ADAM_EPOCHS > 0 and loss_info is not None:
                self.callback.on_epoch_end(ep, lr=self.lr, ema_loss=ema_loss, **loss_info)
                self.callback.plot_fields(ep)
                self._run_ntk_analysis(ep)

            for _ in range(LBFGS_EPOCHS):
                ep += 1
                loss_info = self._train_epoch_lbfgs(bd_iter)
                avg_loss = loss_info["loss"]

                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = alpha * ema_loss + (1 - alpha) * avg_loss

                if ep % 1 == 0:
                    self.callback.on_epoch_end(
                        ep, lr=self.lr, ema_loss=ema_loss, **loss_info
                    )

                if ep % 100 == 0:
                    self.callback.plot_fields(ep)

                if ema_loss < best_ema - min_delta:
                    best_ema = ema_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if ep % NTK_ANALYSIS_EVERY == 0:
                    self._run_ntk_analysis(ep)

                if patience_counter >= patience:
                    self.logger(
                        f"[Early Stopping] L-BFGS остановлен на эпохе {ep} "
                        f"(EMA без улучшений {patience} эпох)."
                    )
                    break

        if loss_info is not None:
            self.callback.on_epoch_end(ep, lr=self.lr, ema_loss=ema_loss, **loss_info)
            self.callback.plot_fields(ep)
            self._run_ntk_analysis(ep)
        else:
            self.logger("[Warning] Обучение не выполнялось (0 эпох).")

        self.callback.plot_metrics(ep_adam)
        self._save_model()
        self.callback.on_training_end()
        self._finalize()

    def _train_epoch_adam(self, bd_iter) -> dict:
        ep_loss = ep_pde = ep_dir = ep_neu = 0.0
        n_batches = len(self.data.in_loader)

        for batch_in in self.data.in_loader:
            xy_in, f_in, _, _, _ = batch_in
            xy_bd, normals, g_D, g_N, bc_mask, _, _, _ = next(bd_iter)

            self.opt_adam.zero_grad(set_to_none=True)
            xq = xy_in.clone().requires_grad_(True)
            xb = xy_bd.clone().requires_grad_(True)

            loss_dict = self._compute_loss(xq, xb, f_in, normals, g_D, g_N, bc_mask)
            loss = loss_dict["total"]

            loss.backward()
            nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
            self.opt_adam.step()

            ep_loss += loss.item()
            ep_pde  += loss_dict["pde"].item()
            ep_dir  += loss_dict["dirichlet"].item()
            ep_neu  += loss_dict["neumann"].item()

        return {
            "loss": ep_loss / n_batches, "pde": ep_pde / n_batches,
            "dir_loss": ep_dir / n_batches, "neu_loss": ep_neu / n_batches,
        }

    def _train_epoch_lbfgs(self, bd_iter) -> dict:

        batch_in = next(iter(self.data.in_loader))
        xy_in, f_in, _, _, _ = batch_in
        xy_bd, normals, g_D, g_N, bc_mask, _, _, _ = next(bd_iter)

        xq = xy_in.clone().requires_grad_(True)
        xb = xy_bd.clone().requires_grad_(True)

        loss_dict_tracker = {}

        def closure():
            self.opt_lbfgs.zero_grad(set_to_none=True)
            loss_dict = self._compute_loss(xq, xb, f_in, normals, g_D, g_N, bc_mask)
            loss = loss_dict["total"]
            loss.backward()

            loss_dict_tracker["total"] = loss.item()
            loss_dict_tracker["pde"] = loss_dict["pde"].item()
            loss_dict_tracker["dirichlet"] = loss_dict["dirichlet"].item()
            loss_dict_tracker["neumann"] = loss_dict["neumann"].item()

            return loss

        self.opt_lbfgs.step(closure)

        return {
            "loss": loss_dict_tracker.get("total", 0.0),
            "pde": loss_dict_tracker.get("pde", 0.0),
            "dir_loss": loss_dict_tracker.get("dirichlet", 0.0),
            "neu_loss": loss_dict_tracker.get("neumann", 0.0),
        }

    def grad_norm(self, loss: torch.Tensor) -> float:
        self.pinn.zero_grad()
        loss.backward(retain_graph=True)

        grads = []

        for p in self.pinn.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))

        if len(grads) == 0:
            return 0.0

        grad_vector = torch.cat(grads)

        norm = torch.norm(grad_vector, p=2)

        return norm.item()

    def _compute_loss(
                self,
                xq: torch.Tensor,
                xb: torch.Tensor,
                f_in: torch.Tensor,
                normals: torch.Tensor,
                g_D: torch.Tensor,
                g_N: torch.Tensor,
                bc_mask: torch.Tensor,
            ) -> dict:
        from functionals.operators import laplacian, gradient

        v = self.pinn(xq)
        _, lap_v = laplacian(v, xq)
        residual = -lap_v - f_in
        loss_pde = residual.pow(2).mean()

        v_bd = self.pinn(xb)
        diff_D = v_bd - g_D
        mask_sum = bc_mask.sum()
        loss_dir = (diff_D ** 2 * bc_mask).sum() / (mask_sum + 1e-8)

        loss_neu = torch.tensor(0.0, device=DEVICE)
        if self.has_neumann:
            grad_v_bd = gradient(v_bd, xb)
            v_n = (grad_v_bd * normals).sum(dim=1, keepdim=True)
            diff_N = v_n - g_N
            neu_mask = 1.0 - bc_mask
            loss_neu = (diff_N ** 2 * neu_mask).sum() / (neu_mask.sum() + 1e-8)

        if AUTO_BALANCE_ENABLED:
            grad_pde_norm = self.grad_norm(loss_pde)
            grad_dir_norm =  self.grad_norm(loss_dir)

            grad_neu_norm = 0.0
            if self.has_neumann:
                grad_neu_norm =  self.grad_norm(loss_neu)

            w_pde, w_dir, w_neu = self.weight_balancer.update_from_gradients(
                grad_pde_norm, grad_dir_norm, grad_neu_norm, bc_penalty=BC_PENALTY
            )
            self.pinn.zero_grad() 
        else:

            w_pde, w_dir, w_neu = self.weight_balancer.update_from_gradients(
                0.0, 0.0, 0.0, bc_penalty=BC_PENALTY
            )

        total_loss = w_pde * loss_pde + w_dir * loss_dir + w_neu * loss_neu

        return {
            "total": total_loss,
            "pde": loss_pde,
            "dirichlet": loss_dir,
            "neumann": loss_neu,
        }

    def _save_model(self) -> None:
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{self.callback.domain_name}_best_pinn.pth",
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(self.pinn.state_dict(), save_path)
        self.logger(f"[Model Saved]")

    def _run_ntk_analysis(self, epoch: int) -> None:
        quad = self.data.sample.quad
        weights = self.weight_balancer.get_weights()

        result = self.ntk_analyzer.analyze(
            epoch=epoch,
            X_interior=quad.xy_in,
            X_boundary=quad.xy_bd,
            normals=quad.normals,
            bc_mask=quad.bc_mask,
            w_pde=weights['pde'],
            w_dirichlet=weights['dirichlet'],
            w_neumann=weights['neumann'] 
        )

        if AUTO_BALANCE_ENABLED:
            self.logger(
                f"[Weight-Info] Current adaptive weights: "
                f"w_pde={weights['pde']:.3f}, "
                f"w_dir={weights['dirichlet']:.3f}, "
                f"w_neu={weights['neumann']:.3f}"
            )

    def _finalize(self) -> None:
        self.ntk_analyzer.plot_evolution()
        self.logger("[Training] Completed successfully")