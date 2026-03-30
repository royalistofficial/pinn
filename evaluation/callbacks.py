from __future__ import annotations
import os
from typing import Optional, Callable, TYPE_CHECKING

import torch
import torch.nn as nn

from config import OUTPUT_DIR
from evaluation.metrics import MetricsCalculator
from evaluation.metrics_history import MetricsHistory
from functionals.operators import gradient
from visualization.training_plots import (
    plot_training_metrics,
    plot_solution_fields,
)
from visualization.mesh_plots import refine_mesh, evaluate_fields

if TYPE_CHECKING:
    from training.weight_balancer import WeightBalancer
    from training.ntk_analyzer import NTKAnalyzer

class TrainingCallback:
    def __init__(
                self,
                pinn: nn.Module,
                data,
                solution,
                logger: Callable[[str], None],
                domain_name: str = "domain",
                metrics_calculator: Optional[MetricsCalculator] = None,
                weight_balancer: Optional["WeightBalancer"] = None,
                ntk_analyzer: Optional["NTKAnalyzer"] = None,
            ):
        self.pinn = pinn
        self.data = data
        self.solution = solution
        self.logger = logger
        self.domain_name = domain_name
        self.weight_balancer = weight_balancer
        self.ntk_analyzer = ntk_analyzer

        self.metrics_calculator = metrics_calculator or MetricsCalculator(solution)
        self.history = MetricsHistory()

        self.l2_pred_initial = None
        self.energy_pred_initial = None
        self.poincare_constant = None
        self.stability_constant = None

    def on_epoch_end(self, epoch: int, lr: float, **loss_info) -> float:
        s = self.data.sample
        xq = s.quad.xy_in.clone().requires_grad_(True)

        with torch.enable_grad():
            v = self.pinn(xq)
            gv = gradient(v, xq, create_graph=False)

        with torch.no_grad():
            metrics = self.metrics_calculator.compute_all(v, gv, xq, s.quad.vol_w)
            ee = metrics["energy"]
            rel = metrics["rel_l2"]
            rel_en = metrics["rel_energy"]

            w_pde, w_dirichlet, w_neumann = 1.0, 1.0, 0.0
            if self.weight_balancer is not None:
                weights = self.weight_balancer.get_weights()
                w_pde = weights.get("pde", 1.0)
                w_dirichlet = weights.get("dirichlet", 1.0)
                w_neumann = weights.get("neumann", 0.0)

            l2_pred = None
            energy_pred = None

            if self.ntk_analyzer is not None and len(self.ntk_analyzer.history) > 0:
                latest_ntk = self.ntk_analyzer.history[-1]
                if getattr(latest_ntk, 'convergence_prediction', None) is not None:
                    pred = latest_ntk.convergence_prediction
                    if hasattr(pred, 'error_bounds') and pred.error_bounds is not None:
                        eb = pred.error_bounds

                        if self.l2_pred_initial is None:
                            self.l2_pred_initial = eb.l2_error_predicted
                            self.energy_pred_initial = eb.energy_error_predicted
                            self.poincare_constant = eb.poincare_constant
                            self.stability_constant = eb.stability_constant

                        pred_epochs = pred.predicted_epochs
                        if len(pred_epochs) > 0 and len(eb.l2_error_predicted) > 0:
                            import numpy as np
                            idx = np.searchsorted(pred_epochs, epoch)
                            if idx < len(eb.l2_error_predicted):
                                l2_pred = float(eb.l2_error_predicted[idx])
                                energy_pred = float(eb.energy_error_predicted[idx])
                            elif len(eb.l2_error_predicted) > 0:
                                l2_pred = float(eb.l2_error_predicted[-1])
                                energy_pred = float(eb.energy_error_predicted[-1])

            self.history.pretrain.log(
                epoch,
                loss=loss_info.get("loss", 0),
                pde=loss_info.get("pde", 0),
                dir_loss=loss_info.get("dir_loss", 0),
                neu_loss=loss_info.get("neu_loss", 0),
                energy=ee,
                rel_err=rel,
                rel_energy=rel_en,
                lr=lr,
                w_pde=w_pde,
                w_dirichlet=w_dirichlet,
                w_neumann=w_neumann,
                l2_pred=l2_pred if l2_pred is not None else 0.0,
                energy_pred=energy_pred if energy_pred is not None else 0.0,
            )

            log_msg = (
                f"  Epoch {epoch:04d}: loss={loss_info.get('loss', 0):.4e} | "
                f"pde={loss_info.get('pde', 0):.2e} "
                f"dir={loss_info.get('dir_loss', 0):.2e} "
                f"neu={loss_info.get('neu_loss', 0):.2e} | "
                f"E={ee:.4e} | RelL2={rel:.4e} RelE={rel_en:.4e}"
            )

            if self.weight_balancer is not None and self.weight_balancer.config.enabled:
                log_msg += f" | w=[{w_pde:.2f},{w_dirichlet:.2f},{w_neumann:.2f}]"

            if l2_pred is not None:
                log_msg += f" | L²_pred={l2_pred:.2e}"

            self.logger(log_msg)

            plot_training_metrics(
                self.history.pretrain.as_dict(),
                self.domain_name,
                os.path.join(OUTPUT_DIR, f"{self.domain_name}_metrics.png"),
            )

        return ee

    def on_training_end(self) -> None:
        best = min(self.history.pretrain.data.get("energy", [float("inf")]))
        self.logger(f"Training completed. Best energy error: E={best:.4e}")

    def _plot_fields(self, epoch: int) -> None:
        mesh = self.data.sample.quad.mesh
        tri_refi, pts_refi = refine_mesh(mesh)

        xy_t = torch.tensor(
            pts_refi, dtype=torch.float32,
            device=self.data.sample.quad.xy_in.device
        )

        self.pinn.eval()
        fv = evaluate_fields(xy_t, self.pinn, self.solution)
        self.pinn.train()

        path = os.path.join(OUTPUT_DIR, f"{self.domain_name}_fields_{epoch:05d}.png")
        plot_solution_fields(epoch, tri_refi, fv.v, fv.u_exact, fv.abs_error, fv.energy_density, path)