from __future__ import annotations
import os
from typing import Optional, Callable

import torch
import torch.nn as nn

from config import OUTPUT_DIR
from evaluation.metrics import MetricsCalculator
from evaluation.metrics_history import MetricsHistory
from visualization.training_plots import (
    plot_training_metrics,
    plot_solution_fields,
)
from visualization.mesh_plots import refine_mesh, evaluate_fields

class TrainingCallback:
    def __init__(
                self,
                pinn: nn.Module,
                data,
                solution,
                logger: Callable[[str], None],
                domain_name: str = "domain",
                metrics_calculator: Optional[MetricsCalculator] = None,
                plot_every: int = 100,
            ):
        self.pinn = pinn
        self.data = data
        self.solution = solution
        self.logger = logger
        self.domain_name = domain_name
        self.plot_every = plot_every

        self.metrics_calculator = metrics_calculator or MetricsCalculator(solution)
        self.history = MetricsHistory()

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
            )

            self.logger(
                f"  Epoch {epoch:04d}: loss={loss_info.get('loss', 0):.4e} | "
                f"pde={loss_info.get('pde', 0):.2e} "
                f"dir={loss_info.get('dir_loss', 0):.2e} "
                f"neu={loss_info.get('neu_loss', 0):.2e} | "
                f"E={ee:.4e} | RelL2={rel:.4e} RelE={rel_en:.4e} | LR={lr:.2e}"
            )

            plot_training_metrics(
                self.history.pretrain.as_dict(),
                self.domain_name,
                os.path.join(OUTPUT_DIR, f"{self.domain_name}_metrics.png"),
            )

        if epoch == 1 or epoch % self.plot_every == 0:
            self._plot_fields(epoch)

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

def gradient(v: torch.Tensor, xy: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=create_graph)[0]
