from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class WeightConfig:
    enabled: bool = False
    method: str = "ntk_trace"
    update_every: int = 100
    momentum: float = 0.9
    min_weight: float = 0.1
    max_weight: float = 10.0

    w_pde_init: float = 1.0
    w_dirichlet_init: float = 1.0
    w_neumann_init: float = 1.0

class WeightBalancer:
    def __init__(self, config: Optional[WeightConfig] = None):
        self.config = config or WeightConfig()
        self.reset()

    def reset(self) -> None:
        self.w_pde = self.config.w_pde_init
        self.w_dirichlet = self.config.w_dirichlet_init
        self.w_neumann = self.config.w_neumann_init

        self._trace_history: Dict[str, list] = {
            "pde": [], "dir": [], "neu": []
        }
        self._grad_history: Dict[str, list] = {
            "pde": [], "dir": [], "neu": []
        }
        self._smooth_trace: Dict[str, float] = {
            "pde": 1.0, "dir": 1.0, "neu": 1.0
        }

    def update_from_ntk(
                self,
                trace_pde: float,
                trace_dirichlet: float,
                trace_neumann: float = 0.0,
            ) -> Tuple[float, float, float]:
        if not self.config.enabled:
            return self.w_pde, self.w_dirichlet, self.w_neumann

        if self.config.method != "ntk_trace":
            return self.w_pde, self.w_dirichlet, self.w_neumann

        self._trace_history["pde"].append(trace_pde)
        self._trace_history["dir"].append(trace_dirichlet)
        self._trace_history["neu"].append(trace_neumann)

        alpha = 1 - self.config.momentum

        self._smooth_trace["pde"] = alpha * trace_pde + (1 - alpha) * self._smooth_trace["pde"]
        self._smooth_trace["dir"] = alpha * trace_dirichlet + (1 - alpha) * self._smooth_trace["dir"]
        self._smooth_trace["neu"] = alpha * trace_neumann + (1 - alpha) * self._smooth_trace["neu"]

        eps = 1e-10
        inv_sqrt = {
            "pde": 1.0 / np.sqrt(self._smooth_trace["pde"] + eps),
            "dir": 1.0 / np.sqrt(self._smooth_trace["dir"] + eps),
            "neu": 1.0 / np.sqrt(self._smooth_trace["neu"] + eps) if trace_neumann > eps else 0.0,
        }

        total = inv_sqrt["pde"] + inv_sqrt["dir"] + inv_sqrt["neu"]
        if total < eps:
            total = 1.0

        new_w_pde = inv_sqrt["pde"] / total * 3
        new_w_dir = inv_sqrt["dir"] / total * 3
        new_w_neu = inv_sqrt["neu"] / total * 3 if trace_neumann > eps else 0.0

        new_w_pde = np.clip(new_w_pde, self.config.min_weight, self.config.max_weight)
        new_w_dir = np.clip(new_w_dir, self.config.min_weight, self.config.max_weight)
        new_w_neu = np.clip(new_w_neu, self.config.min_weight, self.config.max_weight) if trace_neumann > eps else 0.0

        self.w_pde = alpha * new_w_pde + (1 - alpha) * self.w_pde
        self.w_dirichlet = alpha * new_w_dir + (1 - alpha) * self.w_dirichlet
        self.w_neumann = alpha * new_w_neu + (1 - alpha) * self.w_neumann if trace_neumann > eps else 0.0

        return self.w_pde, self.w_dirichlet, self.w_neumann

    def update_from_grad_norms(
                self,
                grad_norm_pde: float,
                grad_norm_dirichlet: float,
                grad_norm_neumann: float = 0.0,
            ) -> Tuple[float, float, float]:
        if not self.config.enabled:
            return self.w_pde, self.w_dirichlet, self.w_neumann

        if self.config.method != "grad_norm":
            return self.w_pde, self.w_dirichlet, self.w_neumann

        self._grad_history["pde"].append(grad_norm_pde)
        self._grad_history["dir"].append(grad_norm_dirichlet)
        self._grad_history["neu"].append(grad_norm_neumann)

        alpha = 1 - self.config.momentum

        smooth_grad = {
            "pde": alpha * grad_norm_pde + (1 - alpha) * (self._grad_history["pde"][-2] if len(self._grad_history["pde"]) > 1 else grad_norm_pde),
            "dir": alpha * grad_norm_dirichlet + (1 - alpha) * (self._grad_history["dir"][-2] if len(self._grad_history["dir"]) > 1 else grad_norm_dirichlet),
            "neu": alpha * grad_norm_neumann + (1 - alpha) * (self._grad_history["neu"][-2] if len(self._grad_history["neu"]) > 1 else grad_norm_neumann),
        }

        eps = 1e-10
        inv_norm = {
            "pde": 1.0 / (smooth_grad["pde"] + eps),
            "dir": 1.0 / (smooth_grad["dir"] + eps),
            "neu": 1.0 / (smooth_grad["neu"] + eps) if grad_norm_neumann > eps else 0.0,
        }

        total = inv_norm["pde"] + inv_norm["dir"] + inv_norm["neu"]
        if total < eps:
            total = 1.0

        self.w_pde = np.clip(inv_norm["pde"] / total * 3,
                            self.config.min_weight, self.config.max_weight)
        self.w_dirichlet = np.clip(inv_norm["dir"] / total * 3,
                                  self.config.min_weight, self.config.max_weight)
        self.w_neumann = np.clip(inv_norm["neu"] / total * 3,
                                self.config.min_weight, self.config.max_weight) if grad_norm_neumann > eps else 0.0

        return self.w_pde, self.w_dirichlet, self.w_neumann

    def get_weights(self) -> Dict[str, float]:
        return {
            "pde": self.w_pde,
            "dirichlet": self.w_dirichlet,
            "neumann": self.w_neumann,
        }

    def get_diagnostics(self) -> Dict[str, any]:
        return {
            "weights": self.get_weights(),
            "trace_history": {k: v[-10:] if v else [] for k, v in self._trace_history.items()},
            "smooth_trace": self._smooth_trace,
            "config": {
                "enabled": self.config.enabled,
                "method": self.config.method,
                "update_every": self.config.update_every,
            },
        }
