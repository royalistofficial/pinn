from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class WeightConfig:
    enabled: bool = False
    update_every: int = 100
    momentum: float = 0.95
    min_weight = 1e-2
    max_weight = 1e+3

    w_pde_init: float = 1.0
    w_dirichlet_init: float = 1.0
    w_neumann_init: float = 1.0

class WeightBalancer:
    def __init__(self, config: Optional[WeightConfig] = None):
        self.config = config or WeightConfig()
        self.reset()

    def reset(self) -> None:

        self.w_pde = 1.0
        self.w_dirichlet = 1.0 
        self.w_neumann = 1.0

        self._trace_history: Dict[str, list] = {"pde": [], "dir": [], "neu": []}
        self._smooth_norm: Dict[str, float] = {"pde": 1.0, "dir": 1.0, "neu": 1.0}

    def update_from_gradients(
                self,
                grad_pde_norm: float,
                grad_dir_norm: float,
                grad_neu_norm: float = 0.0,
                bc_penalty: float = 1.0 
            ) -> Tuple[float, float, float]:

        if not self.config.enabled:
            self.w_pde = 1.0
            self.w_dirichlet = bc_penalty
            self.w_neumann = bc_penalty if grad_neu_norm > 0 or self.w_neumann > 0 else 0.0
            return self.w_pde, self.w_dirichlet, self.w_neumann

        self._trace_history["pde"].append(grad_pde_norm)
        self._trace_history["dir"].append(grad_dir_norm)
        self._trace_history["neu"].append(grad_neu_norm)

        alpha = 1 - self.config.momentum

        self._smooth_norm["pde"] = alpha * grad_pde_norm + (1 - alpha) * self._smooth_norm["pde"]
        self._smooth_norm["dir"] = alpha * grad_dir_norm + (1 - alpha) * self._smooth_norm["dir"]
        self._smooth_norm["neu"] = alpha * grad_neu_norm + (1 - alpha) * self._smooth_norm["neu"]

        eps = 1e-10

        max_grad = self._smooth_norm["pde"]

        new_w_pde = max_grad / (self._smooth_norm["pde"] + eps)
        new_w_dir = max_grad / (self._smooth_norm["dir"] + eps)

        self.w_pde = np.clip(alpha * new_w_pde + (1 - alpha) * self.w_pde, self.config.min_weight, self.config.max_weight)
        self.w_dirichlet = np.clip(alpha * new_w_dir + (1 - alpha) * self.w_dirichlet, self.config.min_weight, self.config.max_weight)

        if grad_neu_norm > eps:
            new_w_neu = max_grad / (self._smooth_norm["neu"] + eps)
            self.w_neumann = np.clip(alpha * new_w_neu + (1 - alpha) * self.w_neumann, self.config.min_weight, self.config.max_weight)
        else:
            self.w_neumann = 0.0

        return self.w_pde, self.w_dirichlet, self.w_neumann

    def get_weights(self) -> Dict[str, float]:
        return {
            "pde": self.w_pde,
            "dirichlet": self.w_dirichlet,
            "neumann": self.w_neumann,
        }
