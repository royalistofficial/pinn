from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PhaseMetrics:
    epoch: List[int] = field(default_factory=list)
    data: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, epoch: int, **kwargs: float) -> None:
        self.epoch.append(epoch)
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def as_dict(self) -> Dict[str, list]:
        return {"epoch": self.epoch, **self.data}

    def last(self, key: str, default: float = 0.0) -> float:
        vals = self.data.get(key, [])
        return vals[-1] if vals else default

    def __len__(self) -> int:
        return len(self.epoch)

class MetricsHistory:
    PRETRAIN_KEYS = [
        "loss", "pde", "dir_loss", "neu_loss",
        "energy", "rel_err", "rel_energy", "lr",
    ]

    def __init__(self):
        self.pretrain = PhaseMetrics()
