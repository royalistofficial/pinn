from __future__ import annotations
import abc
import math
from typing import Dict, Type
import numpy as np

class BaseDomain(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def boundary_vertices(self) -> np.ndarray: ...

    @abc.abstractmethod
    def boundary_segments(self) -> np.ndarray: ...

    @abc.abstractmethod
    def bc_type(self, edge_idx: int) -> float: ...

    @abc.abstractmethod
    def holes(self) -> np.ndarray: ...

    @property
    def has_neumann(self) -> bool:
        bs = self.boundary_segments()
        return any(self.bc_type(i) < 0.5 for i in range(len(bs)))

class SquareDomain(BaseDomain):
    @property
    def name(self): return "square"
    def boundary_vertices(self):
        return np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
    def boundary_segments(self):
        return np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
    def bc_type(self, i): return 1.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class SquareMixedDomain(BaseDomain):
    @property
    def name(self): return "square_mixed"
    def boundary_vertices(self):
        return np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
    def boundary_segments(self):
        return np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
    def bc_type(self, i): return 1.0 if i in (0,1) else 0.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class CircleDomain(BaseDomain):
    def __init__(self, n_bd=64): self._n = n_bd
    @property
    def name(self): return "circle"
    def boundary_vertices(self):
        t = np.linspace(0, 2*math.pi, self._n, endpoint=False)
        return np.column_stack([np.cos(t), np.sin(t)])
    def boundary_segments(self):
        n = self._n
        return np.column_stack([np.arange(n), (np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i): return 1.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class CircleMixedDomain(BaseDomain):
    def __init__(self, n_bd=64): self._n = n_bd
    @property
    def name(self): return "circle_mixed"
    def boundary_vertices(self):
        t = np.linspace(0, 2*math.pi, self._n, endpoint=False)
        return np.column_stack([np.cos(t), np.sin(t)])
    def boundary_segments(self):
        n = self._n
        return np.column_stack([np.arange(n), (np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i):
        t = np.linspace(0, 2*math.pi, self._n, endpoint=False)
        mid_angle = 0.5*(t[i] + t[(i+1)%self._n])
        return 1.0 if math.sin(mid_angle) > 0 else 0.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class LShapeDomain(BaseDomain):
    @property
    def name(self): return "l_shape"
    def boundary_vertices(self):
        return np.array([[-1,-1],[0,-1],[0,0],[1,0],[1,1],[-1,1]], dtype=np.float64)
    def boundary_segments(self):
        n = 6; return np.column_stack([np.arange(n),(np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i): return 1.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class LShapeMixedDomain(BaseDomain):
    @property
    def name(self): return "l_shape_mixed"
    def boundary_vertices(self):
        return np.array([[-1,-1],[0,-1],[0,0],[1,0],[1,1],[-1,1]], dtype=np.float64)
    def boundary_segments(self):
        n = 6; return np.column_stack([np.arange(n),(np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i): return 1.0 if i in (0,3,4,5) else 0.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class HollowSquareDomain(BaseDomain):
    @property
    def name(self): return "hollow_square"
    def boundary_vertices(self):
        o = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
        i = np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], dtype=np.float64)
        return np.vstack([o, i])
    def boundary_segments(self):
        return np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]], dtype=np.int32)
    def bc_type(self, i): return 1.0
    def holes(self): return np.array([[0.0,0.0]], dtype=np.float64)

class HollowSquareMixedDomain(BaseDomain):
    @property
    def name(self): return "hollow_square_mixed"
    def boundary_vertices(self):
        o = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
        i = np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]], dtype=np.float64)
        return np.vstack([o, i])
    def boundary_segments(self):
        return np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]], dtype=np.int32)
    def bc_type(self, i): return 1.0 if i < 4 else 0.0
    def holes(self): return np.array([[0.0,0.0]], dtype=np.float64)

class PShapeDomain(BaseDomain):
    @property
    def name(self): return "p_shape"
    def boundary_vertices(self):
        return np.array([[0.5,-1],[1,-1],[1,1],[-1,1],[-1,-1],[-0.5,-1],[-0.5,0.5],[0.5,0.5]], dtype=np.float64)
    def boundary_segments(self):
        n = 8; return np.column_stack([np.arange(n),(np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i): return 1.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

class PShapeMixedDomain(BaseDomain):
    @property
    def name(self): return "p_shape_mixed"
    def boundary_vertices(self):
        return np.array([[0.5,-1],[1,-1],[1,1],[-1,1],[-1,-1],[-0.5,-1],[-0.5,0.5],[0.5,0.5]], dtype=np.float64)
    def boundary_segments(self):
        n = 8; return np.column_stack([np.arange(n),(np.arange(n)+1)%n]).astype(np.int32)
    def bc_type(self, i): return 1.0 if i < 5 else 0.0
    def holes(self): return np.zeros((0,2), dtype=np.float64)

DOMAIN_REGISTRY: Dict[str, Type[BaseDomain]] = {
    "square": SquareDomain, "square_mixed": SquareMixedDomain,
    "circle": CircleDomain, "circle_mixed": CircleMixedDomain,
    "l_shape": LShapeDomain, "l_shape_mixed": LShapeMixedDomain,
    "hollow_square": HollowSquareDomain, "hollow_square_mixed": HollowSquareMixedDomain,
    "p_shape": PShapeDomain, "p_shape_mixed": PShapeMixedDomain,
}

def make_domain(name: str, **kwargs) -> BaseDomain:
    if name not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain '{name}'. Available: {list(DOMAIN_REGISTRY)}")
    return DOMAIN_REGISTRY[name](**kwargs)
