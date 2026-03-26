from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from geometry.domains import BaseDomain
from geometry.mesher import Mesher, get_inside_mask

@dataclass
class FEMMesh:
    points: np.ndarray        
    elements: np.ndarray      
    boundary_nodes: np.ndarray  
    boundary_edges: np.ndarray  
    boundary_edge_types: np.ndarray  

@dataclass
class FEMResult:
    u: np.ndarray             
    mesh: FEMMesh             
    h_max: float              
    h_min: float              
    n_dof: int                

def build_fem_mesh(domain: BaseDomain, max_area: float = 0.01,
                   boundary_density: int = 100) -> FEMMesh:
    mesher = Mesher(max_area=max_area, lloyd_iters=3, boundary_density=boundary_density)
    mesh_data = mesher.build(domain)

    points = mesh_data["points"]
    elements = mesh_data["triangles"]
    bv = mesh_data["boundary_vertices"]
    bs = mesh_data["boundary_segments"]

    boundary_edges, edge_types = _extract_boundary_edges(
        points, elements, bv, bs, domain
    )
    boundary_nodes = np.unique(boundary_edges)

    return FEMMesh(
        points=points,
        elements=elements,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        boundary_edge_types=edge_types,
    )

def _extract_boundary_edges(
        points: np.ndarray, elements: np.ndarray,
        bv: np.ndarray, bs: np.ndarray, domain: BaseDomain
    ) -> Tuple[np.ndarray, np.ndarray]:

    edge_count: Dict[Tuple[int, int], int] = {}
    for tri in elements:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges_list = []
    for e, count in edge_count.items():
        if count == 1:
            boundary_edges_list.append(e)

    boundary_edges = np.array(boundary_edges_list, dtype=np.int32) if boundary_edges_list else np.zeros((0, 2), dtype=np.int32)

    edge_types = np.ones(len(boundary_edges), dtype=np.float64)
    for i, (n0, n1) in enumerate(boundary_edges):
        mid = 0.5 * (points[n0] + points[n1])

        best_dist = float('inf')
        best_type = 1.0
        for ei, (si0, si1) in enumerate(bs):
            p0, p1 = bv[si0], bv[si1]
            seg_mid = 0.5 * (p0 + p1)
            dist = np.linalg.norm(mid - seg_mid)

            d = p1 - p0
            L = np.linalg.norm(d)
            if L > 1e-14:
                t = np.clip(np.dot(mid - p0, d) / (L * L), 0, 1)
                proj = p0 + t * d
                dist = np.linalg.norm(mid - proj)

            if dist < best_dist:
                best_dist = dist
                best_type = domain.bc_type(ei)

        edge_types[i] = best_type

    return boundary_edges, edge_types

class FEMSolver:
    def __init__(
        self,
        mesh: FEMMesh,
        f_func: Callable[[float, float], float],
        u_exact_func: Optional[Callable[[float, float], float]] = None,
        grad_exact_func: Optional[Callable[[float, float], Tuple[float, float]]] = None,
    ):
        self.mesh = mesh
        self._f = f_func
        self._u_exact = u_exact_func
        self._grad_exact = grad_exact_func
        self.result: Optional[FEMResult] = None

    def solve(self) -> FEMResult:
        N = len(self.mesh.points)
        K = lil_matrix((N, N))
        F = np.zeros(N)

        for elem in self.mesh.elements:
            coords = self.mesh.points[elem]
            Ke, Fe = self._local_matrices(coords)
            for i in range(3):
                I = elem[i]
                F[I] += Fe[i]
                for j in range(3):
                    J = elem[j]
                    K[I, J] += Ke[i, j]

        for idx, edge in enumerate(self.mesh.boundary_edges):
            if self.mesh.boundary_edge_types[idx] < 0.5:  
                self._apply_neumann_edge(edge, F)

        K_csr = K.tocsr()
        K_lil = K_csr.tolil()
        for node in self.mesh.boundary_nodes:
            if self._is_dirichlet_node(node):
                x, y = self.mesh.points[node]
                value = self._u_exact(x, y) if self._u_exact else 0.0
                F -= value * np.array(K_csr[:, node].todense()).flatten()
                K_lil[node, :] = 0.0
                K_lil[:, node] = 0.0
                K_lil[node, node] = 1.0
                F[node] = value

        K_final = K_lil.tocsr()
        u = spsolve(K_final, F)

        h_max, h_min = self._compute_mesh_sizes()

        self.result = FEMResult(
            u=u,
            mesh=self.mesh,
            h_max=h_max,
            h_min=h_min,
            n_dof=N,
        )
        return self.result

    def _local_matrices(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        A = np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]])
        area = 0.5 * np.linalg.det(A)

        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.vstack([b, c]) / (2 * area)

        Ke = abs(area) * (B.T @ B)

        xc, yc = np.mean(coords, axis=0)
        fval = self._f(xc, yc)
        Fe = np.full(3, fval * abs(area) / 3.0)

        return Ke, Fe

    def _apply_neumann_edge(self, edge: np.ndarray, F: np.ndarray):
        p1 = self.mesh.points[edge[0]]
        p2 = self.mesh.points[edge[1]]
        length = np.linalg.norm(p2 - p1)

        if self._grad_exact is not None:

            mid = 0.5 * (p1 + p2)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            nx, ny = dy / length, -dx / length
            gx, gy = self._grad_exact(mid[0], mid[1])
            g_N = gx * nx + gy * ny
        else:
            g_N = 0.0

        F[edge[0]] += g_N * length / 2
        F[edge[1]] += g_N * length / 2

    def _is_dirichlet_node(self, node: int) -> bool:
        for idx, edge in enumerate(self.mesh.boundary_edges):
            if node in edge and self.mesh.boundary_edge_types[idx] > 0.5:
                return True
        return False

    def _compute_mesh_sizes(self) -> Tuple[float, float]:
        h_max = 0.0
        h_min = float('inf')
        for elem in self.mesh.elements:
            coords = self.mesh.points[elem]
            for i in range(3):
                for j in range(i+1, 3):
                    d = np.linalg.norm(coords[i] - coords[j])
                    h_max = max(h_max, d)
                    h_min = min(h_min, d)
        return h_max, h_min

    def compute_errors(self) -> Dict[str, float]:
        if self.result is None:
            raise RuntimeError("Сначала вызовите solve()")
        if self._u_exact is None:
            return {}

        l2_err_sq = 0.0
        h1_err_sq = 0.0
        l2_norm_sq = 0.0
        h1_norm_sq = 0.0

        for elem in self.mesh.elements:
            coords = self.mesh.points[elem]
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]

            area = 0.5 * abs(
                (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            )
            if area < 1e-14:
                continue

            qpts = [
                (coords[0] + coords[1]) / 2,
                (coords[1] + coords[2]) / 2,
                (coords[2] + coords[0]) / 2,
            ]

            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            u_elem = self.result.u[elem]
            grad_uh_x = np.dot(b, u_elem) / (2 * area)
            grad_uh_y = np.dot(c, u_elem) / (2 * area)

            for qp in qpts:
                xq, yq = qp
                u_ex = self._u_exact(xq, yq)

                det = (y2 - y3) * (xq - x3) + (x3 - x2) * (yq - y3)
                denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if abs(denom) < 1e-14:
                    continue
                l1 = det / denom
                l2_bc = ((y3 - y1) * (xq - x3) + (x1 - x3) * (yq - y3)) / denom
                l3 = 1 - l1 - l2_bc
                u_h = l1 * u_elem[0] + l2_bc * u_elem[1] + l3 * u_elem[2]

                l2_err_sq += (u_h - u_ex) ** 2 * area / 3.0
                l2_norm_sq += u_ex ** 2 * area / 3.0

                if self._grad_exact:
                    gx, gy = self._grad_exact(xq, yq)
                    h1_err_sq += ((grad_uh_x - gx)**2 + (grad_uh_y - gy)**2) * area / 3.0
                    h1_norm_sq += (gx**2 + gy**2) * area / 3.0

        results = {
            "l2_error": math.sqrt(l2_err_sq),
            "relative_l2": math.sqrt(l2_err_sq / max(l2_norm_sq, 1e-30)),
            "h_max": self.result.h_max,
            "n_dof": self.result.n_dof,
        }

        if self._grad_exact:
            results["energy_error"] = math.sqrt(h1_err_sq)
            results["relative_energy"] = math.sqrt(h1_err_sq / max(h1_norm_sq, 1e-30))

        return results
