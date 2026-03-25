from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch

from geometry.domains import BaseDomain

@dataclass
class QuadratureData:
    xy_in: torch.Tensor       
    vol_w: torch.Tensor        
    tri_indices: torch.Tensor  

    xy_bd: torch.Tensor        
    surf_w: torch.Tensor       
    normals: torch.Tensor      
    bc_mask: torch.Tensor      
    tri_indices_bd: torch.Tensor  

    mesh: Dict[str, np.ndarray]

def ref_triangle_gauss(order: int) -> Tuple[np.ndarray, np.ndarray]:
    if order <= 1:
        pts = np.array([[1/3, 1/3]])
        wts = np.array([0.5])
    elif order == 2:
        pts = np.array([[1/6,1/6],[2/3,1/6],[1/6,2/3]])
        wts = np.full(3, 1/6)
    elif order in (3, 4):
        a1, a2 = 0.445948490915965, 0.091576213509771
        b1, b2 = 0.108103018168070, 0.816847572980459
        w1, w2 = 0.111690794839005, 0.054975871827661
        pts = np.array([[a1,a1],[b1,a1],[a1,b1],[a2,a2],[b2,a2],[a2,b2]])
        wts = np.array([w1,w1,w1,w2,w2,w2])
    else:
        sqrt15 = math.sqrt(15.0)
        b1 = (6.0 + sqrt15) / 21.0
        a1 = 1.0 - 2.0 * b1
        w1 = (155.0 + sqrt15) / 2400.0
        b2 = (6.0 - sqrt15) / 21.0
        a2 = 1.0 - 2.0 * b2
        w2 = (155.0 - sqrt15) / 2400.0
        pts = np.array([[1/3,1/3],[a1,b1],[b1,a1],[b1,b1],[a2,b2],[b2,a2],[b2,b2]])
        wts = np.array([9.0/80.0, w1,w1,w1, w2,w2,w2])
    return pts.astype(np.float64), wts.astype(np.float64)

def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (t + 1), 0.5 * w

class QuadratureBuilder:
    def __init__(self, tri_order: int = 5, line_order: int = 7,
                 device: torch.device = None):
        self.tri_order = tri_order
        self.line_order = line_order
        self.device = device or torch.device("cpu")

    def build(self, mesh: Dict[str, np.ndarray], domain: BaseDomain) -> QuadratureData:
        xy_in, vol_w, tri_idx = self._build_interior(mesh["points"], mesh["triangles"])
        xy_bd, surf_w, normals, bc_mask = self._build_boundary(domain, mesh)

        tri_indices_bd = self._locate_boundary_points(
            xy_bd, mesh["points"], mesh["triangles"]
        )

        return QuadratureData(
            xy_in=torch.tensor(xy_in, dtype=torch.float32, device=self.device),
            vol_w=torch.tensor(vol_w, dtype=torch.float32, device=self.device).unsqueeze(-1),
            tri_indices=torch.tensor(tri_idx, dtype=torch.long, device=self.device),
            xy_bd=torch.tensor(xy_bd, dtype=torch.float32, device=self.device),
            surf_w=torch.tensor(surf_w, dtype=torch.float32, device=self.device).unsqueeze(-1),
            normals=torch.tensor(normals, dtype=torch.float32, device=self.device),
            bc_mask=torch.tensor(bc_mask, dtype=torch.float32, device=self.device).unsqueeze(-1),
            tri_indices_bd=torch.tensor(tri_indices_bd, dtype=torch.long, device=self.device),
            mesh=mesh,
        )

    def _locate_boundary_points(self, xy_bd: np.ndarray,
                                pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
        from scipy.spatial import Delaunay
        tri_del = Delaunay(pts)
        simplex_idx = tri_del.find_simplex(xy_bd)

        bad = simplex_idx < 0
        if np.any(bad):
            centroids = pts[tris].mean(axis=1)
            for i in np.where(bad)[0]:
                dists = np.sum((centroids - xy_bd[i]) ** 2, axis=1)
                simplex_idx[i] = np.argmin(dists)

        return np.clip(simplex_idx, 0, len(tris) - 1).astype(np.int64)

    def _build_interior(self, pts, tris):
        ref_pts, ref_wts = ref_triangle_gauss(self.tri_order)
        nq = len(ref_wts)

        all_xy, all_w, all_tri = [], [], []
        for t_num, tri_idx in enumerate(tris):
            v0, v1, v2 = pts[tri_idx[0]], pts[tri_idx[1]], pts[tri_idx[2]]
            area = 0.5 * abs((v1[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(v1[1]-v0[1]))
            if area < 1e-14:
                continue
            for q in range(nq):
                xi, eta = ref_pts[q]
                px = v0[0] + xi*(v1[0]-v0[0]) + eta*(v2[0]-v0[0])
                py = v0[1] + xi*(v1[1]-v0[1]) + eta*(v2[1]-v0[1])
                all_xy.append([px, py])
                all_w.append(2.0 * area * ref_wts[q])
                all_tri.append(t_num)

        return (np.array(all_xy, dtype=np.float64),
                np.array(all_w, dtype=np.float64),
                np.array(all_tri, dtype=np.int64))

    def _build_boundary(self, domain: BaseDomain, mesh: Dict):
        bv = mesh["boundary_vertices"]
        bs = mesh["boundary_segments"]
        gl_t, gl_w = gauss_legendre_01(self.line_order)
        max_sub_len = math.sqrt(self.device.type == "cpu" and 0.05 or 0.05)

        bd_xy, bd_w, bd_n, bd_bc = [], [], [], []
        for ei, (i0, i1) in enumerate(bs):
            p0, p1 = bv[i0], bv[i1]
            dx, dy = p1[0]-p0[0], p1[1]-p0[1]
            length = math.hypot(dx, dy)
            if length < 1e-14:
                continue
            nx, ny = dy / length, -dx / length
            bc_flag = domain.bc_type(ei)

            n_sub = max(1, int(math.ceil(length / max_sub_len)))
            corner_margin = 1e-4

            for s in range(n_sub):
                t0 = s / n_sub
                t1 = (s + 1) / n_sub
                sub_len = length / n_sub
                for q in range(len(gl_t)):
                    t = t0 + gl_t[q] * (t1 - t0)
                    t_clipped = max(corner_margin, min(1.0 - corner_margin, t))
                    bd_xy.append([p0[0]+t_clipped*dx, p0[1]+t_clipped*dy])
                    bd_w.append(gl_w[q] * sub_len)
                    bd_n.append([nx, ny])
                    bd_bc.append(bc_flag)

        return (np.array(bd_xy, dtype=np.float64),
                np.array(bd_w, dtype=np.float64),
                np.array(bd_n, dtype=np.float64),
                np.array(bd_bc, dtype=np.float64))