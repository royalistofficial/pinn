from __future__ import annotations
import math
from typing import Dict, Set, Tuple
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path

from geometry.domains import BaseDomain

def get_inside_mask(pts: np.ndarray, bv: np.ndarray, bs: np.ndarray) -> np.ndarray:
    adj = {start: end for start, end in bs}
    visited: set = set()
    loops = []

    for start_node in adj:
        if start_node not in visited:
            curr = start_node
            loop_verts = [bv[curr]]
            while True:
                visited.add(curr)
                if curr not in adj:
                    break
                curr = adj[curr]
                loop_verts.append(bv[curr])
                if curr == start_node:
                    break
            loops.append(Path(np.array(loop_verts)))

    if not loops:
        return np.zeros(len(pts), dtype=bool)

    outer_loop = max(loops, key=lambda p:
        (p.vertices[:,0].max()-p.vertices[:,0].min()) *
        (p.vertices[:,1].max()-p.vertices[:,1].min()))

    mask = outer_loop.contains_points(pts)
    for loop in loops:
        if loop is not outer_loop:
            mask &= ~loop.contains_points(pts)
    return mask

class Mesher:
    def __init__(self, max_area: float, lloyd_iters: int = 3, boundary_density: int = 100):
        self.max_area = max_area
        self.lloyd_iters = lloyd_iters
        self.boundary_density = boundary_density

    def build(self, domain: BaseDomain) -> Dict[str, np.ndarray]:
        bv = domain.boundary_vertices()
        bs = domain.boundary_segments()

        dense_pts = self._densify_boundary(bv, bs)
        all_pts = dense_pts.copy()

        seeds = self._make_interior_seeds(domain, spacing=math.sqrt(self.max_area) * 0.8)
        if seeds is not None and len(seeds) > 0:
            all_pts = np.vstack([all_pts, seeds])

        boundary_set = set(range(len(dense_pts)))
        all_pts = self._apply_lloyd(all_pts, boundary_set, domain)

        tri = Delaunay(all_pts)
        points, simplices = tri.points, tri.simplices

        simplices = self._clip_triangles(points, simplices, domain)
        simplices = self._filter_degenerate(points, simplices, min_angle_deg=5.0)
        points, simplices = self._remove_unused_vertices(points, simplices)

        return {
            "points": points,
            "triangles": simplices,
            "boundary_vertices": bv,
            "boundary_segments": bs,
        }

    def _densify_boundary(self, bv: np.ndarray, bs: np.ndarray) -> np.ndarray:
        target_pts = self.boundary_density
        lengths = np.array([math.hypot(bv[i1][0]-bv[i0][0], bv[i1][1]-bv[i0][1]) for i0, i1 in bs])
        total_len = np.sum(lengths)

        exact_subs = target_pts * (lengths / total_len)
        n_subs = np.floor(exact_subs).astype(int)
        n_subs[n_subs < 1] = 1

        diff = target_pts - np.sum(n_subs)
        if diff > 0:
            remainders = exact_subs - n_subs
            for idx in np.argsort(remainders)[::-1][:diff]:
                n_subs[idx] += 1
        elif diff < 0:
            for _ in range(abs(diff)):
                valid_idx = np.where(n_subs > 1)[0]
                if len(valid_idx) == 0: break
                longest_valid = valid_idx[np.argmax(lengths[valid_idx])]
                n_subs[longest_valid] -= 1

        new_pts = []
        for (i0, i1), n_sub in zip(bs, n_subs):
            p0, p1 = bv[i0], bv[i1]
            for k in range(n_sub):
                t = k / n_sub
                new_pts.append([p0[0]+t*(p1[0]-p0[0]), p0[1]+t*(p1[1]-p0[1])])
        return np.array(new_pts)

    def _make_interior_seeds(self, domain: BaseDomain, spacing: float) -> np.ndarray | None:
        bv = domain.boundary_vertices()
        bs = domain.boundary_segments()
        xmin, ymin = bv.min(axis=0)
        xmax, ymax = bv.max(axis=0)
        xs = np.arange(xmin + spacing*0.5, xmax, spacing)
        ys = np.arange(ymin + spacing*0.5, ymax, spacing)
        grid = np.array([[x, y] for x in xs for y in ys])
        if len(grid) == 0:
            return None
        inside = get_inside_mask(grid, bv, bs)
        return grid[inside] if inside.any() else None

    def _apply_lloyd(self, all_pts: np.ndarray, boundary_set: Set[int],
                     domain: BaseDomain) -> np.ndarray:
        bv = domain.boundary_vertices()
        bs = domain.boundary_segments()

        for _ in range(self.lloyd_iters):
            tri_obj = Delaunay(all_pts)
            valid_simplices = self._clip_triangles(all_pts, tri_obj.simplices, domain)
            new_pts = all_pts.copy()

            for i in range(len(all_pts)):
                if i in boundary_set:
                    continue
                mask = np.any(valid_simplices == i, axis=1)
                if not mask.any():
                    continue
                neighbor_tris = valid_simplices[mask]
                centroids = all_pts[neighbor_tris].mean(axis=1)
                candidate = centroids.mean(axis=0)
                if get_inside_mask(np.array([candidate]), bv, bs)[0]:
                    new_pts[i] = candidate

            all_pts = new_pts

        tri_final = Delaunay(all_pts)
        valid_final = self._clip_triangles(all_pts, tri_final.simplices, domain)
        all_pts, _ = self._remove_unused_vertices(all_pts, valid_final)
        return all_pts

    def _clip_triangles(self, pts: np.ndarray, simplices: np.ndarray,
                        domain: BaseDomain) -> np.ndarray:
        bv = domain.boundary_vertices()
        bs = domain.boundary_segments()

        centroids = pts[simplices].mean(axis=1)
        keep = get_inside_mask(centroids, bv, bs)

        v0 = pts[simplices[:, 0]]
        v1 = pts[simplices[:, 1]]
        v2 = pts[simplices[:, 2]]
        shift = 0.01
        for mid in [(v0+v1)/2, (v1+v2)/2, (v2+v0)/2]:
            mid_shifted = mid * (1.0 - shift) + centroids * shift
            keep &= get_inside_mask(mid_shifted, bv, bs)
        return simplices[keep]

    def _filter_degenerate(self, pts: np.ndarray, simplices: np.ndarray,
                           min_angle_deg: float = 5.0) -> np.ndarray:
        good = []
        for tri_idx in simplices:
            v0, v1, v2 = pts[tri_idx[0]], pts[tri_idx[1]], pts[tri_idx[2]]
            area = 0.5 * abs((v1[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(v1[1]-v0[1]))
            if area < 1e-14:
                continue
            a = np.linalg.norm(v1 - v0)
            b = np.linalg.norm(v2 - v1)
            c = np.linalg.norm(v0 - v2)
            sides = sorted([a, b, c])
            if sides[0] < 1e-12:
                continue
            sin_min = 2 * area / (sides[1] * sides[2])
            if math.degrees(math.asin(min(sin_min, 1.0))) < min_angle_deg:
                continue
            good.append(tri_idx)
        return np.array(good) if good else simplices

    @staticmethod
    def _remove_unused_vertices(pts: np.ndarray, simplices: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        active = np.unique(simplices)
        if len(active) < len(pts):
            remap = np.zeros(len(pts), dtype=np.int32)
            remap[active] = np.arange(len(active))
            simplices = remap[simplices]
            pts = pts[active]
        return pts, simplices
