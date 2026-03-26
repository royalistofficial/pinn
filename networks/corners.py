from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from geometry.domains import BaseDomain
from geometry.mesher import get_inside_mask

class CornerEnrichment(nn.Module):
    def __init__(self, corners: torch.Tensor, angles: torch.Tensor, n_harmonics: int = 4):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.features_per_corner = 1 + 2 * n_harmonics
        n_corners = corners.shape[0]
        self.register_buffer("corners", corners.float())
        init_alpha = math.pi / angles.float().clamp(min=0.1)
        self.log_alpha = nn.Parameter(torch.log(init_alpha))
        self.corner_scales = nn.Parameter(torch.zeros(n_corners))
        self.out_dim = n_corners * self.features_per_corner

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def forward(self, xy):
        delta = xy.unsqueeze(1) - self.corners.unsqueeze(0)
        r = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-12)
        theta = torch.atan2(delta[..., 1], delta[..., 0])
        alpha = self.alpha
        scales = torch.sigmoid(self.corner_scales)
        parts = [r.pow(alpha.unsqueeze(0))]
        for k in range(1, self.n_harmonics + 1):
            r_ka = r.pow(k * alpha.unsqueeze(0))
            kt = k * theta
            parts.append(r_ka * torch.cos(kt))
            parts.append(r_ka * torch.sin(kt))
        features = torch.stack(parts, dim=-1)
        features = features * scales.unsqueeze(0).unsqueeze(-1)
        return features.reshape(xy.shape[0], -1)

def extract_corners(domain: BaseDomain, angle_threshold_deg: float = 170.0
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    bv = domain.boundary_vertices()
    bs = domain.boundary_segments()
    vertex_edges: dict = {}
    for ei, (i0, i1) in enumerate(bs):
        vertex_edges.setdefault(i0, []).append((i1, ei))
        vertex_edges.setdefault(i1, []).append((i0, ei))
    corners, angles = [], []
    for vi in range(len(bv)):
        if vi not in vertex_edges:
            continue
        neighbors = vertex_edges[vi]
        if len(neighbors) < 2:
            continue
        p = bv[vi]
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                v1 = bv[neighbors[j][0]] - p
                v2 = bv[neighbors[k][0]] - p
                l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if l1 < 1e-12 or l2 < 1e-12:
                    continue
                cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
                angle = math.acos(cos_a)
                bisector = v1/l1 + v2/l2
                blen = np.linalg.norm(bisector)
                if blen > 1e-8:
                    test_pt = p + (bisector/blen) * 1e-5
                    is_inside = get_inside_mask(np.array([test_pt]), bv, bs)[0]
                    interior_angle = angle if is_inside else (2*math.pi - angle)
                else:
                    interior_angle = angle
                if abs(math.degrees(interior_angle) - 180.0) > (180.0 - angle_threshold_deg):
                    corners.append(p.copy())
                    angles.append(interior_angle)
    if not corners:
        return torch.zeros(0, 2), torch.zeros(0)
    corners_arr = np.array(corners)
    angles_arr = np.array(angles)
    _, unique_idx = np.unique(np.round(corners_arr, decimals=8), axis=0, return_index=True)
    return (torch.tensor(corners_arr[unique_idx], dtype=torch.float32),
            torch.tensor(angles_arr[unique_idx], dtype=torch.float32))

def build_corner_enrichment(domain: BaseDomain, device: torch.device
                            ) -> CornerEnrichment | None:
    corners, angles = extract_corners(domain)
    if corners.shape[0] == 0:
        return None
    enrichment = CornerEnrichment(corners.to(device), angles.to(device))
    print(f"[CornerEnrichment] {corners.shape[0]} corners, "
          f"angles (deg): {[f'{math.degrees(a):.1f}' for a in angles.tolist()]}, "
          f"init alpha: {[f'{a:.3f}' for a in enrichment.alpha.tolist()]}")
    return enrichment
