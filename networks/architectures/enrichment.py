from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from geometry.domains import BaseDomain
from geometry.mesher import get_inside_mask

class GeometryEnrichment(nn.Module):
    def __init__(
        self,
        corners,
        angles,
        bisectors,
        corner_types,
        segments,
        n_harmonics=4,
        n_boundary=2,
    ):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.n_boundary = n_boundary

        self.register_buffer("corners", corners)
        self.register_buffer("angles", angles)
        self.register_buffer("bisectors", bisectors)
        self.register_buffer("corner_types", corner_types)

        self.log_alpha = nn.Parameter(torch.log(math.pi / angles.clamp(min=1e-3)))
        self.log_corner_scale = nn.Parameter(torch.zeros(len(corners)))

        self.register_buffer("segments", segments)  

        self.log_beta = nn.Parameter(torch.zeros(len(segments)))

        self.out_dim = (
            len(corners) * 2 * n_harmonics
            + len(segments) * 2 * n_boundary
        )

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, xy: torch.Tensor):

        N = xy.shape[0]

        Nc = self.corners.shape[0]

        delta = xy.unsqueeze(1) - self.corners.unsqueeze(0)

        r = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-12).clamp(min=1e-6)

        ex = self.bisectors.unsqueeze(0)
        ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)

        dx = (delta * ex).sum(dim=-1)
        dy = (delta * ey).sum(dim=-1)

        cos_theta = (dx / r).clamp(-1.0, 1.0)
        sin_theta = (dy / r).clamp(-1.0, 1.0)

        alpha = self.alpha
        scale = torch.exp(self.log_corner_scale)

        eff_alpha = torch.where(
            self.corner_types == 1,
            alpha,
            torch.clamp(alpha, min=1.0),
        )

        weight = torch.where(
            self.corner_types == 1,
            1.0,
            0.3,
        )

        decay = torch.exp(-scale.unsqueeze(0) * r)

        corner_feat = []

        cos_k = cos_theta
        sin_k = sin_theta

        cos_prev = torch.ones_like(cos_theta)
        sin_prev = torch.zeros_like(sin_theta)

        for k in range(1, self.n_harmonics + 1):

            k_alpha = k * eff_alpha

            r_pow = torch.exp(k_alpha.unsqueeze(0) * torch.log(r))

            f1 = r_pow * sin_k * decay
            f2 = r_pow * cos_k * decay

            f = torch.stack([f1, f2], dim=-1)
            f = f * weight.unsqueeze(0).unsqueeze(-1)

            corner_feat.append(f)

            cos_next = 2 * cos_theta * cos_k - cos_prev
            sin_next = 2 * cos_theta * sin_k - sin_prev

            cos_prev, cos_k = cos_k, cos_next
            sin_prev, sin_k = sin_k, sin_next

        corner_feat = torch.cat(corner_feat, dim=-1).view(N, -1)

        Ns = self.segments.shape[0]

        p0 = self.segments[:, 0]
        p1 = self.segments[:, 1]

        seg_vec = p1 - p0
        seg_len = torch.norm(seg_vec, dim=-1).clamp(min=1e-6)

        t_dir = seg_vec / seg_len.unsqueeze(-1)
        n_dir = torch.stack([-t_dir[:, 1], t_dir[:, 0]], dim=-1)

        rel = xy.unsqueeze(1) - p0.unsqueeze(0)

        t_coord = (rel * t_dir.unsqueeze(0)).sum(dim=-1)
        d = torch.abs((rel * n_dir.unsqueeze(0)).sum(dim=-1))

        beta = self.beta

        boundary_feat = []

        cos_k = torch.cos(t_coord)
        sin_k = torch.sin(t_coord)

        cos_prev = torch.ones_like(t_coord)
        sin_prev = torch.zeros_like(t_coord)

        decay = torch.exp(-beta.unsqueeze(0) * d)

        for k in range(1, self.n_boundary + 1):

            f1 = sin_k * decay
            f2 = cos_k * decay

            f = torch.stack([f1, f2], dim=-1)
            boundary_feat.append(f)

            cos_next = 2 * torch.cos(t_coord) * cos_k - cos_prev
            sin_next = 2 * torch.cos(t_coord) * sin_k - sin_prev

            cos_prev, cos_k = cos_k, cos_next
            sin_prev, sin_k = sin_k, sin_next

        boundary_feat = torch.cat(boundary_feat, dim=-1).view(N, -1)

        return torch.cat([xy, corner_feat, boundary_feat], dim=-1)

def extract_geometry(domain: BaseDomain):

    bv = domain.boundary_vertices()
    bs = domain.boundary_segments()

    corners, angles, bisectors, types = [], [], [], []

    for vi in range(len(bv)):
        neighbors = [j for (i, j) in bs if i == vi] + [i for (i, j) in bs if j == vi]

        if len(neighbors) < 2:
            continue

        p = bv[vi]

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):

                v1 = bv[neighbors[i]] - p
                v2 = bv[neighbors[j]] - p

                l1 = np.linalg.norm(v1)
                l2 = np.linalg.norm(v2)

                if l1 < 1e-12 or l2 < 1e-12:
                    continue

                angle = math.acos(np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1))

                bis = v1 / l1 + v2 / l2
                bl = np.linalg.norm(bis)

                if bl < 1e-8:
                    continue

                bis /= bl

                test = p + bis * 1e-5
                inside = get_inside_mask(np.array([test]), bv, bs)[0]

                interior = angle if inside else (2 * math.pi - angle)

                corners.append(p.copy())
                angles.append(interior)
                bisectors.append(bis)
                types.append(1 if interior > math.pi else 0)

    segments = np.array([[bv[i], bv[j]] for (i, j) in bs])

    corners = np.array(corners, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)
    bisectors = np.array(bisectors, dtype=np.float32)
    types = np.array(types, dtype=np.int64)
    segments = np.array(segments, dtype=np.float32)

    return (
        torch.from_numpy(corners),
        torch.from_numpy(angles),
        torch.from_numpy(bisectors),
        torch.from_numpy(types),
        torch.from_numpy(segments),
    )

def build_enrichment(domain, device):

    c, a, b, t, s = extract_geometry(domain)

    if len(c) == 0:
        print("No corners found")

    return GeometryEnrichment(
        c.to(device),
        a.to(device),
        b.to(device),
        t.to(device),
        s.to(device),
    )