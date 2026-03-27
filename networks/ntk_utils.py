from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import numpy as np

def compute_jacobian(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()

    N = X.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    J = torch.zeros(N, P, device=X.device)

    with torch.enable_grad():
        for i in range(N):
            model.zero_grad()
            xi = X[i:i + 1].detach().clone().requires_grad_(False)
            out = model(xi).squeeze()
            out.backward(retain_graph=False)

            row = []
            for p in params:
                if p.grad is not None:
                    row.append(p.grad.detach().clone().flatten())
                    p.grad.zero_()
                else:
                    row.append(torch.zeros(p.numel(), device=X.device))
            J[i] = torch.cat(row)

    if was_training:
        model.train()
    return J

def compute_pde_jacobian(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()

    N = X.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    J_L = torch.zeros(N, P, device=X.device)

    with torch.enable_grad():
        for i in range(N):
            model.zero_grad()

            xi = X[i:i + 1].detach().clone().requires_grad_(True)

            u = model(xi)  

            gu = torch.autograd.grad(
                u, xi,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]  

            d2x = torch.autograd.grad(
                gu[:, 0:1], xi,
                grad_outputs=torch.ones_like(gu[:, 0:1]),
                create_graph=True,
                retain_graph=True,
            )[0][:, 0]  

            d2y = torch.autograd.grad(
                gu[:, 1:2], xi,
                grad_outputs=torch.ones_like(gu[:, 1:2]),
                create_graph=True,
                retain_graph=True,
            )[0][:, 1]  

            neg_lap = -(d2x + d2y)  

            neg_lap.sum().backward()

            row = []
            for p in params:
                if p.grad is not None:
                    row.append(p.grad.detach().clone().flatten())
                    p.grad.zero_()
                else:
                    row.append(torch.zeros(p.numel(), device=X.device))
            J_L[i] = torch.cat(row)

    if was_training:
        model.train()
    return J_L

def compute_bc_jacobian(
        model: nn.Module,
        xy_boundary: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        bc_type: str = "dirichlet",
    ) -> torch.Tensor:
    was_training = model.training
    model.eval()

    N = xy_boundary.shape[0]
    if N == 0:
        params = [p for p in model.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)
        return torch.zeros(0, P, device=xy_boundary.device)

    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    J_bc = torch.zeros(N, P, device=xy_boundary.device)

    with torch.enable_grad():
        for i in range(N):
            model.zero_grad()

            xi = xy_boundary[i:i + 1].detach().clone().requires_grad_(True)

            if bc_type.lower() == "dirichlet":
                u = model(xi).squeeze()
                u.backward(retain_graph=False)

            elif bc_type.lower() == "neumann":
                if normals is None:
                    raise ValueError("Для Неймана требуются нормали (normals)")

                u = model(xi)
                grad_u = torch.autograd.grad(
                    u, xi,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True,
                )[0]

                n_i = normals[i:i + 1]
                normal_derivative = (grad_u * n_i).sum()

                normal_derivative.backward(retain_graph=False)
            else:
                raise ValueError(f"Неизвестный тип граничного условия: {bc_type}")

            row = []
            for p in params:
                if p.grad is not None:
                    row.append(p.grad.detach().clone().flatten())
                    p.grad.zero_()
                else:
                    row.append(torch.zeros(p.numel(), device=xy_boundary.device))
            J_bc[i] = torch.cat(row)

    if was_training:
        model.train()
    return J_bc

def compute_ntk_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    return J @ J.T

@torch.no_grad()
def compute_empirical_ntk(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    J = compute_jacobian(model, X)
    return compute_ntk_from_jacobian(J)

@torch.no_grad()
def compute_pde_ntk(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    J_L = compute_pde_jacobian(model, X)
    return compute_ntk_from_jacobian(J_L)

@torch.no_grad()
def compute_bc_ntk(
        model: nn.Module,
        xy_boundary: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        bc_type: str = "dirichlet",
    ) -> torch.Tensor:
    J_bc = compute_bc_jacobian(model, xy_boundary, normals, bc_type)
    if J_bc.shape[0] == 0:
        return torch.empty(0, 0, device=xy_boundary.device)
    return compute_ntk_from_jacobian(J_bc)

@torch.no_grad()
def compute_full_pde_ntk(
        model: nn.Module,
        X_interior: torch.Tensor,
        X_dirichlet: torch.Tensor,
        X_neumann: torch.Tensor,
        normals_neumann: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
    ) -> Dict[str, Any]:
    device = next(model.parameters()).device

    X_in = X_interior.to(device) if len(X_interior) > 0 else torch.empty(0, 2, device=device)
    X_D = X_dirichlet.to(device) if len(X_dirichlet) > 0 else torch.empty(0, 2, device=device)
    X_N = X_neumann.to(device) if len(X_neumann) > 0 else torch.empty(0, 2, device=device)

    n_in, n_D, n_N = len(X_in), len(X_D), len(X_N)

    J_L = compute_pde_jacobian(model, X_in) if n_in > 0 else torch.empty(0, 0, device=device)
    J_D = compute_bc_jacobian(model, X_D, bc_type="dirichlet") if n_D > 0 else torch.empty(0, 0, device=device)
    J_N = compute_bc_jacobian(model, X_N, normals_neumann, "neumann") if n_N > 0 else torch.empty(0, 0, device=device)

    J_all_list = [J for J in [J_L, J_D, J_N] if J.shape[0] > 0]

    if len(J_all_list) > 0:
        J_all = torch.cat(J_all_list, dim=0)
        K_full = (J_all @ J_all.T).cpu().numpy()
    else:
        K_full = np.zeros((0, 0))

    blocks = {
        "K_LL": (J_L @ J_L.T).cpu().numpy() if n_in > 0 else np.zeros((0, 0)),
        "K_DD": (J_D @ J_D.T).cpu().numpy() if n_D > 0 else np.zeros((0, 0)),
        "K_NN": (J_N @ J_N.T).cpu().numpy() if n_N > 0 else np.zeros((0, 0)),
    }

    cross_terms = {
        "K_LD": (J_L @ J_D.T).cpu().numpy() if n_in > 0 and n_D > 0 else np.zeros((n_in, n_D)),
        "K_DL": (J_D @ J_L.T).cpu().numpy() if n_D > 0 and n_in > 0 else np.zeros((n_D, n_in)),
        "K_LN": (J_L @ J_N.T).cpu().numpy() if n_in > 0 and n_N > 0 else np.zeros((n_in, n_N)),
        "K_NL": (J_N @ J_L.T).cpu().numpy() if n_N > 0 and n_in > 0 else np.zeros((n_N, n_in)),
        "K_DN": (J_D @ J_N.T).cpu().numpy() if n_D > 0 and n_N > 0 else np.zeros((n_D, n_N)),
        "K_ND": (J_N @ J_D.T).cpu().numpy() if n_N > 0 and n_D > 0 else np.zeros((n_N, n_D)),
    }

    block_info = {
        "labels": [],
        "sizes": [],
        "n_interior": n_in,
        "n_dirichlet": n_D,
        "n_neumann": n_N,
        "n_total": n_in + n_D + n_N,
    }

    if n_in > 0:
        block_info["labels"].append("PDE (K_L)")
        block_info["sizes"].append(n_in)
    if n_D > 0:
        block_info["labels"].append("Dirichlet")
        block_info["sizes"].append(n_D)
    if n_N > 0:
        block_info["labels"].append("Neumann")
        block_info["sizes"].append(n_N)

    return {
        "K_full": K_full,
        "J_L": J_L.cpu().numpy() if n_in > 0 else np.zeros((0, 0)),
        "J_D": J_D.cpu().numpy() if n_D > 0 else np.zeros((0, 0)),
        "J_N": J_N.cpu().numpy() if n_N > 0 else np.zeros((0, 0)),
        "blocks": blocks,
        "cross_terms": cross_terms,
        "block_info": block_info,
    }

def split_boundary_points(
        xy_boundary: torch.Tensor,
        bc_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_flat = bc_mask.squeeze(-1) if bc_mask.dim() > 1 else bc_mask

    idx_dirichlet = (mask_flat > 0.5).nonzero(as_tuple=True)[0]
    idx_neumann = (mask_flat <= 0.5).nonzero(as_tuple=True)[0]

    xy_dirichlet = xy_boundary[idx_dirichlet] if len(idx_dirichlet) > 0 else torch.empty(0, xy_boundary.shape[1], device=xy_boundary.device)
    xy_neumann = xy_boundary[idx_neumann] if len(idx_neumann) > 0 else torch.empty(0, xy_boundary.shape[1], device=xy_boundary.device)

    return xy_dirichlet, xy_neumann, idx_dirichlet, idx_neumann

def _compute_condition_number(eigenvalues: np.ndarray) -> float:
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) < 2:
        return float("inf")
    return float(eig_pos[0] / eig_pos[-1])

def _compute_effective_rank(eigenvalues: np.ndarray) -> float:
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) == 0:
        return 0.0
    p = eig_pos / eig_pos.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-30))))

def _compute_spectral_decay_rate(eigenvalues: np.ndarray) -> float:
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) < 3:
        return 0.0

    log_k = np.log(np.arange(1, len(eig_pos) + 1))
    log_eig = np.log(eig_pos + 1e-30)

    n = len(log_k)
    slope = (n * np.sum(log_k * log_eig) - np.sum(log_k) * np.sum(log_eig)) / \
            (n * np.sum(log_k**2) - np.sum(log_k)**2 + 1e-30)

    return float(-slope)

def _compute_mode_utilization(eigenvalues: np.ndarray, threshold: float = 0.9) -> int:
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eig_pos) == 0:
        return 0

    total = eig_pos.sum()
    cumsum = np.cumsum(eig_pos)

    idx = np.searchsorted(cumsum / total, threshold)
    return int(idx + 1)

def _compute_extended_metrics(
    eigenvalues_K: np.ndarray,
    eigenvalues_KL: np.ndarray,
    eigenvalues_KD: Optional[np.ndarray] = None,
    eigenvalues_KN: Optional[np.ndarray] = None,
) -> Dict[str, Any]:

    kappa_K = _compute_condition_number(eigenvalues_K)
    kappa_KL = _compute_condition_number(eigenvalues_KL)
    rank_K = _compute_effective_rank(eigenvalues_K)
    rank_KL = _compute_effective_rank(eigenvalues_KL)

    kappa_ratio = kappa_KL / kappa_K if kappa_K > 0 and kappa_K < float("inf") else float("inf")
    rank_ratio = rank_KL / rank_K if rank_K > 0 else 0.0

    decay_K = _compute_spectral_decay_rate(eigenvalues_K)
    decay_KL = _compute_spectral_decay_rate(eigenvalues_KL)

    mode_50_K = _compute_mode_utilization(eigenvalues_K, 0.5)
    mode_90_K = _compute_mode_utilization(eigenvalues_K, 0.9)
    mode_50_KL = _compute_mode_utilization(eigenvalues_KL, 0.5)
    mode_90_KL = _compute_mode_utilization(eigenvalues_KL, 0.9)

    energy_K = float(eigenvalues_K.sum())
    energy_KL = float(eigenvalues_KL.sum())

    health_score = 100.0

    if kappa_ratio > 100:
        health_score -= 30
    elif kappa_ratio > 10:
        health_score -= 15
    elif kappa_ratio > 5:
        health_score -= 5

    if rank_ratio < 0.5:
        health_score -= 25
    elif rank_ratio < 0.7:
        health_score -= 10

    if decay_KL < 0.5:
        health_score -= 15
    elif decay_KL < 1.0:
        health_score -= 5

    if kappa_ratio < 2 and rank_ratio > 0.8:
        health_score += 10

    health_score = max(0, min(100, health_score))

    energy_KD = float(eigenvalues_KD.sum()) if eigenvalues_KD is not None and len(eigenvalues_KD) > 0 else 0.0
    energy_KN = float(eigenvalues_KN.sum()) if eigenvalues_KN is not None and len(eigenvalues_KN) > 0 else 0.0

    energy_total = energy_K + energy_KD + energy_KN
    energy_balance = {
        "K": energy_K / energy_total if energy_total > 0 else 0.0,
        "KL": energy_KL / energy_total if energy_total > 0 else 0.0,
        "KD": energy_KD / energy_total if energy_total > 0 else 0.0,
        "KN": energy_KN / energy_total if energy_total > 0 else 0.0,
    }

    return {
        "kappa_K": kappa_K,
        "kappa_KL": kappa_KL,
        "rank_K": rank_K,
        "rank_KL": rank_KL,
        "kappa_ratio": kappa_ratio,
        "rank_ratio": rank_ratio,
        "spectral_decay_rate_K": decay_K,
        "spectral_decay_rate_KL": decay_KL,
        "mode_utilization_50_K": mode_50_K,
        "mode_utilization_90_K": mode_90_K,
        "mode_utilization_50_KL": mode_50_KL,
        "mode_utilization_90_KL": mode_90_KL,
        "health_score": health_score,
        "energy_K": energy_K,
        "energy_KL": energy_KL,
        "energy_KD": energy_KD,
        "energy_KN": energy_KN,
        "energy_balance": energy_balance,
    }

@torch.no_grad()
def ntk_comprehensive_analysis(
        model: nn.Module,
        X_interior: torch.Tensor,
        X_boundary: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        bc_mask: Optional[torch.Tensor] = None,
        n_interior: int = 64,
        n_boundary: int = 32,
        compute_full_pde: bool = True,
    ) -> Dict[str, Any]:
    device = next(model.parameters()).device

    def _subsample(X: torch.Tensor, n: int) -> torch.Tensor:
        N = len(X)
        if N <= n:
            return X
        idx = torch.linspace(0, N - 1, n, device=X.device).long()
        return X[idx]

    X_in = _subsample(X_interior.to(device), n_interior)
    N_in = len(X_in)

    print(f"[NTK] Starting comprehensive analysis...")
    print(f"[NTK] Points: interior={N_in}, boundary sample={n_boundary}")

    print(f"[NTK] Computing interior Jacobians (N={N_in})...")
    J_in = compute_jacobian(model, X_in)
    J_L_in = compute_pde_jacobian(model, X_in)

    K_in = compute_ntk_from_jacobian(J_in).cpu().numpy()
    K_L_in = compute_ntk_from_jacobian(J_L_in).cpu().numpy()

    eig_K_in = np.sort(np.linalg.eigvalsh(K_in))[::-1].clip(0)
    eig_KL_in = np.sort(np.linalg.eigvalsh(K_L_in))[::-1].clip(0)

    interior_result = {
        "X": X_in.cpu().numpy(),
        "K": K_in,
        "K_L": K_L_in,
        "J": J_in.cpu().numpy(),
        "J_L": J_L_in.cpu().numpy(),
        "eigenvalues_K": eig_K_in,
        "eigenvalues_KL": eig_KL_in,
        "condition_number_K": _compute_condition_number(eig_K_in),
        "condition_number_KL": _compute_condition_number(eig_KL_in),
        "effective_rank_K": _compute_effective_rank(eig_K_in),
        "effective_rank_KL": _compute_effective_rank(eig_KL_in),
        "trace_K": float(eig_K_in.sum()),
        "trace_KL": float(eig_KL_in.sum()),
    }

    boundary_result = {"dirichlet": None, "neumann": None}
    xy_D, xy_N = None, None

    if X_boundary is not None and len(X_boundary) > 0:
        X_bd = _subsample(X_boundary.to(device), n_boundary)
        N_bd = len(X_bd)

        normals_sub = _subsample(normals.to(device), n_boundary) if normals is not None else None
        bc_mask_sub = _subsample(bc_mask.to(device), n_boundary) if bc_mask is not None else None

        if bc_mask_sub is not None:
            xy_D, xy_N, idx_D, idx_N = split_boundary_points(X_bd, bc_mask_sub)

            normals_D = normals_sub[idx_D] if normals_sub is not None and len(idx_D) > 0 else None
            normals_N = normals_sub[idx_N] if normals_sub is not None and len(idx_N) > 0 else None

            if len(xy_D) > 0:
                print(f"[NTK] Computing Dirichlet BC Jacobian (N={len(xy_D)})...")
                J_D = compute_bc_jacobian(model, xy_D, normals_D, "dirichlet")
                K_D = compute_ntk_from_jacobian(J_D).cpu().numpy()
                eig_D = np.sort(np.linalg.eigvalsh(K_D))[::-1].clip(0)

                boundary_result["dirichlet"] = {
                    "X": xy_D.cpu().numpy(),
                    "K": K_D,
                    "J": J_D.cpu().numpy(),
                    "eigenvalues": eig_D,
                    "condition_number": _compute_condition_number(eig_D),
                    "effective_rank": _compute_effective_rank(eig_D),
                    "trace": float(eig_D.sum()),
                    "n_points": len(xy_D),
                }

            if len(xy_N) > 0:
                print(f"[NTK] Computing Neumann BC Jacobian (N={len(xy_N)})...")
                J_N = compute_bc_jacobian(model, xy_N, normals_N, "neumann")
                K_N = compute_ntk_from_jacobian(J_N).cpu().numpy()
                eig_N = np.sort(np.linalg.eigvalsh(K_N))[::-1].clip(0)

                boundary_result["neumann"] = {
                    "X": xy_N.cpu().numpy(),
                    "K": K_N,
                    "J": J_N.cpu().numpy(),
                    "eigenvalues": eig_N,
                    "condition_number": _compute_condition_number(eig_N),
                    "effective_rank": _compute_effective_rank(eig_N),
                    "trace": float(eig_N.sum()),
                    "n_points": len(xy_N),
                }

    print("[NTK] Computing full block matrices...")

    J_all = J_in
    block_labels = ["interior_K", "interior_KL"]
    block_sizes = [N_in, N_in]

    n_dirichlet = 0
    n_neumann = 0

    if boundary_result["dirichlet"] is not None:
        J_D_tensor = torch.from_numpy(boundary_result["dirichlet"]["J"]).to(device)
        J_all = torch.cat([J_all, J_D_tensor], dim=0)
        block_labels.append("dirichlet")
        n_dirichlet = boundary_result["dirichlet"]["n_points"]
        block_sizes.append(n_dirichlet)

    if boundary_result["neumann"] is not None:
        J_N_tensor = torch.from_numpy(boundary_result["neumann"]["J"]).to(device)
        J_all = torch.cat([J_all, J_N_tensor], dim=0)
        block_labels.append("neumann")
        n_neumann = boundary_result["neumann"]["n_points"]
        block_sizes.append(n_neumann)

    K_full = (J_all @ J_all.T).cpu().numpy()
    K_L_full = K_L_in

    eig_full = np.sort(np.linalg.eigvalsh(K_full))[::-1].clip(0)

    full_result = {
        "K_full": K_full,
        "K_L_full": K_L_full,
        "block_labels": block_labels,
        "block_sizes": block_sizes,
        "eigenvalues": eig_full,
        "condition_number": _compute_condition_number(eig_full),
        "effective_rank": _compute_effective_rank(eig_full),
        "trace": float(eig_full.sum()),
        "n_total": len(J_all),
        "n_interior": N_in,
        "n_dirichlet": n_dirichlet,
        "n_neumann": n_neumann,
    }

    full_pde_result = None
    if compute_full_pde:
        print("[NTK] Computing full PDE NTK with boundary conditions...")
        xy_D_pts = xy_D if xy_D is not None and len(xy_D) > 0 else torch.empty(0, 2, device=device)
        xy_N_pts = xy_N if xy_N is not None and len(xy_N) > 0 else torch.empty(0, 2, device=device)

        normals_N_pts = None
        if xy_N is not None and len(xy_N) > 0 and normals is not None:
            normals_N_pts = normals_sub[idx_N] if bc_mask_sub is not None else None

        full_pde_result = compute_full_pde_ntk(
            model=model,
            X_interior=X_in,
            X_dirichlet=xy_D_pts,
            X_neumann=xy_N_pts,
            normals_neumann=normals_N_pts,
        )

    eig_D = boundary_result["dirichlet"]["eigenvalues"] if boundary_result["dirichlet"] else None
    eig_N = boundary_result["neumann"]["eigenvalues"] if boundary_result["neumann"] else None

    extended_metrics = _compute_extended_metrics(eig_K_in, eig_KL_in, eig_D, eig_N)

    spectrum_result = {
        "interior_K": {
            "eigenvalues": eig_K_in,
            "condition_number": interior_result["condition_number_K"],
            "effective_rank": interior_result["effective_rank_K"],
        },
        "interior_KL": {
            "eigenvalues": eig_KL_in,
            "condition_number": interior_result["condition_number_KL"],
            "effective_rank": interior_result["effective_rank_KL"],
        },
        "full": {
            "eigenvalues": eig_full,
            "condition_number": full_result["condition_number"],
            "effective_rank": full_result["effective_rank"],
        },
    }

    if boundary_result["dirichlet"] is not None:
        spectrum_result["dirichlet"] = {
            "eigenvalues": boundary_result["dirichlet"]["eigenvalues"],
            "condition_number": boundary_result["dirichlet"]["condition_number"],
            "effective_rank": boundary_result["dirichlet"]["effective_rank"],
        }

    if boundary_result["neumann"] is not None:
        spectrum_result["neumann"] = {
            "eigenvalues": boundary_result["neumann"]["eigenvalues"],
            "condition_number": boundary_result["neumann"]["condition_number"],
            "effective_rank": boundary_result["neumann"]["effective_rank"],
        }

    n_top = min(32, N_in)
    convergence_rates = {
        "interior_K": 1.0 - np.exp(-eig_K_in[:n_top]),
        "interior_KL": 1.0 - np.exp(-eig_KL_in[:n_top]),
    }
    spectrum_result["convergence_rates"] = convergence_rates

    print(f"[NTK] Analysis complete. Health score: {extended_metrics['health_score']:.1f}/100")

    return {
        "interior": interior_result,
        "boundary": boundary_result,
        "full": full_result,
        "full_pde": full_pde_result,
        "extended_metrics": extended_metrics,
        "spectrum": spectrum_result,
    }
