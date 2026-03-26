from __future__ import annotations
from typing import Optional
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
            xi = X[i:i + 1].clone()
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

def compute_ntk_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    return J @ J.T

@torch.no_grad()
def compute_empirical_ntk(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    J = compute_jacobian(model, X)
    return compute_ntk_from_jacobian(J)

@torch.no_grad()
def ntk_predict(model, X_train, y_train, X_test, lr, epoch, reg=1e-6):
    X_all = torch.cat([X_train, X_test], dim=0)
    J_all = compute_jacobian(model, X_all)
    N_tr = X_train.shape[0]
    J_train, J_test = J_all[:N_tr], J_all[N_tr:]

    K_train = J_train @ J_train.T
    K_test_train = J_test @ J_train.T
    K_reg = K_train + reg * torch.eye(N_tr, device=K_train.device)

    eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)
    eigenvalues = eigenvalues.clamp(min=1e-10)

    exp_diag = torch.exp(-lr * eigenvalues * epoch)
    I_minus_exp = eigenvectors @ torch.diag(1.0 - exp_diag) @ eigenvectors.T
    K_inv = eigenvectors @ torch.diag(1.0 / eigenvalues) @ eigenvectors.T

    y_flat = y_train.squeeze(-1) if y_train.dim() > 1 else y_train
    y_pred = K_test_train @ K_inv @ I_minus_exp @ y_flat
    return y_pred.unsqueeze(-1)

@torch.no_grad()
def ntk_train_dynamics(model, X_train, y_train, lr, epoch, reg=1e-6):
    J = compute_jacobian(model, X_train)
    K = J @ J.T + reg * torch.eye(X_train.shape[0], device=X_train.device)

    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.clamp(min=1e-10)

    exp_diag = torch.exp(-lr * eigenvalues * epoch)
    I_minus_exp = eigenvectors @ torch.diag(1.0 - exp_diag) @ eigenvectors.T

    y_flat = y_train.squeeze(-1)
    return (I_minus_exp @ y_flat).unsqueeze(-1)

@torch.no_grad()
def ntk_spectrum_analysis(model: nn.Module, X: torch.Tensor) -> dict:
    K = compute_empirical_ntk(model, X)
    eigenvalues = torch.linalg.eigvalsh(K).clamp(min=0)

    eig_sorted = eigenvalues.sort(descending=True).values
    eig_pos = eig_sorted[eig_sorted > 1e-10]

    cond_number = (eig_pos[0] / eig_pos[-1]).item() if len(eig_pos) > 1 else float('inf')

    p = eig_pos / eig_pos.sum()
    eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-30))).item()

    rates = 1.0 - torch.exp(-eig_sorted[:min(10, len(eig_sorted))])

    return {
        "eigenvalues": eig_sorted.cpu().numpy(),
        "condition_number": cond_number,
        "effective_rank": eff_rank,
        "trace": eigenvalues.sum().item(),
        "top_convergence_rates": rates.cpu().numpy(),
        "ntk_matrix": K.cpu().numpy(),
    }

def ntk_preconditioned_step(
        model: nn.Module,
        X: torch.Tensor,
        residual: torch.Tensor,
        lr: float,
        reg: float = 1e-4,
    ) -> float:
    J = compute_jacobian(model, X)
    N = X.shape[0]

    K = J @ J.T + reg * torch.eye(N, device=X.device)

    r = residual.squeeze(-1) if residual.dim() > 1 else residual

    alpha = torch.linalg.solve(K, r)

    delta_theta = lr * (J.T @ alpha)

    update_norm = 0.0
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                update = delta_theta[idx:idx + n].reshape(p.shape)
                p.data -= update
                update_norm += update.norm().item() ** 2
                idx += n

    return math.sqrt(update_norm)

def ntk_preconditioned_step_batched(
        model: nn.Module,
        X_full: torch.Tensor,
        residual_full: torch.Tensor,
        lr: float,
        n_sample: int = 200,
        reg: float = 1e-4,
    ) -> float:
    N = X_full.shape[0]
    if N <= n_sample:
        return ntk_preconditioned_step(model, X_full, residual_full, lr, reg)

    indices = torch.randperm(N, device=X_full.device)[:n_sample]
    X_sub = X_full[indices]
    r_sub = residual_full[indices]

    return ntk_preconditioned_step(model, X_sub, r_sub, lr, reg)

def extract_learned_frequencies(model: nn.Module) -> np.ndarray:
    freqs = []
    if hasattr(model, 'branches'):
        for branch in model.branches:
            if hasattr(branch, 'w_x'):
                Bx = branch.w_x.exp().detach().cpu().numpy()
                By = branch.w_y.exp().detach().cpu().numpy()
                freqs.extend(Bx.tolist())
                freqs.extend(By.tolist())
    return np.array(freqs) if freqs else np.array([])
