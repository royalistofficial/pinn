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

    cond_number = (eig_pos[0] / eig_pos[-1]).item() if len(eig_pos) > 1 else float("inf")
    p = eig_pos / eig_pos.sum()
    eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-30))).item()
    rates = 1.0 - torch.exp(-eig_sorted[:min(20, len(eig_sorted))])

    return {
        "eigenvalues": eig_sorted.cpu().numpy(),
        "condition_number": cond_number,
        "effective_rank": eff_rank,
        "trace": eigenvalues.sum().item(),
        "top_convergence_rates": rates.cpu().numpy(),
        "ntk_matrix": K.cpu().numpy(),
    }

@torch.no_grad()
def ntk_pde_spectrum_analysis(model: nn.Module, X: torch.Tensor) -> dict:
    K_L = compute_pde_ntk(model, X)
    eigenvalues = torch.linalg.eigvalsh(K_L).clamp(min=0)

    eig_sorted = eigenvalues.sort(descending=True).values
    eig_pos = eig_sorted[eig_sorted > 1e-10]

    cond_number = (eig_pos[0] / eig_pos[-1]).item() if len(eig_pos) > 1 else float("inf")
    p = eig_pos / eig_pos.sum()
    eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-30))).item()
    rates = 1.0 - torch.exp(-eig_sorted[:min(20, len(eig_sorted))])

    return {
        "eigenvalues": eig_sorted.cpu().numpy(),
        "condition_number": cond_number,
        "effective_rank": eff_rank,
        "trace": eigenvalues.sum().item(),
        "top_convergence_rates": rates.cpu().numpy(),
        "ntk_matrix": K_L.cpu().numpy(),
    }

def extract_learned_frequencies(model: nn.Module) -> np.ndarray:
    freqs = []
    net = model.model if hasattr(model, "model") else model
    if hasattr(net, "branches"):
        for branch in net.branches:
            if hasattr(branch, "w_x"):
                Bx = branch.w_x.exp().detach().cpu().numpy()
                By = branch.w_y.exp().detach().cpu().numpy()
                freqs.extend(Bx.tolist())
                freqs.extend(By.tolist())
    return np.array(freqs) if freqs else np.array([])