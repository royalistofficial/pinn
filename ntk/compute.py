import torch
import torch.nn as nn
from typing import Optional

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
            gu = torch.autograd.grad(u, xi, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            d2x = torch.autograd.grad(gu[:, 0:1], xi, grad_outputs=torch.ones_like(gu[:, 0:1]), create_graph=True)[0][:, 0]
            d2y = torch.autograd.grad(gu[:, 1:2], xi, grad_outputs=torch.ones_like(gu[:, 1:2]), create_graph=True)[0][:, 1]
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
        bc_type: str = "dirichlet"
    ) -> torch.Tensor:
    was_training = model.training
    model.eval()
    N = xy_boundary.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    if N == 0:
        return torch.zeros(0, P, device=xy_boundary.device)

    J_bc = torch.zeros(N, P, device=xy_boundary.device)

    with torch.enable_grad():
        for i in range(N):
            model.zero_grad()
            xi = xy_boundary[i:i + 1].detach().clone().requires_grad_(True)
            if bc_type.lower() == "dirichlet":
                u = model(xi).squeeze()
                u.backward(retain_graph=False)
            elif bc_type.lower() == "neumann":
                u = model(xi)
                grad_u = torch.autograd.grad(u, xi, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                n_i = normals[i:i + 1]
                normal_derivative = (grad_u * n_i).sum()
                normal_derivative.backward(retain_graph=False)

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
    K = J @ J.T

    K = (K + K.T) / 2.0
    return K