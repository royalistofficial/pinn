import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev, hessian
from typing import Optional

def _get_functional_params(model: nn.Module):
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = dict(model.named_buffers())
    return params, buffers

def compute_jacobian(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()

    params, buffers = _get_functional_params(model)

    def fmodel(p, b, x):
        return functional_call(model, (p, b), (x.unsqueeze(0),)).squeeze()

    def compute_single_jacobian(p, b, x):
        def f(weights):
            return fmodel(weights, b, x)
        return jacrev(f)(p)

    jac_dict = vmap(compute_single_jacobian, in_dims=(None, None, 0))(params, buffers, X)

    J = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1).detach()

    if was_training:
        model.train()
    return J

def compute_pde_jacobian(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()

    params, buffers = _get_functional_params(model)

    def fmodel(p, b, x):
        return functional_call(model, (p, b), (x.unsqueeze(0),)).squeeze()

    def compute_single_pde_jacobian(p, b, x):

        def pde_res(weights):
            def u_fn(x_in):
                return fmodel(weights, b, x_in)

            H = hessian(u_fn)(x)

            laplacian = H[0, 0] + H[1, 1] 
            return -laplacian 

        return jacrev(pde_res)(p)

    jac_dict = vmap(compute_single_pde_jacobian, in_dims=(None, None, 0))(params, buffers, X)
    J_L = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1).detach()

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

    params, buffers = _get_functional_params(model)

    if xy_boundary.shape[0] == 0:
        P = sum(p.numel() for p in params.values())
        return torch.zeros(0, P, device=xy_boundary.device)

    def fmodel(p, b, x):
        return functional_call(model, (p, b), (x.unsqueeze(0),)).squeeze()

    def compute_single_dirichlet(p, b, x):
        def f(weights):
            return fmodel(weights, b, x)
        return jacrev(f)(p)

    def compute_single_neumann(p, b, x, n):
        def neu_res(weights):
            def u_fn(x_in):
                return fmodel(weights, b, x_in)
            grad_x = jacrev(u_fn)(x)
            return torch.dot(grad_x, n) 
        return jacrev(neu_res)(p)

    if bc_type.lower() == "dirichlet":
        jac_dict = vmap(compute_single_dirichlet, in_dims=(None, None, 0))(params, buffers, xy_boundary)
    elif bc_type.lower() == "neumann":
        jac_dict = vmap(compute_single_neumann, in_dims=(None, None, 0, 0))(params, buffers, xy_boundary, normals)

    J_bc = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1).detach()

    if was_training:
        model.train()
    return J_bc

def compute_ntk_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    K = J @ J.T
    K = (K + K.T) / 2.0
    return K