import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from typing import List

@torch.no_grad()
def compute_empirical_ntk(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    N = X.shape[0]
    device = X.device
    K = torch.zeros(N, N, device=device)
    
    for i in range(N):
        xi = X[i:i+1].requires_grad_(True)
        pred_i = model(xi).squeeze()
        jac_i = torch.autograd.functional.jacobian(
            lambda x: model(x).squeeze(), xi
        ).view(-1)
        
        for j in range(i, N):
            xj = X[j:j+1].requires_grad_(True)
            pred_j = model(xj).squeeze()
            jac_j = torch.autograd.functional.jacobian(
                lambda x: model(x).squeeze(), xj
            ).view(-1)
            
            K[i, j] = K[j, i] = torch.dot(jac_i, jac_j)
    return K


def plot_ntk_and_freq(model: nn.Module, epoch: int, output_dir: str = "data/ntk_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    n_pts = 64
    xy = torch.rand(n_pts, 2, device=device) * 2 - 1
    K = compute_empirical_ntk(model, xy)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im = axs[0].imshow(K.cpu().numpy(), cmap='viridis')
    axs[0].set_title(f'Empirical NTK matrix (epoch {epoch})')
    plt.colorbar(im, ax=axs[0])
    
    eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy()
    axs[1].plot(sorted(eigenvalues, reverse=True), 'o-', markersize=3)
    axs[1].set_title(f'NTK eigenvalues spectrum (epoch {epoch})')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Eigenvalue index')
    axs[1].set_ylabel('λ (log scale)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ntk_epoch_{epoch:04d}.png', dpi=200)
    plt.close(fig)
    
    freqs = []
    for branch in model.branches:        
        Bx = branch.w_x.exp().detach().cpu().numpy()
        By = branch.w_y.exp().detach().cpu().numpy()
        freqs.extend(Bx)
        freqs.extend(By)
    freqs = torch.tensor(freqs)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(freqs.numpy(), bins=50, alpha=0.8, color='steelblue', edgecolor='black')
    ax.set_title(f'Frequency distribution |B| (epoch {epoch})')
    ax.set_xlabel('Frequency magnitude |B|')
    ax.set_ylabel('Count')
    plt.savefig(f'{output_dir}/freq_dist_epoch_{epoch:04d}.png', dpi=200)
    plt.close(fig)
    
    print(f"[NTK] Saved plots for epoch {epoch}")