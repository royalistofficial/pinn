import numpy as np
from typing import Dict, Any

def compute_condition_number(eigenvalues: np.ndarray) -> float:
    return float(eigenvalues[0] / eigenvalues[-1])

def compute_effective_rank(eigenvalues: np.ndarray) -> float:

    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))

def compute_trace(K: np.ndarray) -> float:
    return float(np.trace(K))

def compute_frobenius_norm(K: np.ndarray) -> float:
    return float(np.linalg.norm(K, 'fro'))

def compute_spectral_decay(eigenvalues: np.ndarray) -> Dict[str, float]:
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_eig = np.log(eigenvalues)

    power_fit = np.polyfit(log_k, log_eig, 1)
    alpha = -power_fit[0]

    exp_fit = np.polyfit(k, log_eig, 1)
    beta = -exp_fit[0]

    return {"power_law_alpha": float(alpha), "exp_decay_beta": float(beta)}

def get_all_metrics(K: np.ndarray, eigenvalues: np.ndarray) -> Dict[str, Any]:
    return {
        "condition_number": compute_condition_number(eigenvalues),
        "effective_rank": compute_effective_rank(eigenvalues),
        "trace": compute_trace(K),
        "frobenius_norm": compute_frobenius_norm(K),
        "decay": compute_spectral_decay(eigenvalues)
    }