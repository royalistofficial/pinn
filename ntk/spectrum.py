import numpy as np

def compute_spectrum(K: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(K)

    eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[eigenvalues >= threshold]

    return eigenvalues