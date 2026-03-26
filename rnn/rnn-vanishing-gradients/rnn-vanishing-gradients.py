import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh,ord=2)

    norms = []
    current_norm = 1

    for t in range(T):
        norms.append(current_norm)
        current_norm*=spectral_norm

    return norms