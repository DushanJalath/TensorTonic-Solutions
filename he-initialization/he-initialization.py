import numpy as np

def he_initialization(W, fan_in):
    """
    Scale raw weights W (uniform [0,1]) to He uniform initialization.
    """
    L = np.sqrt(6 / fan_in)
    return np.array(W) * 2 * L - L  # maps [0,1] → [-L, L]