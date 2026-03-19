import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x=np.array(x,dtype=float)
    p=np.array(p,dtype=float)
    if not np.allclose(np.sum(p), 1.0):      # ← np.allclose instead of np.isclose
        raise ValueError("ValueError")
    return float(np.sum(x*p))
