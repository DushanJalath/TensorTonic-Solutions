import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    if v_norm < 1e-10 or w_norm < 1e-10:
        return np.nan

    cos_angle = np.dot(v, w) / (v_norm * w_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.arccos(cos_angle))