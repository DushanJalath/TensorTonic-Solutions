import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    if norm_type not in ('l2', 'l1', 'max'):
        return None

    if axis is not None and axis not in (0, 1):
        return None
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    
    ord_map = {'l2': 2, 'l1': 1, 'max': np.inf}
    ord_val = ord_map[norm_type]

    if axis is None:
        norm = np.linalg.norm(matrix.flatten(), ord=ord_val)
        norm = norm if norm != 0 else 1
        return matrix / norm
    else:
        norm = np.linalg.norm(matrix, ord=ord_val, axis=axis, keepdims=True)
        norm[norm == 0] = 1      
        return matrix / norm