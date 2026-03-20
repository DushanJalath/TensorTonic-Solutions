import numpy as np

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    matrix=np.array(matrix,dtype=float)
    result = np.zeros_like(matrix)

    for i, row in enumerate(matrix):
        rated_mask = row != 0                      # Boolean mask of rated items
        if rated_mask.any():
            user_mean = row[rated_mask].mean()     # Mean of rated items only
            result[i, rated_mask] = row[rated_mask] - user_mean 

    return result.tolist()