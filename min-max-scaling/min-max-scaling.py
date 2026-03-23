import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    data = np.array(data)

    col_min = np.min(data,axis = 0)
    col_max = np.max(data,axis = 0)

    col_range = col_max - col_min
    col_range[col_range == 0] = 1      # constant column → range=1, so (x-min)/1 = 0
    return ((data - col_min) / col_range).tolist()