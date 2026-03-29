import numpy as np

def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    R = np.array(ratings_matrix,dtype=float)

    if mode == 'user':
        for i in range(R.shape[0]):
            row = R[i]
            nonzero_vals = row[row!=0]
            if len(nonzero_vals)>0:
                user_mean = nonzero_vals.mean()
                row[row==0]=user_mean
    elif mode == 'item':
        for j in range(R.shape[1]):
            col = R[:,j]
            nonzero_vals = col[col!=0]
            if len(nonzero_vals)>0:
                item_mean = nonzero_vals.mean()
                col[col==0]=item_mean
    return R.tolist()