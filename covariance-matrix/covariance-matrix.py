import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    
    X=np.array(X)
    if len(X)<2 or X.ndim<2:
        return None
        
    miu=np.mean(X,axis=0)
    X_cen = X-miu

    cov_var = np.dot(X_cen.T,X_cen)/(len(X)-1)

    return cov_var