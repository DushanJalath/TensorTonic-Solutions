import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape


    X_centered = X - np.mean(X, axis=0)

    cov_matrix = (X_centered.T @ X_centered) / N

    std_devs = np.std(X, axis=0)

    denominator = np.outer(std_devs, std_devs)

    corr_matrix = np.where(denominator == 0, np.nan, cov_matrix / denominator)

    np.fill_diagonal(corr_matrix, np.where(std_devs != 0, 1.0, np.nan))

    return corr_matrix