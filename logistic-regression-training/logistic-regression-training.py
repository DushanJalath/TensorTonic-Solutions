import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    N,D = X.shape
    
    W = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X@W+b)
        error = p - y

        b -=lr*np.mean(error)

        W -= lr*(X.T@error)/N

    return (W,b)


        