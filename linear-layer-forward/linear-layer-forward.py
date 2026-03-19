import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    W=np.array(W, dtype=float)
    X=np.array(X, dtype=float)
    b=np.array(b, dtype=float)
    
    return (np.dot(X,W)+b).tolist()