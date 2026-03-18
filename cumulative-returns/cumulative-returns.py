import numpy as np

def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    returns=np.array(returns)
    W=(np.cumprod(returns+1)-1).tolist()
    return W
    