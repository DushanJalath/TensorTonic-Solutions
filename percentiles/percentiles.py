import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """

    x = np.sort(np.array(x,dtype=float))
    n=len(x)
    result = []

    for percentile in q:
        idx = percentile/100*(n-1)

        lower=int(np.floor(idx))
        upper=int(np.ceil(idx))
        frac=idx-lower

        value = x[lower]+frac*(x[upper]-x[lower])
        result.append(value)

    return np.array(result)