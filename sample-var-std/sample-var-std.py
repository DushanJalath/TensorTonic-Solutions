import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x=np.array(x)
    sum = np.sum(x)
    n=len(x)
    mean = sum/n

    variance = np.sum((x-mean)**2)/(n-1)

    return float(variance),float(np.sqrt(variance))