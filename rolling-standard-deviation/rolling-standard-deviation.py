import numpy as np

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    values = np.array(values)
    result = []

    for i in range(len(values)+1-window_size):
        window = values[i:i+window_size]
        mean = np.mean(window)
        variance = np.sum((window - mean) ** 2) / window_size
        result.append(np.sqrt(variance))
    return result