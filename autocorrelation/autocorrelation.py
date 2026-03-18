import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    series = np.array(series)
    x_bar = np.mean(series)
    var = sum((series-x_bar)**2)
    results = []

    if var == 0:                              
        return [1.0] + [0.0] * max_lag    
    for lag in range(0,max_lag+1):
        numerator = sum(
            (series[t] - x_bar) * (series[t + lag] - x_bar)
            for t in range(0, len(series) - lag)         
        )
        results.append(numerator / var)  

    return results