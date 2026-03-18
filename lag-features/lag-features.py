import numpy as np

def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    series = np.array(series)
    max_lag = max(lags)

    return [
        [series[t-lag] for lag in lags]
        for t in range(max_lag,len(series))
    ]