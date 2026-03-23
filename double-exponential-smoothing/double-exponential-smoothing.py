def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    level = series[0]
    trend = series[1]-series[0]
    result = [level]

    for i in range(1,len(series)):
        new_level = alpha*series[i]+(1-alpha)*(level+trend)
        new_trend = beta*(new_level-level)+(1-beta)*trend
        level,trend = new_level,new_trend

        result.append(new_level)

    return result