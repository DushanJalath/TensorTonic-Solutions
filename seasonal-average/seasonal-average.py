def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    result = []
    for p in range(period):
        seasonal_values = series[p::period]
        result.append(sum(seasonal_values)/len(seasonal_values))

    return result