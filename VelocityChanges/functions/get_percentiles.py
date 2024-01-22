def __get_percentiles(arr, p_low=2.5, p_high=97.5):

    from numpy import zeros, nanpercentile, shape

    percentiles_lower = zeros(shape(arr)[1])
    percentiles_upper = zeros(shape(arr)[1])

    for kk in range(shape(arr)[1]):
        out = nanpercentile(arr[:, kk],  [p_low, p_high])
        percentiles_upper[kk] = out[1]
        percentiles_lower[kk] = out[0]

    return percentiles_lower, percentiles_upper