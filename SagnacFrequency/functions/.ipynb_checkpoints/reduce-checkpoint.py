def __reduce(dat, n_samples):
    from numpy import nanmean
    return dat - nanmean(dat[:n_samples])