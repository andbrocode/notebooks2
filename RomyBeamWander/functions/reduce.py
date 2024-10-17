def __reduce(dat, n_samples):
    from numpy import nanmean, isnan
    _dat = dat[~isnan(dat)]
    return dat - nanmean(_dat[:n_samples])