def __get_minimal_psd(psds):

    from numpy import nanmin, array, nonzero, zeros

    min_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        min_psd[f] = nanmin(a[nonzero(a)])

    return min_psd