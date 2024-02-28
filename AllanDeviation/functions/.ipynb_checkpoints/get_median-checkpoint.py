def __get_median(psds):

    from numpy import median, zeros, isnan

    med_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:, f]
        med_psd[f] = median(a[~isnan(a)])

    return med_psd