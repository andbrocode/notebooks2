def __get_mean_psd(psds):

    from numpy import mean, zeros, isnan

    med_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:, f]
        med_psd[f] = mean(a[~isnan(a)])

    return med_psd