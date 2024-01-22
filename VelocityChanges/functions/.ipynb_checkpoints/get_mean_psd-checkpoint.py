def __get_mean_psd(psds):

    from numpy import mean, zeros, isnan

    mean_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:, f]
        mean_psd[f] = mean(a[~isnan(a)])

    return mean_psd