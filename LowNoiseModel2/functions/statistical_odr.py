def statistical_odr(_df, Ndraws=10, percentile=90):

    import numpy as np
    from functions.compute_odr import __compute_orthogonal_distance_regression

    slopes = np.ones(Ndraws) * np.nan
    inters = np.ones(Ndraws) * np.nan

    Nsize = _df.shape[0]

    Nsmpls = int(0.8*Nsize)

    for nn in range(Ndraws):
        try:
            # select random sample subset
            tmp = _df.sample(Nsmpls)

            # perform odr
            slopes[nn], inters[nn] = __compute_orthogonal_distance_regression(tmp['x'], tmp['y'], xerr=None, yerr=None, bx=None, by=None)

        except:
            pass

    # get percentile limits
    plower = (100 - percentile) // 2
    pupper = 100 - plower

    out = {}
    out['slope_median'] = np.nanmedian(slopes)
    out['slope_pupper'] = np.nanpercentile(slopes, pupper)
    out['slope_plower'] = np.nanpercentile(slopes, plower)
    out['inter_median'] = np.nanmedian(inters)
    out['inter_pupper'] = np.nanpercentile(inters, pupper)
    out['inter_plower'] = np.nanpercentile(inters, plower)

    return out