def find_minimum(arr1, arr2, lagmin=-100, lagmax=100, scalemin=0.1, scalemax=2, dlag=1, dscale=0.1plot=True):

    import matplotlib.pyplot as plt
    import numpy as np
    from functions.variance_reduction import __variance_reduction

    def get_variance_reduction(arr1, arr2):

        from numpy import var

        sig1 = var(arr1)
        sig2 = var(arr2)

        return round( (sig1 - sig2) / sig1 * 100, 2)

    # preapre axis for lags
    dshft = dlag
    shifts = np.arange(lagmin, lagmax+dshft, dshft)

    # preapre axis for scaling
    dfac = dscale
    factors = np.arange(scalemin, scalemax+dfac, dfac)

    Ns, Nf = len(shifts), len(factors)
    vr = np.zeros((Ns, Nf))

    for i, s in enumerate(shifts):
        for j, f in enumerate(factors):

            _arr1 = f*np.roll(arr1, s)

            vr[i, j] = get_variance_reduction(arr2, arr2 - _arr1)

    imax, jmax = np.unravel_index(vr.argmax(), vr.shape)

    vr_max, s_max, f_max = round(vr[imax, jmax], 1), round(shifts[imax], 0), round(factors[jmax], 2)

    # create checkup plot, if required
    if plot:

        fig = plt.figure()

        cm = plt.pcolormesh(shifts,
                            factors,
                            vr[:-1, :-1].T,
                            rasterized=True,
                            cmap=plt.colormaps.get("seismic"),
                            vmin=-100, vmax=100,
                           )

        plt.scatter(s_max, f_max, color="black", s=15, edgecolor="w")

        plt.text(0, 0.99, f"{s_max}, {f_max}, {vr_max}%", ha='center', va='top', color="w")

        plt.xlabel("Shifts (samples)")
        plt.ylabel("Scaling Factor")

        cb = plt.colorbar(cm)
        cb.set_label("Variance Reduction (%)")

    return s_max, f_max, vr_max