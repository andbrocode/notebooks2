def __get_fband_average(freq, psd, faction_of_octave=1, average="mean", plot=False):

    """
    Computing averages for frequency octave bands. 

    Arguments:
        - freq:    (array) frequency values
        - psd:    (array) psd values
        - fraction_of_octave:    (int) octave fraction (e.g. [1] = octaves, 3 = third octaves, 12 = 12th octaves)
        - plot:    (bool) show psd and psd-average

    Return:
        - out:    (dict) output dictionary

    Example:

    >>> out = __get_fband_average(freq, psd, faction_of_octave=1, average="mean", plot=False)



    """

    import matplotlib.pyplot as plt
    from numpy import nanmean, nanmedian, array

    def __get_octave_bands(fmin, fmax, faction_of_octave=1, plot=False):

        """
        Computing octave bands

        Arguments:
            - fmin:    (float) minimum center frequency
            - fmax:    (float) maximum center frequency
            - fraction_of_octave:    (int) octave fraction (e.g. [1] = octaves, 3 = third octaves, 12 = 12th octaves)
            - plot:    (bool) show frequency bands

        Example:

        >>> flower, fupper, fcenter = __get_octave_bands(f_min, f_max, fband_type="octave", plot=False)

        """

        import matplotlib.pyplot as plt
        from acoustics.octave import Octave
        from numpy import array

        ## avoid fmin = zero
        if fmin == 0:
            # print(f" -> set fmin to 1e-10 instead of 0")
            fmin = 1e-10

        f_lower, f_upper, f_centers = [], [], []

        _octaves = Octave(fraction=faction_of_octave, interval=None, fmin=fmin, fmax=fmax, unique=False, reference=1000.0)

        f_centers = _octaves.center
        f_lower = _octaves.lower
        f_upper = _octaves.upper

        if plot:
            plt.figure(figsize=(15, 5))
            for fl, fc, fu in zip(f_lower, f_centers, f_upper):
                plt.axvline(fu, color="r")
                plt.axvline(fl, color="r")
                plt.axvline(fc, ls="--")
                plt.axvline(fmin, color="g")
                plt.axvline(fmax, color="g")
                plt.xscale("log")
            plt.show();

        return array(f_lower), array(f_upper), array(f_centers)


    ## get octave bands
    f_center, f_upper, f_lower = __get_octave_bands(freq[0], freq[-1], faction_of_octave=faction_of_octave, plot=False)

    ## get frequency indices
    fl_idx, fu_idx = [], []

    for _k, (fl, fu) in enumerate(zip(f_lower, f_upper)):
        if _k <= len(f_center):

            for _i, _f in enumerate(freq):
                if _f >= fl:
                    fl_idx.append(int(_i))
                    break

            for _i, _f in enumerate(freq):
                if _f >= fu:
                    fu_idx.append(int(_i))
                    break

    ## compute mean per band
    psd_average, fc, fu, fl = [], [], [], []
    for _n, (ifl, ifu) in enumerate(zip(fl_idx, fu_idx)):
        if ifl != ifu:
            if average == "mean":
                psd_average.append(nanmean(psd[ifl:ifu]))
            elif average == "median":
                psd_average.append(nanmedian(psd[ifl:ifu]))

            fc.append(f_center[_n])
            fu.append(f_upper[_n])
            fl.append(f_lower[_n])

    psd_average = array(psd_average)

    ## check up plot
    if plot:
        fig = plt.figure(figsize=(15, 5))

        plt.plot(freq, psd, label="raw psd")
        plt.plot(fc, psd_average, label=average)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("PSD")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        plt.show();


    ## output
    out = {}
    out['psd_means'] = array(psd_average)
    out['fcenter'] = array(fc)
    out['fupper'] = array(fu)
    out['flower'] = array(fl)

    if plot:
        out['fig'] = fig

    return out