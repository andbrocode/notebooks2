def __get_fband_average(freq, psd, faction_of_octave=1, fmin=0.001, fmax=10, average="mean"):

    import matplotlib.pyplot as plt
    from numpy import nanmean, nanmedian, array
    from acoustics.octave import Octave
    from numpy import array


    ## avoid fmin = zero
    if fmin == 0:
        # print(f" -> set fmin to 1e-10 instead of 0")
        fmin = 1e-10

    _octaves = Octave(fraction=faction_of_octave, interval=None, fmin=fmin, fmax=fmax, unique=False, reference=1000.0)

    f_center = _octaves.center
    f_lower = _octaves.lower
    f_upper = _octaves.upper


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

    ## output
    out = {}
    out['psd_means'] = array(psd_average)
    out['fcenter'] = array(fc)
    out['fupper'] = array(fu)
    out['flower'] = array(fl)

    return out