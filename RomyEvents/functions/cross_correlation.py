def __cross_correlation(arr1, arr2, dt, zerolag=True, demean=True):

    from obspy.signal.cross_correlation import correlate, xcorr_max

    if zerolag:
        ccf = correlate(arr1, arr2, 0, demean=demean, normalize='naive', method='fft')
    else:
        ccf = correlate(arr1, arr2, len(arr1), demean=demean, normalize='naive', method='fft')

    shift, cc_max = xcorr_max(ccf)

    tshift = shift * dt

    return round(cc_max, 2), tshift