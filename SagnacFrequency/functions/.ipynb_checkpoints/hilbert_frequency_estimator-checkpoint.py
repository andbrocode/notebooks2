def __hilbert_frequency_estimator(st, nominal_sagnac, fband):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()

    ## extract sampling rate
    df = st0[0].stats.sampling_rate

    ## define frequency band around Sagnac Frequency
    f_lower = nominal_sagnac - fband
    f_upper = nominal_sagnac + fband

    ## bandpass with butterworth around Sagnac Frequency
    st0.detrend("linear")
    st0.taper(0.01)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)


    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * df)

    ## cut first and last 5% (corrupted data)
    dd = int(0.05*len(instantaneous_frequency))
    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## get times
    t = st0[0].times()
    t_center = t[int((len(t))/2)]

    ## averaging of frequencies
    insta_f_cut_mean = np.mean(insta_f_cut)
    # insta_f_cut_mean = np.median(insta_f_cut)

    return t_center, insta_f_cut_mean