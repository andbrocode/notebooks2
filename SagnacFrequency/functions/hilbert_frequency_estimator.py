def __hilbert_frequency_estimator(st, nominal_sagnac, fband=10, cut=0):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()

    # extract sampling rate
    df = st0[0].stats.sampling_rate

    # define frequency band around Sagnac Frequency
    f_lower = nominal_sagnac - fband
    f_upper = nominal_sagnac + fband

    # bandpass with butterworth around Sagnac Frequency
    st0 = st0.detrend("linear")
    st0 = st0.taper(0.01)
    st0 = st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

    # estimate instantaneous frequency with hilbert
    signal = st0[0].data

    # compute analystic signal
    analytic_signal = hilbert(signal)

    # compute envelope
    amplitude_envelope = np.abs(analytic_signal)

    # estimate phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # estimate frequeny
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * df)

    # cut first and last 5% (corrupted data)
    # dd = int(0.05*len(instantaneous_frequency))
    dd = int(cut*df)
    insta_f_cut = instantaneous_frequency[dd:-dd]

    # get times
    t = st0[0].times()
    t_mid = t[int((len(t))/2)]

    # averaging of frequencies
    # insta_f_cut_avg = np.mean(insta_f_cut)
    insta_f_cut_avg = np.median(insta_f_cut)

    # standard error
    insta_f_cut_std = np.std(insta_f_cut)

    return t_mid, insta_f_cut_avg, np.mean(amplitude_envelope), insta_f_cut_std