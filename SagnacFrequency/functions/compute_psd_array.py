def __compute_psd_array(st0, twin_sec=60, spec="PSD", sort=True):

    from scipy.signal import welch
    from scipy.signal import get_window
    from scipy.fft import fft, fftfreq, fftshift
    from scipy import signal
    from numpy import angle, imag

    def __get_fft(signal_in, dt, window=None):

        ## determine length of the input time series
        n = int(len(signal_in))

        # calculate spectrum (with or without window function applied to time series)
        if window:
            win = signal.get_window(window, n);
            spectrum = fft(signal_in * win)

        else:
            spectrum = fft(signal_in)

        # calculate frequency array
        frequencies = fftfreq(n, d=dt)

        # correct amplitudes of spectrum
        magnitude = abs(spectrum) * 2.0 / n

        phase = angle(spectrum, deg=False)
        # phase = imag(spectrum)

        ## return the positive frequencies
        return magnitude[0:n//2], frequencies[0:n//2], phase[0:n//2]

    _st = st0.copy()

    if sort:
        _st.sort(keys=['channel'], reverse=True)

    nblock = int(_st[0].stats.sampling_rate * twin_sec)
    overlap = int(0.5*nblock)
    win = get_window('hann', nblock, True)

    Pxxs, ffs, chs = [], [], []

    for i, tr in enumerate(_st):
        if spec.upper() == "PSD":
            ff, Pxx = welch(tr.data,
                            fs=tr.stats.sampling_rate,
                            window=win,
                            noverlap=overlap,
                            nfft=nblock,
                            scaling="density",
                            return_onesided=True)

        elif spec.upper() == "SPEC":
            ff, Pxx = welch(tr.data,
                            fs=tr.stats.sampling_rate,
                            window=win,
                            noverlap=overlap,
                            nfft=nblock,
                            scaling="spectrum",
                            return_onesided=True)

        elif spec.upper() == "FFT":
            Pxx, ff, ph = __get_fft(tr.data, tr.stats.delta)

        ffs.append(ff)
        chs.append(tr.stats.channel)
        Pxxs.append(Pxx)

    out = {}
    out['Pxxs'] = Pxxs
    out['ffs'] = ffs
    out['chs'] = chs

    return out