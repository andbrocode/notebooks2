def __get_fft_values(signal_in, dt, f_sagn, window=None):

    from numpy import argmax, sqrt, where, argmin, gradient, mean
    from scipy.fft import fft, fftfreq, fftshift
    from scipy import signal
    from numpy import angle, imag, unwrap

    # determine length of the input time series
    n = int(len(signal_in))

    signal_in = fftshift(signal_in)

    # calculate spectrum (with or without window function applied to time series)
    if window:
        win = signal.get_window(window, n);
        spectrum = fft(signal_in * win, norm="forward")

    else:
        spectrum = fft(signal_in, norm="forward")

    # calculate frequency array
    frequencies = fftfreq(n, d=dt)

    # correct amplitudes of spectrum
    # magnitude_corrected = abs(spectrum) *2 /n

    # none corrected magnitudes
    magnitude = abs(spectrum)

    # phase spectrum in radians
    phase = angle(spectrum, deg=False)

    # only one-sided
    freq = frequencies[0:n//2]
    spec = magnitude[0:n//2]
    pha = phase[0:n//2]

    # specify f-band around Sagnac frequency
    fl = f_sagn - 2
    fu = f_sagn + 2

    # get index of Sagnac peak
    idx_fs = where(spec == max(spec[(freq > fl) & (freq < fu)]))[0][0]

    # estimate Sagnac frequency
    f_sagn_est = freq[idx_fs]

    # estimate AC value at Sagnac peak
    AC_est = spec[idx_fs] * 2

    # estimate DC value at ff = 0
    DC_est = spec[0]

    # estimate phase at Sagnac peak
    phase_est = pha[idx_fs]

    return f_sagn_est, AC_est, DC_est, phase_est