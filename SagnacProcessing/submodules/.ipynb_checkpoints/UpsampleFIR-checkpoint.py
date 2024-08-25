def __upsample_FIR(signal_in, sps, T, sampling_factor=2):

    from scipy.signal import upfirdn, resample_poly
    from numpy import arange

    lower = 50
    upper = sampling_factor*lower

    # upsampling using FIR filter
    signal_out = resample_poly(signal_in, upper, lower, padtype="line")

    # adjsut sampling frequency with sampling factor
    sps_new = int(sps*sampling_factor)

    # adjust time axis
    times = arange(0, T+1/sps_new, 1/sps_new)

    return signal_out[:-1], times
