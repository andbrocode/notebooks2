#!/usr/bin/env python
# coding: utf-8

from scipy.signal import upfirdn, resample_poly

def __upsample_FIR(signal_in, sps, T, sampling_factor=2):
    
    
    lower = 50
    upper = sampling_factor*lower
    
    signal_out = resample_poly(signal_in, upper, lower, padtype="line") ## using FIR filter
    
    ## adjsut sampling frequency with sampling factor
    sps_new = int(sps*sampling_factor)
    
    ## adjust time axis
    time_out = np.arange(0, T+1/sps_new, 1/sps_new)
    
    return signal_out[:-1], time_out
