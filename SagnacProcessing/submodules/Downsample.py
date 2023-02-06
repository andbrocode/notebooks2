#!/usr/bin/env python
# coding: utf-8

from scipy.signal import decimate 
from numpy import arange 

def __downsample(signal_in, sps, ds_factor=2):

    
    ## downsample using a FIR filter
    signal_out = decimate(signal_in, ds_factor, n=None, ftype='fir', axis=-1, zero_phase=True)


    ## adjust sampling frequency
    sps = int(sps/ds_factor)

    ## adjust time axis
    time_out = arange(signal_out.size)/sps

    return signal_out, time_out, sps
