#!/usr/bin/env python
# coding: utf-8

from scipy.signal import resample
from numpy import arange


def __interpolation(trace, time, T, sps):
    
    
    ## interpolate modeltrace to sps 
    l1 = trace.size
    
    trace = resample(trace, int(T*sps+1)) ## using FFT
#     time = resample(time, int(T*sps+1))    
    
    l2 = trace.size
    
    print(f"modeltrace is interpolated: {l1} samples --> {l2} samples")
#     print(f"time_modeltrace is interpolated: {l1} samples --> {l2} samples")

    time = arange(0, T+1/sps, 1/sps)

    return trace, time