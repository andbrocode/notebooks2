#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

from submodules.FastFourierTransform import __fast_fourier_transform
from numpy import arange, argmax



def __makeplot_trace_and_spectrum(trace, timeline, fmax=None, fmin=None, text=True):
    
    font = 13
    
    N = len(trace)
    
    if timeline is None:
        delta = 1/sps
        timeline = arange(0, N/sps, 1/sps)
    else:
        if str(type(trace)) == "<class 'obspy.core.trace.Trace'>": 
            delta = trace.stats.delta
        else:
            delta = timeline[1]-timeline[0]
    
    trace_fft, ff = __fast_fourier_transform(signal_in=trace, dt=delta , window=None, normalize=None)
    
    ## __________________________________________________________
    ##
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))


    ax1.plot(timeline, trace)

    ax2.plot(ff[:N // 2],abs(trace_fft[:N // 2]))
    
    ax1.set_xlabel("Time (s)", fontsize=font)
    ax1.set_ylabel(r"Amplitude ($\frac{rad}{s}$)", fontsize=font)
    
    ax2.set_xlabel("Frequency (Hz)", fontsize=font)
    ax2.set_ylabel(r"Amplitude Spectral Density ($\frac{rad}{s \sqrt{Hz} }$)", fontsize=font)

    
    if text:
        ff_cut = ff[:N // 2]
        max_y, max_x = max(abs(trace_fft[:N // 2])), ff_cut[argmax(abs(trace_fft[:N // 2]))]
        ax2.annotate(f'(x: {round(max_x,2)} | y: {round(max_y,5)})', xy=(max_x, max_y), xytext=(max_x, max_y))


    
    if fmax and fmin:
        ax2.set_xlim(fmin, fmax)
    elif fmax and not fmin:
        ax2.set_xlim(0, fmax)
  
    plt.show();
    
    return fig
