#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

from numpy import arange
from numpy.fft import fftshift
from scipy.signal import welch

def __makeplot_trace_and_psd(trace, timeline, fmax=None, fmin=None, t1=None, t2=None):
    
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
    
    fs = 1/(timeline[1]-timeline[0])

    freqs, signal_psd = welch(trace, fs, return_onesided=False, nperseg=1e5, scaling="density")
    freqs = fftshift(freqs)
    signal_psd = fftshift(signal_psd)    
    
    print(freqs[abs(signal_psd).argmax()], max(abs(signal_psd)))
        
    ## __________________________________________________________
    ##
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    
    
    ax1.plot(timeline, trace)

#     ax2.plot(freqs[:N // 2],abs(signal_psd[:N // 2]))
    ax2.plot(freqs, abs(signal_psd))
    ax2.set_yscale('log')
    ax2.grid(which='minor', color='k', alpha=0.2, linestyle='--')
    ax2.grid(which='major', color='k', alpha=0.4, linestyle='--')
    
    
    ax1.set_xlabel("Time (s)", fontsize=font)
    ax1.set_ylabel(r"Amplitude ($\frac{rad}{s}$)", fontsize=font)
    
    ax2.set_xlabel("Frequency (Hz)", fontsize=font)
    ax2.set_ylabel(r"Power Spectral Density ($\frac{rad^2}{s^2 Hz }$)", fontsize=font)
    
    if fmax and fmin:
        ax2.set_xlim(fmin, fmax)
    elif fmax and not fmin:
        ax2.set_xlim(0, fmax)

    if t1 is not None and t2 is not None:
        ax1.set_xlim(t1, t2)
        
    plt.show();
    
    return fig
