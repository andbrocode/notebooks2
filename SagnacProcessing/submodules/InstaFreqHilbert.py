#!/usr/bin/env python
# coding: utf-8

from numpy import unwrap, diff, insert, isnan, nan, pi, angle
from scipy.signal import hilbert

def __insta_freq_hilbert(sig_in, time_in, fs, sgnc):
    
    ''' Estimation of the instantaneous frequency (modulated signal) by using integrated python methods '''
    
    sig_hil = hilbert(sig_in)
    
    insta_phase = unwrap(angle(sig_hil))
    
    insta_freq  = diff(insta_phase) * (2.0 * pi)  * fs

    ## instert nan value for time zero (exluded because of np.diff() ) 
    insta_freq = insert(insta_freq, 0, nan, axis=0)
    
    
    c=0
    for i, spl in enumerate(insta_freq):
        if isnan(spl):
            insta_freq[i] = sgnc
            c += 1
    print(f"{c} nan removed !")
    
    
    return time_in, insta_freq