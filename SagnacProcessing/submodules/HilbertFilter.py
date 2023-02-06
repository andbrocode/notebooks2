#!/usr/bin/env python
# coding: utf-8

from scipy.signal import hilbert
from numpy import gradient, imag, append, real, pi, isnan

def __hibert_filter(sig_in, time_in, fs):
    
    '''
    estimating the instantaneous frequency by using the formula of Jo
    
    sig_in    = input signal
    time_in   = input timeline
    fs        = sampling frequency of digital signal
    '''

    def __check_for_NaN(array):
        
        sum0, idx = 0, []
        for k, x in enumerate(array):
            if isnan(x):
                sum0 += 1
                idx.append(k)

        if sum0 != 0:
            print(sum0, f" nan found of {len(array)}")
    
        return(idx)
            
            
            
    ## calulcate hilbert transform
    hil0 = hilbert(sig_in)
    
    ## extract imaginary part of hilbert transform 
    hil = imag(hil0)
    
#     env = abs(hil)
#     hil = hil/max(env)
    
    ## calculate derivatives (second order central differences)
    d_hil = gradient(hil, edge_order=1)*fs
    d_sig = gradient(sig_in, edge_order=1)*fs
    
    
    ## check if nan are in derivative
#     idx = __check_for_NaN(d_hil)
    
    
#     delta_f_full = (sig_in * d_hil - d_sig * hil) / (2*np.pi*np.sqrt(sig_in**2 + hil**2))
    
    ## without sqrt as found on Wikipedia
    delta_f_full = (sig_in * d_hil - d_sig * hil) / (2*pi*(sig_in**2 + hil**2))


    ## extract real part
    delta_f = real(delta_f_full)
    
    
    ## instert nan value for time zero (exluded bevause of np.diff() ) 
    idx = __check_for_NaN(delta_f)
    
    for l in idx:
        delta_f[l] = 0.0

    
    return time_in, delta_f
