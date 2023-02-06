#!/usr/bin/env python
# coding: utf-8

import random
from scipy.signal import ricker, resample
from numpy import zeros, random, convolve, arange, kaiser, hanning, array, pi, exp, interp, roll
from submodules.MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
from obspy.core.trace import Trace

def __create_synthetic_event_v3(T, sps, f_lower, f_upper, noise_level=None):
    
    '''
    Creates a tapered random signal with defined bandwidth based on a convolution of ricker wavelets and random refelectivty. 
    
    Dependency:
       
        import random
        from scipy.signal import ricker, resample
        from numpy import zeros, random, convolve, arange, kaiser, hamming, array
        from submodules.MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
        from obspy.core.trace import Trace

    Example:
    
        x, t = __create_synthetic_event_v3(T, sps, f_lower, f_upper, noise_level=None)
        
    '''
    
    def ricker_wavelet(f, length=0.128, dt=0.001):
        t = arange(-length/2, (length-dt)/2, dt)
        y = (1.0 - 2.0*(pi**2)*(f**2)*(t**2)) * exp(-(pi**2)*(f**2)*(t**2))
        return t, y
    
    
    ## reduce the window due to efficiency
    Npts = int(T*sps)
    dt = 1/sps
    
    
    ## define a ricker wavelet
    _ ,rick = ricker_wavelet(f_upper/2, length=Npts, dt=dt)

    ## set spikes with random amplitude within 0.8 and 1.2
    spikes = [random.randint(0,100)/100 for i in range(0, Npts)]

    
    ## taper spike distribution
#     spikes *= hanning(Npts)

    
    ## convolve ricker wavelet with locations 
    if noise_level is not None:        
        print("noise added ...")
        event = convolve(rick, spikes, 'same') 
        noises = array(random.rand(event.size))
        event += noise_level * noises
    else:
        event = convolve(rick, spikes, 'same')
    
    
    ## cut out event data. lots of padding before and after as a result of the convolution
    event = event[len(event)//2-Npts//2:len(event)//2+Npts//2]


    ## time axis as it should be (for resample)
    timeline = arange(0,T+1/sps,1/sps)

    
    ## resample to acutal sampling rate (= sps) and apply taper
    event = resample(event, int(timeline.size)) 
#     event = interp(timeline, arange(0, Npts, dt), event)
    
    ## apply a hanning window
    event *= hanning(len(event))
    
    return event, timeline
