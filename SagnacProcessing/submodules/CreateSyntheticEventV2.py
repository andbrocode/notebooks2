#!/usr/bin/env python
# coding: utf-8

import random
from scipy.signal import ricker, resample
from numpy import zeros, random, convolve, arange, kaiser, hamming, array
from submodules.MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
from obspy.core.trace import Trace

def __create_synthetic_event_v2(T, sps, f_lower, f_upper, noise=False, padding=None, noise_level=None):
    
    '''
    
    Dependency:
       
        import random
        from scipy.signal import ricker, resample
        from numpy import zeros, random, convolve, arange, kaiser, hamming, array
        from submodules.MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
        from obspy.core.trace import Trace

    Example:
    
        x, t = __create_synthetic_event_v2(T, sps, f_lower, f_upper, noise=False, padding=None, noise_level=None)
        
    '''
    
    
    ## reduce the ull window due to efficiency
    Npts = int(T*sps/10)
    

    ## define a ricker wavelet
    w = 2
    n = 1e2
    rick = ricker(n, w)

    ## define random locations 
    locations = zeros(Npts)

    ## exclude edges to avoid edge effects 
    k = int(Npts/5)
    
    for i in range(k):
        if padding is not None:
            border = padding 
        else:
            border = k
            
        n = random.randint(border, int(Npts-border))
        
        ## set spikes with random amplitude within 0.8 and 1.2
        locations[n] = random.randint(0.8,1.2)
        
    ## taper 
    locations *= hamming(Npts)

    ## convolve ricker wavelet with locations 
    if noise_level:        
        print("noise added ...")
        event = convolve(rick, locations, 'same') 
        noises = array(random.rand(event.size))
        event += noise_level * noises
    else:
        event = convolve(rick, locations, 'same')
        
        
    ## time axis as it should be (for resample)
    timeline = arange(0,T+1/sps,1/sps)

    ## resample to acutal sampling rate (= sps) and apply taper
    event = resample(event, int(timeline.size)) 
    
    ## create a trace for the data
    xtrace = Trace(event)
    
    ## add sampling rate to the meta data
    xtrace.stats.sampling_rate=sps
    
    ## filter the trace with a bandpass to achieve the desired frequency content of the synthetic
    xtrace.filter('bandpass', freqmin=f_lower, freqmax=f_upper/2, corners=4, zerophase=True)
    
    event = xtrace.data

    return event, timeline
