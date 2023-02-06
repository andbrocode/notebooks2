#!/usr/bin/env python
# coding: utf-8

import random
from scipy.signal import ricker, resample
from numpy import zeros, random, convolve, arange, kaiser, hanning, array, pi, exp, interp, roll, sin
from submodules.MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
from obspy.core.trace import Trace
from submodules.Tapering import __tapering

def __create_synthetic_event_v4(config, signal_type='ricker', noise_level=None, set_taper=True, set_normalize=True):
    
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
    
    def __check_config(config):
        
        expected_keys = ['T', 'sps', 'f_lower', 'f_upper']
        provided_keys = config.keys()
        
        for key in expected_keys:
            if key not in provided_keys:
                print(f"configuration for {key} is incomplete!")
    
    def __ricker_wavelet(f, length=0.128, dt=0.001):
        t = arange(-length/2, (length-dt)/2, dt)
        y = (1.0 - 2.0*(pi**2)*(f**2)*(t**2)) * exp(-(pi**2)*(f**2)*(t**2))
        return t, y
    
    def __linear_chirp(config):
        times = arange(0, config['T']+1/config['sps'], 1/config['sps'])
        slope = (config['f_upper']-config['f_lower'])/2/times[-1]
        chirp = sin(2*pi*(config['f_lower'] + slope * times)*times)
        return times, chirp
    
    
    ## check if all variables are contained in the provided configuration
    __check_config(config)
    
    ## reduce the window due to efficiency
    Npts = int(config['T']*config['sps'])
    dt = 1/config['sps']
    
    ## check if array might require to much storage. If true, reduce sampling rate and upsample later
    if Npts > 5000:
        T0, sps0 = config['T'], config['sps']
        
        config['T'], config['sps'] = 1600, 50
        
        Npts = int(config['sps']*config['T'])
        dt = 1/config['sps']        

        upsample = True
    else:
        upsample = False
    
    ## set spikes with random amplitudes
    spikes = array([round(random.uniform(-1,1), 2) for i in range(0, Npts)]    )
    
    ## taper the spikes array
#     spikes = __tapering(spikes, taper_type='hann', percent=0.2)
    
    
    ## define a ricker wavelet
    if signal_type == "ricker":
        _ , signal = __ricker_wavelet(config['f_upper']/2, length=Npts, dt=dt)

    elif signal_type == "chirp":
        _, signal = __linear_chirp(config)

    ## convolve ricker wavelet with locations 
    if noise_level is not None:        
        print("random noise is added ...")
        event = convolve(signal, spikes, 'same') 
        ## add random noise
        event += noise_level * array(random.rand(event.size)) * max(abs(event))
    else:
        event = convolve(signal, spikes, 'same')
    
    ## cut out event data. lots of padding before and after as a result of the convolution
    event = event[len(event)//2-Npts//2:len(event)//2+Npts//2]

    ## if array size had to be reduced due to storage, then resampling is applied to recover required size
    if upsample:
        config['T'], config['sps'] = T0, sps0
        event = resample(event, int(sps0*T0+1))
        print(f'upsamling trace ....')
        
    ## time axis as it should be (for resample)
    timeline = arange(0,config['T']+1/config['sps'],1/config['sps'])
        
    ## apply a taper to the event array
    if set_taper:
        event = __tapering(event, taper_type='hann', percent=0.2)
        print(f'tapering with Hanning window for 0.2 percent...')
        
    ## normalize the event array
    if set_normalize:
        event /= max(abs(event))
        print(f'normalizing trace ...')

    return event, timeline
