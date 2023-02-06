#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from numpy import linspace, size, arange


def __processing_modeltrace(stream, time_shift, duration):
    
    dat = stream[0].copy()
    tmp = stream[0].copy()

    tbeg = stream[0].stats.starttime
    
    
    ## filter corners
    print("\napplying lowpass filter...")
    dat.filter("lowpass", freq=1.0, corners=2)

    ## Sampling rate of original seismogram and sampling interval
    dt, fs  = dat.stats.delta, 1/dat.stats.delta

    ## Choose window (otherwise core runs too long)
    print("\ntrimming trace ...")
    t1 , t2  = time_shift*60, (time_shift*60+duration)
    dat.trim(tbeg + t1, tbeg + t2)
    print(f' from {t1} sec to {t2} sec') 
#     print(min(tmp.times("Matplotlib")), max(tmp.times("Matplotlib")))

    ## Taper at both ends (one could also use obspy taper!)
    print("\napplying lowpass filter...")
    dat.taper(0.1, type='hann', max_length=None, side='both')

    ## Initialize time for original data
#     timeline = linspace(0, size(dat) * dt, size(dat))
    timeline = arange(0, dat.stats.npts*dt,dt)
    
    ## Print metadata and min max values 
    print('\nMaximum amplitude RLAS: ', round(stream[0].max()*1e7,3), 'e-7 rad/s')
    print('Maximum amplitude selection: ', round(dat[0].max()*1e7,3), 'e-7 rad/s')

    ## interpolation
    #dat.interpolate(sampling_rate = sps)

    t_axis = linspace(0,tmp.stats.npts*tmp.stats.delta,tmp.stats.npts)
    odata = dat[0].data
    
    
    ## ____________________________________________________
    
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    
    font = 13
    
    ax.plot(t_axis, tmp.data)
    ax.axvline(t1, color='darkred', linestyle='-')
    ax.axvline(t2, color='darkred', linestyle='-')
    
    ax.set_xlabel("Time (s)", fontsize=font)
    ax.set_ylabel(r"Amplitude $\frac{rad}{s}$", fontsize=font)
    
    plt.show();
    
    return odata, timeline

