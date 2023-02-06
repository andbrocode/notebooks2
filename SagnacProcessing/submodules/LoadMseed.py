#!/usr/bin/env python
# coding: utf-8

from os.path import isfile
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from CreateSyntheticEventV2 import __create_synthetic_event_v2
from obspy import read

def __load_mseed(iname, T, sps, f_lower, f_upper, noise_level=None):
    
    if not isfile(iname):
        
        if noise_level is not None:
            idata, itime = __create_synthetic_event_v2(T, sps, f_lower, f_upper, noise=True, padding=10, noise_level=noise_level)            
        else:
            idata, itime = __create_synthetic_event_v2(T, sps, f_lower, f_upper, noise=False, padding=10)

        ## create trace object
        odata = Trace(idata)
        otime = Trace(itime)

        ## edit header of trace 
        odata.stats.sampling_rate = sps

        otime.sampling_rate = sps

        ##create a stream
        synthetic_event_out = Stream(traces=[odata, otime])

        ## write trace object to file
        # oname = f"data/syn_opt{option}_scheme{scheme}_T{T}.mseed"

        synthetic_event_out.write(iname, format="MSEED")

        print(f"stored synthetic event to: {iname}")


    ## load signal and assign arrays
    signal = read(iname)

    signal[0].filter('bandpass', freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

    return signal[0].data, signal[1].data

    