#!/usr/bin/env python
# coding: utf-8

from obspy import read

def __load_signal_mseed(option, path='./'):
    
    global time_synthetic_signal 
    global synthetic_signal
    global time_synthetic_event
    global synthetic_event
    
    ## load signal and assign arrays
    signal = read(f"synthetic_signal_opt{option}.mseed")

    synthetic_signal = signal[0].data
    time_synthetic_signal = signal[1].data

    ## load event and assign arrays
    event = read(f"synthetic_event_opt{option}.mseed")

    synthetic_event = event[0].data
    time_synthetic_event = event[1].data
    

    