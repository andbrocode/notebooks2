#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from numpy import concatenate, arange

def __trace_padding(modeltrace, sps, T):
    
    
    Npts = int(sps*T)
    
    n = int(Npts-len(modeltrace))+400+1

    noise = (np.random.rand(n)-0.5)*np.std(modeltrace[:500]) 
    
    trace = concatenate((noise[:int(n // 2 + 200)], modeltrace[200:-200],noise[int(n // 2+200):])) 
    
    time_trace = arange(0, T+1/sps, 1/sps)

    
    return trace, time_trace
