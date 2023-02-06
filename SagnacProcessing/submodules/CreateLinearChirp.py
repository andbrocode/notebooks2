#!/usr/bin/env python
# coding: utf-8

from numpy import arange, sin, pi

def __create_linear_chirp(T, sps, f_lower, f_upper):

    Npts = int(T*sps)

    f0 = f_lower   
    f1 = f_upper

    t = arange(0, T+1/sps, 1/sps)

    ## define splope
    m = (f1-f0)/2/t[-1]

    ## create lilnear chirp signal
    chirp = sin(2*pi*(f0 + m * t)*t)

    return chirp, t
