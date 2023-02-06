#!/usr/bin/env python
# coding: utf-8

from numpy import append, array
from tqdm import tqdm
from HilbertFilter import __hibert_filter

def __demodulate_with_windows(x, y, Twindow, sps):

    window_size = int(Twindow * sps)

    stepsize = int(window_size/2) 

    out1, out2 = array([]), array([])
    for i in tqdm(range(0,len(x), stepsize)):

        if int(i+window_size) > len(x):
            print(f"break for {i/sps}")
            break

        win1 = x[i:int(i+window_size)]
        win2 = y[i:int(i+window_size)]


        t, sig = __hibert_filter(win1, win2, sps)


        out1 = append(out1, sig[int(stepsize/2):window_size-int(stepsize/2)])
        out2 = append(out2, t[int(stepsize/2):window_size-int(stepsize/2)])

    return out1, out2

    