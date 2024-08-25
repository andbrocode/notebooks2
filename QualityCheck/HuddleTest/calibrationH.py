#!/usr/bin/env python

# Calibration utilities
#
# main tool is calib_stream which calculates relative sensitivity matrix for huddle test data
# - function plot_calib_matrix creates plot
#
# C. Weidle changed
#

import matplotlib.pyplot as plt
import numpy as np

from obspy.signal.util import next_pow_2
from obspy.signal.invsim import simulate_seismometer as seisSim
from obspy.core import AttribDict, Stream
# for checks of response information type
from obspy.core.inventory import Inventory
from obspy.io.xseed import Parser
from obspy import read_inventory


def calib_stream(st0, inv, path_to_xml, frange):  
    
    if path_to_xml[-1] == "/":
        path_to_xml = path_to_xml[:-1]
    
    st = st0.copy() # work on copy
    
    st.detrend('demean') # needed because submitted data may be sliced from longer (demeaned) original
    
    

    # remove poles and zeros ## but NOT sensitivity ## and replace data with its spectrum
    NPTS = max([tr.stats.npts for tr in st])

    nfft = next_pow_2(NPTS)

    freqs = np.fft.fftfreq(nfft, st[0].stats.delta)[:int(nfft/2+1)] # only real frequencies
    
    freqs[-1] = np.abs(freqs[-1]) # max frequency should be positive

    for tr in st:
        
        ## read local response file 
        inv = read_inventory(path_to_xml + "/" + tr.stats.station + ".xml") #response files
        
        ## extract poles and zeros from response file
        r = inv[0][0][0].response.get_paz()
        
        paz = {'poles': r.poles, 'zeros': r.zeros, 'gain': r.normalization_factor}
        
        ## simulate response function
        tr.simulate(paz_remove=paz, remove_sensitivity=False)
        
        ## transform back to time domain
        tr.data = np.fft.rfft(tr.data, nfft)
        
        print(tr.stats.station + ' response removed')

#     return calib_spec(st, freqs, frange=frange)
    return st, freqs, frange



def calib_spec(sp, freqs, frange=[.3, 3]):  #enter frequency range
    '''
    calibrate relative sensitivity matrix of __SPECTRA__ in Stream object sp

    sp: Stream, contains rfft spectra of traces to be calibrated against each other
    freqs: np.array, frequencies corresponding to samples in Spectra sp

    frange: list, frequency range in which the amplitude response of tr and trref will be compared, 
            commonly around calibration frequency, which is often 1Hz. Thus default is 3s to 3Hz

    returns sensitivity matrix
        each row is a reference instrument, each column abs(XCORR/ACORR), averaged over frequencies in frange
        sensitivity is thus in each row RELATIVE to the reference instrument, i.e. diagonal of matrix = 1
    '''

    sens = np.zeros((sp.count(), sp.count()))
    for i, tr in enumerate(sp):
        acorr = tr.data * np.conj(tr.data) # is real, np.all(acorr.imag == 0) is True
        for j in range(i, sp.count()):
            xcorr = sp[j].data * np.conj(tr.data)
            sens[i, j] = np.abs((xcorr/acorr)[np.where(np.logical_and(freqs>min(frange), freqs<max(frange)))]).mean()
#           if i != j: sens[j, i] = 1 / sens[i, j] ## fill lower triangle of matrix
            if i != j: sens[j, i] = np.nan
    return sens



def myfmt(a):
        if np.log10(abs(a)) > 3: return '%.0e' % a
        elif np.log10(abs(a)) >= 0 or np.log10(abs(a)) == -np.inf: return '%i' % a # a>=1 or a=0
        elif np.log10(abs(a)) >= -1: return '%.1f' % a 
        elif np.log10(abs(a)) >= -2: return '%.2f' % a
        else: return '%.0e' % a

        
        
def plot_calib_matrix(pct, title_str, lbls, fname=None, cmap=plt.get_cmap(), full=False, 
                      clabel='deviation in spectral amplitude [%]', vmax=None):
    '''
    plot calibration matrix

    pct: np.array, 2D matrix of relative sensitivities (result from calib_stream)
    title_str: str, title of figure
    lbls: str, x- and y-axis labels

    optional:
        fname: str, filename of figure to be saved, if 'None' figure will be displayed
        cmap: matplotlib.cmap
        full: bool, fill up half matrix with inverse values, default: False
        clabel: str, colorbar label, what is shown in matrix pct?
        vmax: float or None, limit of colorbar or None (default) which determines it automatically from standard deviation
    '''
    flip = 0
    if full:
        m = pct.copy()
        m[np.isnan(m)] = 0.
        n = np.triu(1/m,1).T
        n[np.isnan(n)] = 0.
        pct = n+m

    if np.isnan(pct[-1, 0]) or full: 
        pct = np.flipud(pct) # flip to lower triangle
        flip = 1

    plt.figure(figsize=(16,12))
    if vmax is None: vmax = min(np.nanstd(pct), 30) # cap to 30%
    plt.imshow(pct, origin="upper", interpolation="nearest", cmap=cmap, vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(shrink=.8)
    cb.set_label(clabel)
    plt.title(title_str)
    plt.xlabel('Instrument ID')
    plt.ylabel('Reference Instrument ID')

    plt.xticks(range(len(lbls)), lbls, rotation=75, ha='center')
    plt.yticks(range(len(lbls)), lbls[::-1] if flip else lbls)

    # overlay values
    #i = np.arange(pct.shape[0])
    i = np.arange(len(lbls))
    x, y = np.meshgrid(i, i)
    for k, l in zip(x.flatten(), y.flatten()):
        if not np.isnan(pct[l, k]): 
            plt.text(k, l, myfmt(pct[l, k]), fontsize=10, ha='center', color='w', 
                     bbox={'pad': 2, 'alpha': .07, 'color': 'k', 'edgecolor': "none"}) 

    # save or show
    if fname is not None: 
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return
