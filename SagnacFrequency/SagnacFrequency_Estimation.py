#!/usr/bin/env python
# coding: utf-8

# ## Compute Sagnac Frequency


import os, gc, json
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from scipy.signal import welch, periodogram
from numpy import zeros, argmax, arange
from tqdm import tqdm
from pandas import DataFrame, date_range
from datetime import datetime, date

from andbro__querrySeismoData import __querrySeismoData


## Configuration

config = {}

config['ring'] = "U"

config['seed'] = f"BW.DROMY..FJ{config['ring']}"

config['tbeg'] = "2022-04-24"
config['tend'] = "2022-06-12"

config['outpath_data'] = f"/import/kilauea-data/sagnac_frequency/hilbert_60_R{config['ring']}_multi/"

config['repository'] = "george"

config['method'] = "hilbert" ## "welch" | "periodogram" | multitaper

rings = {"Z":553, "U":302, "V":448,"W":448}

config['f_expected'] = rings[config['ring']]  ## expected sagnac frequency
config['f_band'] = 3 ## +- frequency band



## Variables

config['dn'] = 60  ## seconds
config['buffer'] = 0  ## seconds
config['offset'] = 30 ## seconds

config['loaded_period'] = 3600  ## seconds
config['NN'] = int(config['loaded_period']/config['dn'])

#config['nblock'] = 300*5000
#config['noverlap'] = None

#config['n_windows'] = 3



## Methods

def __multitaper_estimate(data, fs, n_windows=4, one_sided=True):

    from spectrum import dpss, pmtm
    from numpy import zeros, arange, linspace

    NN = len(data)

    spectra, weights, eigenvalues = pmtm(data, NW=2.5, k=n_windows, show=False)

    ## average spectra
    estimate = zeros(len(spectra[0]))
    for m in range(n_windows):
        estimate += (abs(spectra[m])**2)
    estimate /= n_windows

    l = len(estimate)
    frequencies = linspace(-0.5*fs, 0.5*fs, l)

    if one_sided:
        f_tmp, psd_tmp = frequencies[int(l/2):], estimate[:int(l/2)]
    else:
        f_tmp, psd_tmp = frequencies, estimate


    f_max = f_tmp[argmax(psd_tmp)]
    p_max = max(psd_tmp)
    h_tmp = __half_width_half_max(psd_tmp)

    return f_tmp, f_max, p_max, h_tmp


def __multitaper_hilbert(config, st, fs, n_windows, plot=False):
    
    import numpy as np

    from scipy.signal import hilbert
    from spectrum import dpss, pmtm

    N = len(st[0].data)
    
    [tapers, eigen] = dpss(N, 2.5, n_windows)

    
    f_lower = config['f_expected'] - config['f_band']
    f_upper = config['f_expected'] + config['f_band']
   

    tmp_insta_f_cut_mean = zeros(n_windows)
    tmp_amp = zeros(n_windows)
    tmp_std_dev = zeros(n_windows)
    
    for ii in range(n_windows):

        st0 = st.copy()
    
        ## bandpass with butterworth
        st0.detrend("simple")
        st0.taper(0.01)
        st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)
        
        
        ## estimate instantaneous frequency with hilbert
        signal = st0[0].data*tapers[:, ii]

#         plt.plot(signal)
#         plt.show();
        
        analytic_signal = hilbert(signal)

        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

        ## cut first and last 5% (corrupted)
        dd = int(0.05*len(instantaneous_frequency))

        t = st0[0].times()
        t1 = st0[0].times()[1:]
        t2 = t1[dd:-dd]

        t_mid = t[int((len(t))/2)]

        insta_f_cut = instantaneous_frequency[dd:-dd]

        ## averaging
        tmp_insta_f_cut_mean[ii] = np.mean(insta_f_cut)
    #     insta_f_cut_mean = np.median(insta_f_cut)

        tmp_amp[ii] = np.mean(amplitude_envelope)
        tmp_std_dev[ii] = np.std(insta_f_cut)
    
    
    insta_f_cut_mean = np.sum(tmp_insta_f_cut_mean)/n_windows
    amp = np.sum(tmp_amp)/n_windows
    std_dev = np.sum(tmp_std_dev)/n_windows
    
    
    if plot:
        st0.plot(equal_scale=False);

        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.plot(t, signal, label='signal')
        ax0.plot(t, amplitude_envelope, label='envelope')
        ax0.set_xlabel("time in seconds")
        ax0.legend()
        ax1.plot(t2, insta_f_cut)
        ax1.set_xlabel("time in seconds")
        ax1.set_ylim(552, 555)
        fig.tight_layout()
    
    return t_mid, insta_f_cut_mean, amp, std_dev


def __multitaper_periodogram(data, fs, n_windows=4):

    from spectrum import dpss, pmtm
    from numpy import zeros, arange, linspace
    from scipy.signal import find_peaks, peak_widths

    [tapers, eigen] = dpss(len(data), 2.5, n_windows)

    f_maxima, p_maxima, hh  = zeros(n_windows), zeros(n_windows), zeros(n_windows)

    for ii in range(n_windows):

        f_tmp, psd_tmp = periodogram(data,
                                     fs=fs,
                                     window=tapers[:,ii],
                                     nfft=None,
                                     detrend='constant',
                                     return_onesided=True,
                                     scaling='density',
                                     )

        p_maxima[ii] = max(psd_tmp)
        f_maxima[ii] = f_tmp[argmax(psd_tmp)]

        ## half widths
        xx = psd_tmp[argmax(psd_tmp)-10:argmax(psd_tmp)+10]
        peaks, _ = find_peaks(xx)
        half = peak_widths(xx, peaks, rel_height=0.5)
        idx = argmax(half[1])
        hh[ii] = abs(half[3][idx] -half[2][idx])

    f_mean = sum(f_maxima) / n_windows
    p_mean = sum(p_maxima) / n_windows
    h_mean = sum(hh) / n_windows

    return f_tmp, f_mean, p_mean, h_mean


def __hilbert_frequency_estimator(config, st, fs):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()


    f_lower = config['f_expected'] - config['f_band']
    f_upper = config['f_expected'] + config['f_band']


    ## bandpass with butterworth
    st0.detrend("demean")
    st0.taper(0.1)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)


    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    ## cut first and last 5% (corrupted)

    dd = int(0.05*len(instantaneous_frequency))

    t = st0[0].times()
    t1 = st0[0].times()[1:]
    t2 = t1[dd:-dd]

    t_mid = t[int((len(t))/2)]

    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## averaging
    insta_f_cut_mean = np.mean(insta_f_cut)
#    insta_f_cut_mean = np.median(insta_f_cut)

    return t_mid, insta_f_cut_mean, np.mean(amplitude_envelope) ,np.std(insta_f_cut)


def __compute(config, st0, starttime, method="hilbert"):

    from scipy.signal import find_peaks, peak_widths, welch, periodogram
    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['dn']

    tt, ff, hh, pp = zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    while n2 <= config['loaded_period']:

#         try:

        ## cut stream to chuncks 
        st_tmp = st0.copy().trim(starttime+n1-config['buffer']-config['offset'], starttime+n1+config['dn']+config['buffer']-config['offset'])

        ## get time series from stream
        times = st_tmp[0].times(reftime=UTCDateTime("2016-01-01T00"))

        ## get sampling rate from stream
        df = st_tmp[0].stats.sampling_rate


        if method == "hilbert":

            f_tmp, f_max, p_max, h_tmp = __hilbert_frequency_estimator(config, st_tmp, df)

        elif method == "multitaper_hilbert":

            f_tmp, f_max, p_max, h_tmp = __multitaper_hilbert(config, st_tmp, df, n_windows=config['n_windows'])

        elif method == "multitaper_periodogram":

            f_tmp, f_max, p_max, h_tmp = __multitaper_perio(st_tmp[0].data, df, n_windows=config['n_windows'])

        elif method == "periodogram":

            f_tmp, f_max, p_max, h_tmp = __periodogram_estimate(st_tmp, df)

        elif method == "multitaper":

            f_tmp, f_max, p_max, h_tmp = __multitaper_estimate(st_tmp[0].data, df, n_windows=config['n_windows'], one_sided=True)                


        ## append values to arrays
        tt[ii] = times[int(len(times)/2)]
        ff[ii] = f_max
        pp[ii] = p_max
        hh[ii] = h_tmp

#         except:
#             tt[ii], ff[ii], pp[ii], hh[ii] = nan, nan, nan, nan
#             print(" -> computing failed")

        ii += 1
        n1 += config['dn']
        n2 += config['dn']

    return tt, ff, hh, pp


## Looping

tbeg = date.fromisoformat(config['tbeg'])
tend = date.fromisoformat(config['tend'])

#print(config['method'])

print(json.dumps(config, indent=4, sort_keys=True))

for date in date_range(tbeg, tend):

    print(date)

    idx_count=0
    NNN = int(86400/config['dn'])

    t, f, p, h = zeros(NNN), zeros(NNN), zeros(NNN), zeros(NNN)

    for hh in tqdm(range(24)):

        ## define current time window
        dh = hh*3600
        t1, t2 = UTCDateTime(date)+dh, UTCDateTime(date)+config['loaded_period']+dh

        try:
	    ## load data for current time window
#            print(" -> loading data ...")
            st, inv = __querrySeismoData(
                                seed_id=config['seed'],
                                starttime=t1-config['buffer']-2*config['offset'],
                                endtime=t2+config['buffer']+2*config['offset'],
                                repository=config['repository'],
                                path=None,
                                restitute=None,
                                detail=None,
                                )
        except:
#            print("failed to load data!")
            continue

        ## compute values
#        print(" -> computing ...")
        tt, ff, hh, pp = __compute(config, st, t1, method=config['method'])


        ## combine with previous values
        for mm in range(len(tt)):
            t[idx_count] = tt[mm]
            f[idx_count] = ff[mm]
            p[idx_count] = pp[mm]
            h[idx_count] = hh[mm]
            idx_count += 1

        del st, tt, ff, hh, pp
        gc.collect()


    ## create and write a dataframe
    df = DataFrame()
    df['times'] = t
    df['freqs'] = f
    df['hmhw']  = h
    df['psd_max'] = p

    date_str = str(date)[:10].replace("-","")
    df.to_pickle(f"{config['outpath_data']}{date_str}.pkl")


## END OF FILE
