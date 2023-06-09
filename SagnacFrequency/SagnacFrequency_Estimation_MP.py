#!/usr/bin/env python
# coding: utf-8

# ## Compute Sagnac Frequency


import os, gc, json
import matplotlib.pyplot as plt
import multiprocessing as mp

from obspy import UTCDateTime
from scipy.signal import welch, periodogram
from numpy import zeros, argmax, arange
from tqdm import tqdm
from pandas import DataFrame, date_range
from datetime import datetime, date

from andbro__querrySeismoData import __querrySeismoData
from andbro__utc_to_mjd import __utc_to_mjd



## Configuration

config = {}

config['ring'] = "Z"

config['seed'] = f"BW.DROMY..FJ{config['ring']}"

config['tbeg'] = "2023-04-07"
config['tend'] = "2023-04-07"

#config['outpath_data'] = f"/import/kilauea-data/sagnac_frequency/hilbert_60_R{config['ring']}_multi/"
config['outpath_data'] = f"/import/kilauea-data/sagnac_frequency/tests/"

config['outfile_appendix'] = "test28"

config['repository'] = "archive"

config['method'] = "periodogram" ## "hilbert" | "multitaper_hilbert" | "welch" | "periodogram" | multitaper | multitaper_periodogram

rings = {"Z":553, "U":302, "V":448,"W":448}

config['f_expected'] = rings[config['ring']]  ## expected sagnac frequency
config['f_band'] = 3 ## +- frequency band

config['n_windows'] = 10

config['t_steps'] = 60  ## seconds
config['t_overlap'] = 1200 ## seconds


config['loaded_period'] = 3600  ## seconds
config['NN'] = int(config['loaded_period']/config['t_steps'])

config['nblock'] = 300*5000
config['noverlap'] = None




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


def __multitaper_hilbert(config, st, fs, n_windows):

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
        st0.detrend("linear")
#        st0.taper(0.01)
#        st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)


        ## estimate instantaneous frequency with hilbert
#        signal = st0[0].data*tapers[:, ii]

        st0[0].data *= tapers[:, ii]
        st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)
        signal = st0[0].data

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


    return t_mid, insta_f_cut_mean, amp, std_dev


def __periodogram_estimate(data, fs):

    from spectrum import dpss, pmtm
    from numpy import zeros, arange, linspace, std, argmax, max
    from scipy.signal import find_peaks, peak_widths


    f_tmp, psd_tmp = periodogram(data,
                                 fs=fs,
                                 window='boxcar',
                                 nfft=None,
                                 detrend='constant',
                                 return_onesided=True,
                                 scaling='density',
                                 )


    p_maxima = max(psd_tmp)
    f_maxima = f_tmp[argmax(psd_tmp)]

    f_std = std(f_tmp)

    return f_tmp, f_maxima, p_maxima, f_std



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
    st0.detrend("linear")
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

    return t_mid, insta_f_cut_mean, np.mean(amplitude_envelope), np.std(insta_f_cut)


def __compute(config, st0, starttime, method="hilbert"):

    from scipy.signal import find_peaks, peak_widths, welch, periodogram
    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['t_steps']

    tt1, tt2, ff, hh, pp = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    while n2 <= config['loaded_period']:

#         try:

        ## cut stream to chuncks
        st_tmp = st0.copy().trim(starttime+n1-config['t_overlap']/2, starttime+n1+config['t_steps']+config['t_overlap']/2)

        ## get time series from stream
#        times = st_tmp[0].times(reftime=UTCDateTime("2016-01-01T00"))

        ## get sampling rate from stream
        df = st_tmp[0].stats.sampling_rate


        if method == "hilbert":

            f_tmp, f_max, p_max, h_tmp = __hilbert_frequency_estimator(config, st_tmp, df)

        elif method == "multitaper_hilbert":

            f_tmp, f_max, p_max, h_tmp = __multitaper_hilbert(config, st_tmp, df, n_windows=config['n_windows'])

        elif method == "multitaper_periodogram":

            f_tmp, f_max, p_max, h_tmp = __multitaper_periodogram(st_tmp[0].data, df, n_windows=config['n_windows'])

        elif method == "periodogram":

            f_tmp, f_max, p_max, h_tmp = __periodogram_estimate(st_tmp[0].data, df)

        elif method == "multitaper":

            f_tmp, f_max, p_max, h_tmp = __multitaper_estimate(st_tmp[0].data, df, n_windows=config['n_windows'], one_sided=True)

        times_utc = st_tmp[0].times("utcdatetime")
#        times_mjd = __utc_to_mjd(list(times_utc))

        ## append values to arrays
        tt1[ii] = times_utc[int(len(times_utc)/2)]
        tt2[ii] = __utc_to_mjd(tt1[ii])
        ff[ii] = f_max
        pp[ii] = p_max
        hh[ii] = h_tmp

#         except:
#             tt[ii], ff[ii], pp[ii], hh[ii] = nan, nan, nan, nan
#             print(" -> computing failed")

        ii += 1
        n1 += config['t_steps']
        n2 += config['t_steps']

    return tt1, tt2, ff, hh, pp


## Looping
def main(iii, date):

        idx_count=0
        NNN = int(86400/config['t_steps'])

        t_utc, t_mjd, f, p, h =zeros(NNN), zeros(NNN), zeros(NNN), zeros(NNN), zeros(NNN)

        for hh in tqdm(range(24)):

            ## define current time window
            dh = hh*3600
            t1, t2 = UTCDateTime(date)+dh, UTCDateTime(date)+config['loaded_period']+dh

            try:
                ## load data for current time window
                #                print(" -> loading data ...")
                st, inv = __querrySeismoData(
                                            seed_id=config['seed'],
                                            starttime=t1-2*config['t_overlap'],
                                            endtime=t2+2*config['t_overlap'],
                                            repository=config['repository'],
                                            path=None,
                                            restitute=None,
                                            detail=None,
                )
            except:
                print(" -> failed to load data!")
                continue


            ## compute values
            #       print(" -> computing ...")
            tt_utc, tt_mjd, ff, hh, pp = __compute(config, st, t1, method=config['method'])


            ## combine with previous values
            for mm in range(len(tt_mjd)):
                t_utc[idx_count] = tt_utc[mm]
                t_mjd[idx_count] = tt_mjd[mm]
                f[idx_count] = ff[mm]
                p[idx_count] = pp[mm]
                h[idx_count] = hh[mm]
                idx_count += 1
            try:
                del st, tt_utc, tt_mjd, ff, hh, pp
                gc.collect()
            except:
                pass

        ## create and write a dataframe
        df = DataFrame()
        df['times_utc'] = t_utc
        df['times_mjd'] = t_mjd
        df['freqs'] = f
        df['hmhw'] = h
        df['psd_max'] = p

        date_str = str(date)[:10].replace("-","")
        print(f" -> writing: {config['outpath_data']}FJ{config['ring']}_{date_str}_{config['outfile_appendix']}.pkl")
        df.to_pickle(f"{config['outpath_data']}FJ{config['ring']}_{date_str}_{config['outfile_appendix']}.pkl")



## ________ MAIN  ________
if __name__ == "__main__":

    tbeg = date.fromisoformat(config['tbeg'])
    tend = date.fromisoformat(config['tend'])


    print(json.dumps(config, indent=4, sort_keys=True))

    pool = mp.Pool(mp.cpu_count())


    [pool.apply_async(main, args=(iii, date)) for iii, date in enumerate(date_range(tbeg, tend))]

    pool.close()
    pool.join()


## END OF FILE
