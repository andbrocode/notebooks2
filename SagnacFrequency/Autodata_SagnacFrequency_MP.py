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
# config['seed'] = f"BW.DROMY..FJ{config['ring']}"
config['seeds'] = ["BW.DROMY..FJZ", "BW.DROMY..F1V", "BW.DROMY..F4V"]

config['tbeg'] = "2023-05-08"
config['tend'] = "2023-05-08"

config['outpath_data'] = f"/import/kilauea-data/sagnac_frequency/autodata/"

config['outfile_appendix'] = ""

config['repository'] = "archive"

config['method'] = "hilbert" ## "hilbert" | "multitaper_hilbert" | "welch" | "periodogram" | multitaper | multitaper_periodogram

rings = {"Z":553, "U":302, "V":448,"W":448}

config['f_expected'] = rings[config['ring']]  ## expected sagnac frequency
config['f_band'] = 3 ## +- frequency band

#config['n_windows'] = 10

config['t_steps'] = 60  ## seconds  (-> time delta)
config['t_overlap'] = 180 ## seconds


config['loaded_period'] = 3600  ## seconds
config['NN'] = int(config['loaded_period']/config['t_steps'])

#config['nblock'] = 300*5000
#config['noverlap'] = None




## Methods

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


def __compute_ac_dc_contrast(config, st):

    from numpy import percentile, nanmean

    st0 = st.copy()
    dat = st0[0].data*0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

    percentiles = percentile(dat, [2.5, 97.5])
    ac = percentiles[1]-percentiles[0]
    dc = nanmean(dat)

    # contrast(max(dat)-min(dat))/(max(dat)+min(dat))
    con = (percentiles[1]-percentiles[0])/(percentiles[1]+percentiles[0])


    return ac, dc, con


def __compute(config, st0, starttime, method="hilbert"):

    from scipy.signal import find_peaks, peak_widths, welch, periodogram
    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['t_steps']

    tt1, tt2, ff, hh, pp = zeros(NN), zeros(NN), zeros(NN), zeros(NN), zeros(NN)
    ac, dc, con = zeros(NN), zeros(NN), zeros(NN)


    while n2 <= config['loaded_period']:

        try:

            ## cut stream to chuncks
            st_tmp = st0.copy().trim(starttime+n1-config['t_overlap']/2, starttime+n1+config['t_steps']+config['t_overlap']/2)

            ## get time series from stream
            # times = st_tmp[0].times(reftime=UTCDateTime("2016-01-01T00"))

            ## get sampling rate from stream
            df = st_tmp[0].stats.sampling_rate


            if method == "hilbert":

                f_tmp, f_max, p_max, h_tmp = __hilbert_frequency_estimator(config, st_tmp, df)

            else:
                print(" -> unkown method")
                continue

            times_utc = st_tmp[0].times("utcdatetime")

            ## copmute AC, DC and Contrast
            ac0, dc0, con0 = __compute_ac_dc_contrast(config, st_tmp)


            ## append values to arrays
            tt1[ii] = times_utc[int(len(times_utc)/2)]
            tt2[ii] = __utc_to_mjd(tt1[ii])
            ff[ii] = f_max
            pp[ii] = p_max
            hh[ii] = h_tmp
            ac[ii] = ac0
            dc[ii] = dc0
            con[ii] = con0


        except:
            tt[ii], ff[ii], pp[ii], hh[ii] = nan, nan, nan, nan
            print(" -> computing failed")

        ii += 1
        n1 += config['t_steps']
        n2 += config['t_steps']

    return tt1, tt2, ff, hh, pp, ac, dc, con


## _________________________________________________
## Looping in Main

def main(iii, date):

        idx_count=0

        ## amount of samples
        NNN = int(86400/config['t_steps'])

        ## prepare empty arrays
        t_utc, t_mjd = zeros(NNN), zeros(NNN)
        fz, f1, f2 = zeros(NNN), zeros(NNN), zeros(NNN)
        pz, p1, p2 = zeros(NNN), zeros(NNN), zeros(NNN)


        for hh in tqdm(range(24)):

            ## define current time window
            dh = hh*3600
            t1, t2 = UTCDateTime(date)+dh, UTCDateTime(date)+config['loaded_period']+dh

            ## loop for sagnac and monobeams
            for seed in config['seeds']:
                try:
                    ## load data for current time window
                    #                print(" -> loading data ...")
                    st, inv = __querrySeismoData(
                                                seed_id=seed,
                                                starttime=t1-2*config['t_overlap'],
                                                endtime=t2+2*config['t_overlap'],
                                                repository=config['repository'],
                                                path=None,
                                                restitute=None,
                                                detail=None,
                                                )
                    ## convert from V to count  [0.59604645ug  from obsidian]
                    st[0].data = st[0].data*0.59604645e-6

                except:
                    print(f" -> failed to load data for {seed}!")
                    continue


                ## compute values
                #       print(" -> computing ...")
                tt_utc, tt_mjd, ff, hh, pp, ac, dc, con = __compute(config, st, t1, method=config['method'])

                if seed.split(".")[3] == "FJZ":
                    fz, pz, ac_z, dc_z, con_z = ff, pp, ac, dc, con
                elif seed.split(".")[3] == "F1V":
                    f1, p1 = ff, pp
                elif seed.split(".")[3] == "F4V":
                    f2, p2 = ff, pp


            ## combine with previous values
            for mm in range(len(tt_mjd)):
                t_utc[idx_count] = tt_utc[mm]
                t_mjd[idx_count] = tt_mjd[mm]

                fz[idx_count] = fz[mm]
                f1[idx_count] = f1[mm]
                f2[idx_count] = f2[mm]

                pz[idx_count] = pz[mm]
                p1[idx_count] = p1[mm]
                p2[idx_count] = p2[mm]

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
        df['fz'], df['f1'], df['f2'] = fz, f1, f2
        df['pz'], df['p1'], df['p2'] = pz, p1, p2
        df['ac_z'] = ac
        df['dc_z'] = dc
        df['contrast_z'] = con

        date_str = str(date)[:10].replace("-","")
        print(f" -> writing: {config['outpath_data']}FJ{config['ring']}_{date_str}{config['outfile_appendix']}.pkl")
        df.to_pickle(f"{config['outpath_data']}FJ{config['ring']}_{date_str}{config['outfile_appendix']}.pkl")



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
