"""
 Run Backscatter correction
 
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, date
from pandas import DataFrame, read_pickle, date_range, concat, read_csv
from obspy import UTCDateTime, read
from scipy.signal import hilbert


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'

## ______________________________________
## Configurations

config = {}

config['ring'] = "U"

config['seeds'] = ["BW.DROMY..FJU", "BW.DROMY..F1V", "BW.DROMY..F2V"]

config['interval'] = 60

config['tbeg'] = UTCDateTime("2023-09-19 00:00")
config['tend'] = UTCDateTime("2023-09-19 01:00")


## path to Sagnac data
config['path_to_autodata'] = archive_path+f"romy_autodata/"

config['path_to_data'] = data_path+"sagnac_frequency/data/"

config['path_to_figs'] = data_path+"sagnac_frequency/figures/"

config['path_to_sds'] = archive_path+"romy_archive/"


## ______________________________________
## methods

def __load_romy_raw_data(seed, tbeg, tend, path_to_sds):

    from andbro__read_sds import __read_sds
    from obspy import Stream, UTCDateTime


    print(f" -> loading {seed}...")

    try:
        st00 = __read_sds(path_to_sds, seed, tbeg,tend, data_format='MSEED')
    except:
        print(f" -> failed for {seed}")

    st0 = st00.sort()

    for tr in st0:
        tr.data = tr.data*0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

    return st0

def __get_values(ff, psd, fph, ph, f_sagn):

    from numpy import argmax, sqrt, where, argmin, gradient, mean

    ## specify f-band around Sagnac frequency
    fl = f_sagn-2
    fu = f_sagn+2

    ## get index of Sagnac peak
    idx_fs = where(psd == max(psd[(ff > fl) & (ff < fu)]))[0][0]

    ## estimate Sagnac frequency
    f_sagn_est = ff[idx_fs]

    ## estimate AC value at Sagnac peak
    AC_est = psd[idx_fs]

    ## estimate DC value at ff = 0
    DC_est = psd[0]

    return f_sagn_est, AC_est, DC_est


def __get_fft(signal_in, dt, window=None):

    from scipy.fft import fft, fftfreq, fftshift
    from scipy import signal
    from numpy import angle

    ## determine length of the input time series
    n = int(len(signal_in))


    ## calculate spectrum (with or without window function applied to time series)
    if window:
        win = signal.get_window(window, n);
        spectrum = fft(signal_in * win)

    else:
        spectrum = fft(signal_in)

    ## calculate frequency array 
    frequencies = fftfreq(n, d=dt)


    ## correct amplitudes of spectrum
    magnitude = 2.0 / n * abs(spectrum)

    ## phase angle
    phase = angle(spectrum, deg=False)

    return frequencies[0:n//2], magnitude[0:n//2], phase[0:n//2]


def __get_time_intervals(tbeg, tend, interval_seconds, interval_overlap):

    from obspy import UTCDateTime

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    times = []
    t1, t2 = tbeg, tbeg + interval_seconds
    while t2 <= tend:
        times.append((t1, t2))
        t1 = t1 + interval_seconds - interval_overlap
        t2 = t2 + interval_seconds - interval_overlap

    return times


def main(config):

    ## load data
    sagn = __load_romy_raw_data(config['seeds'][0], config['tbeg'], config['tend'], config['path_to_sds'])
    mon1 = __load_romy_raw_data(config['seeds'][1], config['tbeg'], config['tend'], config['path_to_sds'])
    mon2 = __load_romy_raw_data(config['seeds'][2], config['tbeg'], config['tend'], config['path_to_sds'])

    ## get time intervals for iteration
    times = __get_time_intervals(config['tbeg'], config['tend'], interval_seconds=config['interval'], interval_overlap=0)

    ## prepare output arrays
    fs, ac, dc, ph = np.ones(len(times))*np.nan, np.ones(len(times))*np.nan, np.ones(len(times))*np.nan, np.ones(len(times))*np.nan

    ## prepare output dataframe
    out_df = DataFrame()
    out_df['time1'] = list(zip(*times))[0]
    out_df['time2'] = list(zip(*times))[1]


    for _k, _st in enumerate([sagn, mon1, mon2]):

        print(_k, "...")

        for _n, (t1, t2) in enumerate(times):

            _dat = _st.copy().trim(t1, t2)

            f, psd, pha = __get_fft(_dat[0].data, _dat[0].stats.delta, window=None)

            fs[_n], ac[_n], dc[_n], ph[_n] = __get_values(f, psd, f, pha, 303)

            # dc[_n] = np.mean(_dat)
            # ac[_n] = np.percentile(_dat[0].data, 99.9) - np.percentile(_dat[0].data, 100-99.9)

        ph = np.unwrap(ph)

        if _k == 0:
            out_df['fj_fs'], out_df['fj_ac'], out_df['fj_dc'], out_df['fj_ph'] = fs, ac, dc, ph
        elif _k == 1:
            out_df['f1_fs'], out_df['f1_ac'], out_df['f1_dc'], out_df['f1_ph'] = fs, ac, dc, ph
        elif _k == 2:
            out_df['f2_fs'], out_df['f2_ac'], out_df['f2_dc'], out_df['f2_ph'] = fs, ac, dc, ph

    ## store data
    date_str = f"{config['tbeg'].year}{str(config['tbeg'].month).rjust(2,'0')}{str(config['tbeg'].day).rjust(2,'0')}"
    out_df.to_pickle(config['path_to_data']+f"{date_str}_{method}.pkl")


## ________ MAIN  ________
if __name__ == "__main__":

    main(config)

## End of File