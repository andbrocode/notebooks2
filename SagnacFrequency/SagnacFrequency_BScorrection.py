"""
 Run Backscatter correction

"""

import os, sys
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
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename == 'ambrym':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'

## ______________________________________
## Configurations

config = {}

config['ring'] = "U"

config['seeds'] = ["BW.DROMY..FJU", "BW.DROMY..F1V", "BW.DROMY..F2V"]

config['interval'] = 60

if len(sys.argv) > 2:
    config['tbeg'] = sys.argv[2]
    config['tbeg'] = config['tbeg']+86400
else:
    config['tbeg'] = UTCDateTime("2023-12-01 00:00")
    config['tend'] = UTCDateTime("2023-12-04 00:00")

config['ring_sagnac'] = {"U":303.05, "V":447, "W":447, "Z":553.5}
config['nominal_sagnac'] = config['ring_sagnac'][config['ring']]

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

def __get_values(ff, psd, pha, f_sagn):

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

    ## phase at Sagnac frequency
    ph_est = pha[idx_fs]

    return f_sagn_est, AC_est, DC_est, ph_est


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


def __hilbert_frequency_estimator(st, nominal_sagnac, fband):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()

    ## extract sampling rate
    df = st0[0].stats.sampling_rate

    ## define frequency band around Sagnac Frequency
    f_lower = nominal_sagnac - fband
    f_upper = nominal_sagnac + fband

    ## bandpass with butterworth around Sagnac Frequency
    st0.detrend("linear")
    st0.taper(0.01)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)


    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * df)

    ## cut first and last 5% (corrupted data)
    dd = int(0.05*len(instantaneous_frequency))
    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## get times
    t = st0[0].times()
    t_mid = t[int((len(t))/2)]

    ## averaging of frequencies
    # insta_f_cut_mean = np.mean(insta_f_cut)
    insta_f_cut_mean = np.median(insta_f_cut)

    return t_mid, insta_f_cut_mean, np.mean(amplitude_envelope), np.std(insta_f_cut)


def __backscatter_correction(m01, m02, phase0, w_obs, fs0, cm_filter_factor=1.033):

    ## Correct for bias
    m1 = m01 * ( 1 + m01**2 / 4 )
    m2 = m02 * ( 1 + m02**2 / 4 )

    ## angular correction for phase
    phase = phase0 + 0.5 * m1 * m2 * np.sin( phase0 )

    ## compute squares of common-mode modulations
    m2c = ( m1**2 + m2**2 + 2*m1*m2*np.cos( phase ) ) / 4

    ## compute squares of differential-mode modulations
    m2d = ( m1**2 + m2**2 - 2*m1*m2*np.cos( phase ) ) / 4  ## different angle!

    ## correct m2c for gain saturation of a HeNe laser
    # m2c = m2c * ( 1 + ( beta + theta )**2 * fL**2 * I0**2 / ws**2 )
    m2c = m2c * cm_filter_factor

    ## compute backscatter correction factor
    M = m2c - m2d + 0.25 * m1**2 * m2**2 * np.sin(phase)**2

    ## correction term
    term = ( 4 + M ) / ( 4 - M )

    ## backscatter correction
    correction = -1 * ( term -1 ) * fs0

    w_corrected = np.array(w_obs) + correction

    return w_corrected

## _________________________________

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

        print(" -> processing ", _k, "...")

        for _n, (t1, t2) in enumerate(times):

            _dat = _st.copy().trim(t1, t2)

            f, psd, pha = __get_fft(_dat[0].data, _dat[0].stats.delta, window=None)

            fs[_n], ac[_n], dc[_n], ph[_n] = __get_values(f, psd, pha, config['nominal_sagnac'])

            t, fs[_n], _, _ = __hilbert_frequency_estimator(_dat, nominal_sagnac=config['nominal_sagnac'], fband=10)

            # dc[_n] = np.mean(_dat)
            # ac[_n] = np.percentile(_dat[0].data, 99.9) - np.percentile(_dat[0].data, 100-99.9)

        ph = np.unwrap(ph)

        ## fill output dataframe
        if _k == 0:
            out_df['fj_fs'], out_df['fj_ac'], out_df['fj_dc'], out_df['fj_ph'] = fs, ac, dc, ph
        elif _k == 1:
            out_df['f1_fs'], out_df['f1_ac'], out_df['f1_dc'], out_df['f1_ph'] = fs, ac, dc, ph
        elif _k == 2:
            out_df['f2_fs'], out_df['f2_ac'], out_df['f2_dc'], out_df['f2_ph'] = fs, ac, dc, ph


    ## prepare values for backscatter correction
    m01 = out_df.f1_ac / out_df.f1_dc
    m02 = out_df.f2_ac / out_df.f2_dc
    phase0 = out_df.f1_ph - out_df.f2_ph
    w_obs = out_df.fj_fs

    out_df['w_s'] = __backscatter_correction(m01, m02, phase0, w_obs, config['nominal_sagnac'], cm_filter_factor=1.033)


    ## store data
    date_str = f"{config['tbeg'].year}{str(config['tbeg'].month).rjust(2,'0')}{str(config['tbeg'].day).rjust(2,'0')}"
    out_df.to_pickle(config['path_to_data']+f"{date_str}_backscatter.pkl")
    print(f" -> writing: {config['path_to_data']}{date_str}_backscatter.pkl")


## ________ MAIN  ________
if __name__ == "__main__":

    main(config)

## End of File
