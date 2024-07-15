#!/usr/bin/env python
# coding: utf-8

# ### Demodulation with Sine Fitting

# ### Official Libraries

import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import os

# from scipy.signal import resample, hilbert, correlate
from tqdm import tqdm
from obspy import UTCDateTime, read, Stream


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


# ## Configurations

config = {}

config['ring'] = "U"

config['path_to_archive'] = archive_path+"romy_archive/"

config['path_to_out_file'] = "/home/brotzer/Downloads/tmp/output/"

config['t1'] = UTCDateTime("2024-07-11 15:00")
# config['t2'] = UTCDateTime("2024-07-11 16:00")
config['t2'] = config['t1'] +86400

# define intervals for data loading (in seconds)
config['interval_seconds'] = 600

# define overlap for data loading (in seconds)
config['interval_overlap'] = 0

# interval for fitting (in seconds)
config['Tinterval'] = 1

# overlap for fitting (in seconds)
config['Toverlap'] = 0.95

config['new_delta'] = config['Tinterval']-config['Toverlap']


def __sine_fit_stream(st_in, seed, Tinterval=1, Toverlap=0.8, plot=True):

    '''
    Fitting a sin-function to the data to estimate an instantaneous frequency
    '''

    import matplotlib.pyplot as plt

    from obspy import Trace, Stream
    from scipy import optimize
    from numpy import sin, hanning, pi, arange, array, diag, zeros, nan, isnan, isinf, pi, inf

    # def func(x, a, f):
    #     return a * sin(2 * pi * f * x)

    def func(x, a, f, p):
        return a * sin(2 * pi * f * x + p)

    # ring nominal sagnac frequencies
    rings = {"Z":553.5, "U":302.5, "V":447.5, "W":447.5}

    # codes
    net, sta, loc, cha = seed.split('.')

    # to array
    data = st_in[0].data
    times = st_in[0].times()
    starttime = st_in[0].stats.starttime

    # sampling rate
    df = st_in[0].stats.sampling_rate

    # npts per interval
    Nsamples = int(Tinterval*df)
    Noverlap = int(Toverlap*df)

    # npts in data
    Ndata = data.size

    # create time reference
    tt = times

    # amount of windows
    Nwin = 0
    x2 = Nsamples
    while x2 < Ndata:
        x2 = x2 + Nsamples - Noverlap
        Nwin += 1

    # print(Nwin, Ndata, Nsamples, Noverlap)

    # prepare arrays
    amps = zeros(Nwin)*nan
    freq = zeros(Nwin)*nan
    phas = zeros(Nwin)*nan
    time = zeros(Nwin)*nan
    cfs = zeros(Nwin)*nan
    cas = zeros(Nwin)*nan

    # initial values
    a00 = 0.9
    f00 = rings[cha[-1]]
    p00 = 0

    # specify start indices
    # n1, n2 = 0, int(Nsamples - Noverlap)
    n1, n2 = 0, Nsamples

    # fail counter
    fails = 0

    # looping
    for _win in range(Nwin):

        # set start values at begin
        if _win == 0:
            a0, f0, p0 = a00, f00, p00

        # cut data for interval
        _time = tt[n1:n2]
        _data = data[n1:n2]

        # slightly change start values using round
        # a0, f0, p0 = round(a0, 1), round(f0, 1), round(p0, 1)

        # reset start values if nan
        if isnan(a0) or isnan(f0) or isnan(p0):
            a0, f0, p0 = a00, f00, p00

        if isinf(a0) or isinf(f0) or isinf(p0):
            a0, f0, p0 = a00, f00, p00

        # a0, f0, p0 = a00, f00, p00

        # fit sine to data
        try:
            params, params_covariance = optimize.curve_fit(func,
                                                           _time,
                                                           _data,
                                                           p0=[a0, f0, p0],
                                                           check_finite=True,
                                                           # bounds=([-2, f00-1, -inf],[2, f00+1, inf]),
                                                           # method="trf",
                                                          )
            a0 = params[0]
            f0 = params[1]
            p0 = params[2]

            ca, cf = diag(params_covariance)[0], diag(params_covariance)[1]

        except Exception as e:
            # print(e)
            fails += 1
            # f0, a0, p0, cf, ca = nan, nan, nan, nan, nan

        if cf > 0.001 or ca > 0.001:
            f0, a0, p0 = nan, nan, nan

        # append values
        amps[_win] = a0
        freq[_win] = f0
        phas[_win] = p0
        time[_win] = (tt[n2]-tt[n1])/2 + tt[n1]

        cfs[_win] = cf
        cas[_win] = ca

        # checkup plot for fit
        if plot:
            if _win == Nwin - 1:
                print(f0, a0, p0, cf, ca)
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))

                ax.plot(_time, _data, color='black')

                ax.plot(_time, func(_time, params[0], params[1], params[2]), color='red')

                plt.show();

        # update index
        n1 = n1 + Nsamples - Noverlap
        n2 = n2 + Nsamples - Noverlap

    # checkup plot
    if plot:

        Nrow, Ncol = 3, 1

        font = 12

        fig, ax = plt.subplots(Nrow, Ncol, figsize=(12, 5), sharex=True)

        plt.subplots_adjust(hspace=0.1)

        ax[0].errorbar(time, freq, cfs)
        ax[1].errorbar(time, amps, cas)
        ax[2].plot(time, phas)

        plt.show();

    def streamout(dat, ll):
        tr_out = Trace()
        tr_out.data = dat
        tr_out.stats.network = net
        tr_out.stats.station = sta
        tr_out.stats.location = ll
        tr_out.stats.channel = cha
        tr_out.stats.starttime = starttime + (Tinterval-Toverlap)
        tr_out.stats.delta = (Tinterval-Toverlap)
        return Stream(tr_out)

    print(f" -> fails: {fails}")

    st_out_f = streamout(freq, "60")
    st_out_p = streamout(phas, "70")

    return st_out_f, st_out_p

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

def __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED"):

    '''
    VARIABLES:
     - path_to_archive
     - seed
     - tbeg, tend
     - data_format

    DEPENDENCIES:
     - from obspy.core import UTCDateTime
     - from obspy.clients.filesystem.sds import Client

    OUTPUT:
     - stream

    EXAMPLE:
    >>> st = __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED")

    '''

    import os
    from obspy.core import UTCDateTime, Stream
    from obspy.clients.filesystem.sds import Client

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if not os.path.exists(path_to_archive):
        print(f" -> {path_to_archive} does not exist!")
        return

    ## separate seed id
    net, sta, loc, cha = seed.split(".")

    ## define SDS client
    client = Client(path_to_archive, sds_type='D', format=data_format)

    ## read waveforms
    try:
        st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
    except:
        print(f" -> failed to obtain waveforms!")
        st = Stream()

    return st

def main(config):

    # get intervals for data loading
    times = __get_time_intervals(config['t1'], config['t2'], config['interval_seconds'], config['interval_overlap'])

    # prepare output streams
    stfout = Stream()
    stpout = Stream()


    for _t1, _t2 in times:

        print(_t1, _t2)

        # load data
        st00 = __read_sds(config['path_to_archive'], f"BW.DROMY..FJ{config['ring']", _t1, _t2)

        # convert to volt
        for tr in st00:
            tr.data = tr.data*0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

        # remove trend
        st00 = st00.detrend("linear")
        # st00 = st00.taper(0.01)
        # st00 = st00.filter("bandpass", freqmin=fsagnac-fband, freqmax=fsagnac+fband, corners=4, zerophase=True)

        stf, stp = __sine_fit_stream(st00, f"BW.ROMY..BJ{config['ring']}",
                                     Tinterval=config['Tinterval'],
                                     Toverlap=config['Toverlap'],
                                     plot=False
                                    )

        stfout += stf
        stpout += stp


    # phase stream
    stpout = stpout.merge()
    stpout.write(config['path_to_out_file']+f"tmp_phase.mseed")

    # frequency stream
    stfout = stfout.merge()
    stfout.write(config['path_to_out_file']+f"tmp_frequency.mseed")


if __name__ == "__main__":
    main(config)

# End of File