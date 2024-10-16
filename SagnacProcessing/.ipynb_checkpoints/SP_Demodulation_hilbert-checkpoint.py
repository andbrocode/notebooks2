 #!/usr/bin/env python
# coding: utf-8

# ### Demodulation with Sine Fitting

# ### Official Libraries

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import gc
import os

# from scipy.signal import resample, hilbert, correlate
from tqdm import tqdm
from scipy.signal import hilbert
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

# _________________________________________________________________________
# ## Configurations

config = {}

# extract ring
if len(sys.argv) > 1:
    config['ring'] = sys.argv[1]
else:
    config['ring'] = "U"

# specify path to data archive
config['path_to_archive'] = archive_path+"romy_archive/"

# specify path to output files
config['path_to_out_file'] = archive_path+"temp_archive/"

# set if amplitdes are corrected with envelope
config['correct_amplitudes'] = True

# set prewhitening factor (to avoid division by zero)
config['prewhitening'] = 0.1

# set time interval
config['t1'] = UTCDateTime("2024-07-11 00:00")
config['t2'] = config['t1'] + 86400

# V / count  [0.59604645ug  from obsidian]
config['conversion'] = 0.59604645e-6

# define intervals for data loading (in seconds)
config['interval_seconds'] = 600

# define overlap for data loading (in seconds)
config['interval_overlap'] = 0

# interval for fitting (in seconds)
config['Tinterval'] = 600

# overlap for fitting (in seconds)
config['Toverlap'] = 60

config['new_delta'] = config['Tinterval']-config['Toverlap']

# conversion to from hertz to rotation rate
config['rotation_rate'] = True

# specify ring nominal sagnac frequencies
# config['rings'] = {"Z":553.5, "U":302.5, "V":447.5, "W":447.5}
config['rings'] = {"Z":551.677, "U":302.959, "V":448.092, "W":448.092}

# set upsampling to 10 kHz
config['upsampling'] = False


# _________________________________________________________________________

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

def __write_stream_to_sds(st, path_to_sds):

    import os

    # check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

        if not os.path.exists(path_to_sds+f"{yy}/"):
            os.mkdir(path_to_sds+f"{yy}/")
            print(f"creating: {path_to_sds}{yy}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3, "0")

        try:
            st_tmp = st.copy()
            st_tmp = st_tmp.select(network=nn, station=ss, location=ll, channel=cc)
            st_tmp.write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
            print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")
        except:
            print(f" -> failed to write: {cc}")

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

def __hibert_filter2(stt, cut=20, df_out=200):

    '''
    estimating the instantaneous frequency by using the formula of Jo

    sig_in    = input signal
    time_in   = input timeline
    fs        = sampling frequency of digital signal
    '''

    import numpy as np
    from scipy.signal import decimate, detrend, hilbert
    from scipy import fftpack
    from obspy import Stream, Trace

    sig_in = stt[0].data

    df = stt[0].stats.sampling_rate

    # estimate instantaneous frequency with hilbert
    # analytic_signal = hilbert(sig_in)
    analytic_signal = hilbert(sig_in, fftpack.next_fast_len(len(sig_in)))[:len(sig_in)]

    # compute envelope of signal
    # amplitude_envelope = np.abs(analytic_signal)

    # compute unwrapped phase of signal
    insta_phase = np.unwrap(np.angle(analytic_signal))

    # compute instantaneous frequency of signal
    # insta_frequency = np.diff(insta_phase) / (2.0*np.pi) * df
    insta_frequency = np.gradient(insta_phase) / (2.0*np.pi) * df

    # overwrite data of stream
    stt[0].data = insta_frequency

    # downsampling
    if config['upsampling']:
        stt.decimate(2) # 10000 -> 5000
    stt = stt.decimate(5) # 5000 -> 1000
    stt = stt.decimate(5) # 1000 -> 200
    if df_out == 100 or df_out == 20:
        stt = stt.decimate(2) # 200 -> 100
    if df_out == 20:
        stt = stt.decimate(5) # 100 -> 20

    # cut corrupt start and end
    t1 = stt[0].stats.starttime
    t2 = stt[0].stats.endtime
    stt = stt.trim(t1+cut, t2-cut)

    # remove trend
    stt = stt.detrend("linear")

    return stt

def main(config):

    # get intervals for data loading
    times = __get_time_intervals(config['t1'], config['t2'], config['interval_seconds'], config['interval_overlap'])

    # prepare output streams
    stout = Stream()

    for _t1, _t2 in tqdm(times):

        # print(_t1, _t2)

        # load data
        st00 = __read_sds(config['path_to_archive'], f"BW.DROMY..FJ{config['ring']}", _t1-60, _t2+60)

        # convert to volt
        for tr in st00:
            tr.data = tr.data*config['conversion']

            if config['correct_amplitudes']:
                # scale by envelope
                env = abs(hilbert(tr.data)) + config['prewhitening']
                tr.data = tr.data / env

        # upsampling to 10 kHz
        if config['upsampling']:
            st00 = st00.resample(10000)

        # remove trend
        st00 = st00.detrend("linear")

        # apply bandpass
        # st00 = st00.taper(0.01)
        # st00 = st00.filter("bandpass", freqmin=fsagnac-fband, freqmax=fsagnac+fband, corners=4, zerophase=True)

        st = __hibert_filter2(st00, cut=20, df_out=20)

        # cut to interval
        st = st.trim(_t1, _t2, nearest_sample=False)

        stout += st

    # convert frequency to rotation rate (rad/s)
    if config['rotation_rate']:
        f0 = config['rings'][config['ring']]
        omegaE = 2*np.pi/86400
        for _tr in stout:
            _tr.data = ( _tr.data - f0 ) / f0 * omegaE

    # adjust seed code
    for tr in stout:
        tr.stats.network = "BW"
        tr.stats.station = "ROMY"
        if config['upsampling']:
            tr.stats.location = "80"
        else:
            tr.stats.location = "90"
        tr.stats.channel = f"BJ{config['ring']}"

    # phase stream
    stout = stout.merge()
    stout = stout.split()
    # stpout.write(config['path_to_out_file']+f"tmp_phase.mseed")
    __write_stream_to_sds(stout, config['path_to_out_file'])


if __name__ == "__main__":
    main(config)

# End of File