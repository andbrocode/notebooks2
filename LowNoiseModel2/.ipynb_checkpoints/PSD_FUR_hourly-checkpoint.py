# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__   = '2022'

# In[] ___________________________________________________________
'''---- import libraries ----'''

import matplotlib.pyplot as plt
import pickle, sys
import os


from obspy import UTCDateTime, read
from scipy.signal import welch
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram, arange
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path
from obspy.clients.fdsn import Client

from andbro__read_sds import __read_sds

import warnings
warnings.filterwarnings('ignore')


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'


# In[] ___________________________________________________________
''' ---- set variables ---- '''

config = {}


config['array'] = "FUR"

config['year'] = 2023

config['component'] = "N" # U,V,W,Z

config['cha'] = config['component']

config['date1'] = UTCDateTime(f"{config['year']}-09-23")
config['date2'] = UTCDateTime(f"{config['year']}-10-22")

config['seed'] = "GR.FUR..HHN"

config['station'] = config['seed'].split(".")[1]
config['channel'] = config['seed'].split(".")[3]

config['repository'] = "archive"

# config['type'] = "rot"
config['sampling_rate'] = 20.0
config['tseconds'] = 1800  ## seconds

config['interval'] = 3600

config['interval_overlap'] = None # in percent

config['taper'] = 'hann'
config['nperseg'] = int(config.get('tseconds')*config.get('sampling_rate'))
config['noverlap'] = int(0.5*config.get('tseconds'))
config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0.001,10) # in Hz

# config['dB']= False

## define path and name for output
config['opath'] = data_path+f"LNM2/PSDS/{config['station']}/"

config['oname'] = f"{config['year']}_{config['station']}-{config['channel'][2]}_{config['interval']}"



# In[] ___________________________________________________________
'''---- define methods ----'''


def __request_data(seed, tbeg, tend):

    from obspy.clients.fdsn import Client

    client = Client("BGR")

    net, sta, loc, cha = seed.split(".")

    try:
        inventory = client.get_stations(network=net,
                                         station=sta,
                                         starttime=tbeg,
                                         endtime=tend,
                                         level="response",
                                         )
    except:
        print(" -> Failed to load inventory!")


    try:
        waveform = client.get_waveforms(network=net,
                                        station=sta,
                                        location=loc,
                                        channel=cha,
                                        starttime=tbeg-60,
                                        endtime=tend+60,
                                       )

    except:
        print(" -> Failed to load waveforms!")
        return None, None

    return waveform, inventory


def __calculate_spectra(st, config, idx_count, mode='dB'):

    from datetime import datetime
    from pandas import date_range
    from obspy import UTCDateTime
    from scipy.signal import welch
    from numpy import where, array, zeros

    def __check_stream(st):
        t1 = str(st[0].stats.starttime)
        t2 = str(st[0].stats.endtime)
        for tr in st:
            if str(tr.stats.starttime) != t1 or str(tr.stats.endtime) != t2:
                print(f"ERROR: mismatch in start or endtime of trace: {tr.stats.id}")
                return

    # def __make_decibel(array, relative_value):
    #     return 10*log10(array/relative_value)

    ## check time consistency for all traces
    __check_stream(st)

    ## check how many intervals are possible
    if config['interval_overlap'] is None:
        intervals = int((st[0].stats.endtime - st[0].stats.starttime+10)/config.get('interval'))
        shift = config['interval']
    else:
        shift = int(config.get('interval')*config['interval_overlap']/100)
        intervals = int((st[0].stats.endtime - st[0].stats.starttime)/shift)



    ## pre-define psd array
    size_psd = int(config.get('nperseg')/2)+1
    psd = zeros([intervals, size_psd])

    if size_psd >= len(st[0].data):
        print(f"ERROR: reduce nperseg or noverlap or segments! {size_psd} > {len(st[0].data)}")
        return

    for i, tr in enumerate(st):

        # initite variables for while loop
        dt1 = st[0].stats.starttime
        dt2 = st[0].stats.starttime + config['interval']
        n = 0

        tt_tmp = []

        while dt2 <= st[0].stats.endtime+10:

            tr_tmp = tr.copy()

            try:
                tr_tmp.trim(UTCDateTime(dt1), UTCDateTime(dt2))
            except:
                print(" -> missing hour")

            try:
                f, psd0 = welch(
                                tr_tmp.data,
                                fs=tr_tmp.stats.sampling_rate,
                                window=config.get('taper'),
                                nperseg=config.get('nperseg'),
                                noverlap=config.get('noverlap'),
                                nfft=config.get('nfft'),
                                detrend=config.get('detrend'),
                                return_onesided=config.get('onesided'),
                                scaling=config.get('scaling'),
                               )
            except Exception as e:
                print(" -> welch failed")
                print(e)

            psd[n] = psd0
            tt_tmp.append(tt_nominal[idx_count])



            ## adjust variables
            dt1 += shift
            dt2 += shift
            n += 1
            idx_count += 1


        if config.get('frequency_limits') is not None:
            f1, f2 = config.get('frequency_limits')[0], config.get('frequency_limits')[1]
            idx1, idx2 = int(where(f <= f1)[0][0]), int(where(f >= f2)[0][0])
            ff = f[idx1:idx2]
            tmp = zeros([intervals, len(ff)])
            for j in range(intervals):
                tmp[j] = psd[j, idx1:idx2]
            psd = tmp
        else:
            ff=f

        # if mode is not None and mode.lower() == 'db':
        #     for j in range(intervals):
        #         psd[j] = __make_decibel(psd[j], abs(max(psd[j])))

    return ff, psd, tt_tmp, idx_count


def __write_to_csv(data, text, config):

    import csv

    opath = config['opath']
    oname = config['outname']+"_"+text+"_psd.csv"

    # open the file in the write mode
    with open(opath+oname, 'w') as file:

        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

    if Path(opath+oname).exists():
        print(f" -> created: {opath}{oname}")


def __save_to_pickle(object, name):

    ofile = open(config['outpath']+config['outname']+name+".pkl", 'wb')
    pickle.dump(object, ofile)

    if Path(config['outpath']+config['outname']+name+".pkl").exists():
        print(f"\n -> created: {config['outpath']}{config['outname']}{name}.pkl")


def __get_time_intervals(tbeg, tend, interval_seconds, interval_overlap):

    times = []
    t1, t2 = tbeg, tbeg + interval_seconds
    while t2 <= tend:
        times.append((t1, t2))
        t1 = t1 + interval_seconds - interval_overlap
        t2 = t2 + interval_seconds - interval_overlap
    return times


# In[] ___________________________________________________________

def main(config):

    days = int((config['date2'] - config['date1'])/86400)+1


    ## check if directory exists
    config['outname'] = f"{config['year']}_{config['station']}_{config['cha']}_{config['interval']}"
    # config['subdir'] = f"{config['station']}_{config['year']}_{config['cha']}/"
    # config['outpath'] = f"{config['opath']}{config['subdir']}"
    config['outpath'] = f"{config['opath']}"

    if not Path(config['opath']).exists():
        Path(config['opath']).mkdir()
        print(f" -> created {config['opath']}")

    if not Path(config['outpath']).exists():
        Path(config['outpath']).mkdir()
        print(f" -> created {config['outpath']}")


    minimum_collection = []
    minimal_collection = []
    columns = []
    medians, dd = [], []

    tt = []

    global tt_nominal
    tt_nominal = arange(0, int(86400/config['interval']*365), 1)

    global idx_count
    idx_count = 0



    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {config['array']}  {config['seed']}  {str(date)[:10]} ...")

        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400 + 10

        ## load data to stream
        st, inventory = __request_data(config['seed'], config['tbeg'], config['tend'])

        if st is None or len(st) == 0:
            print(" -> no data")
            continue

        try:
            ## merge traces that might be split due to data gaps or overlaps
            st.merge(fill_value="interpolate")

            ## remove sensitivity / response
            # st.remove_sensitivity(inventory)
            st.remove_response(inventory, output="ACC", water_level=10)

            ## resampling
            st.resample(config['sampling_rate'])

            ## cut to event
            st.trim(config['tbeg'], config['tend'])

            ## remove mean
            st.detrend("demean")

        except:
            print(" -> Failed to process waveform data!")
            continue


        if st is None or len(st) == 0 or st[0].stats.npts < 1000:
            print(f" -> skipping {date.isoformat()[:10]}!")
            continue

        ## compute spectra for each interval
        ff, psds, tt_tmp, idx_count = __calculate_spectra(st, config, idx_count, mode=None)

        ## append hours for which psd has been computed to times array tt
        [tt.append(t) for t in tt_tmp]


        ## save hourly spectra in daily files
        __save_to_pickle(psds, f"_{str(date).split(' ')[0].replace('-','')}_hourly")

        dd.append(str(date).split(" ")[0].replace("-", ""))

        del st



    ## save data to files
    __save_to_pickle(config, "_config")
    __save_to_pickle(ff, "_frequency_axis")
    __save_to_pickle(tt, "_times_axis")


    print("\n -> Done\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)

# End of File


