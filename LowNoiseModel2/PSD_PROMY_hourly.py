# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__   = '2022'

# In[] ___________________________________________________________
'''---- import libraries ----'''

import matplotlib.pyplot as plt
import pickle

from obspy import UTCDateTime, read
from scipy.signal import welch
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path

from andbro__querrySeismoData import __querrySeismoData
from andbro__load_FURT_stream import __load_furt_stream
from andbro__read_sds import __read_sds

import warnings
warnings.filterwarnings('ignore')

# In[] ___________________________________________________________
''' ---- set variables ---- '''

config= {}


config['array'] = "PROMY"

config['year'] = 2023

##  I = absolute pressure
config['component'] = "I"
config['name_appendix'] = ""

# config['date1'] = UTCDateTime(f"{config['year']}-09-23")
# config['date2'] = UTCDateTime(f"{config['year']}-10-23")

config['date1'] = UTCDateTime(f"{config['year']}-11-11")
config['date2'] = UTCDateTime(f"{config['year']}-11-12")

config['seed'] = f"BW.PROMY..LD{config['component']}"

config['ring'] = config['seed'].split(".")[1]

config['path_to_data'] = f"/import/freenas-ffb-01-data/romy_archive/"

## specify unit
config['unit'] = "Pa" ## hPa or Pa

config['type'] = "baro"

config['interval'] = 3600
config['interval_overlap'] = None # in percent
config['taper'] = 'hanning'

config['tseconds'] = 1800 ## seconds

config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0, 0.05) # in Hz

config['dB']= False

config['outname'] = f"{config['year']}_{config['array']}_{config['interval']}"

config['outpath'] = f"/import/kilauea-data/LNM2/PSDS/{config['array']}{config['name_appendix']}/"


# In[] ___________________________________________________________
'''---- define methods ----'''


def __get_data(config):
    '''
    load data and remove response

    VARIABLES:
    '''
    from andbro__read_sds import __read_sds


    net, sta, loc, cha = config['seed'].split(".")

    st0 = __read_sds(config['path_to_data'], f"{net}.{sta}.{loc}.{cha}", config['tbeg'], config['tend'])

    return st0


def __get_minimum_psd(psds):
    for i, psd in enumerate(psds):
        if i == 0:
            lowest_sum = psds[0].sum()
            idx = 0

        value = psd.sum()

        if value < lowest_sum and value != 0:
            lowest_sum = value
            idx = i

    return psds[idx]


def __calculate_spectra(st, config, mode='dB'):

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
                print(f"ERROR: mismatch in start or endtime of trace: {tr.stats.station}")
                return


    ## check time consistency for all traces
    __check_stream(st)

    ## check how many intervals are possible
    if config['interval_overlap'] is None:
        intervals = int((st[0].stats.endtime - st[0].stats.starttime)/config.get('interval'))
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

        while dt2 <= st[0].stats.endtime:

            tr_tmp = tr.copy()
            tr_tmp.trim(starttime = UTCDateTime(dt1), endtime=UTCDateTime(dt2))

#             print(n, dt1, dt2, "\n")

#             print(config.get('nperseg'), config.get('noverlap'), len(tr_tmp.data))

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
            psd[n] = psd0

            ## adjust variables
            dt1 += shift
            dt2 += shift
            n += 1


        if config.get('frequency_limits') is not None:
            f1, f2 = config.get('frequency_limits')[0], config.get('frequency_limits')[1]
            idx1, idx2 = int(where(f <= f1)[0][0]), int(where(f >= f2)[0][0])
            ff = f[idx1:idx2]
            tmp = zeros([intervals, len(ff)])
            for j in range(intervals):
                tmp[j] = psd[j,idx1:idx2]
            psd = tmp
        else:
            ff=f


    return ff, psd


def __write_to_csv(data, text, config):

    import csv

    opath = config['outpath']
    oname = config['outname']+"_"+text+"_psd.csv"

    # open the file in the write mode
    with open(opath+oname, 'w') as file:

        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

    if Path(opath+oname).exists():
        print(f"created: {opath}{oname}")


def __get_minimal_psd(psds):

    from numpy import nanmin, array, nonzero

    min_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        min_psd[f] = nanmin(a[nonzero(a)])

    return min_psd


def __get_median_psd(psds):

    from numpy import median, zeros, isnan

    med_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        med_psd[f] = median(a[~isnan(a)])

    return med_psd


def __save_to_pickle(object, name):

    ofile = open(config['outpath']+config['outname']+name+".pkl", 'wb')
    pickle.dump(object, ofile)

    if Path(config['outpath']+config['outname']+name+".pkl").exists():
        print(f"\n -> created: {config['outpath']}{config['outname']}{name}.pkl")




# In[] ___________________________________________________________

def main(config):

    days = int((config['date2'] - config['date1'])/86400)+1

    if not Path(config['outpath']).exists():
        Path(config['outpath']).mkdir()
        print(f" -> created {config['outpath']}")

    minimum_collection = []
    minimal_collection = []
    columns = []
    medians, dd = [], []

    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {config['array']}  {config['seed']}  {str(date)[:10]}...")

        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400


        try:
            st = __get_data(config)
        except:
            print(f" -> failed ...")
            continue

        if st is None or len(st) == 0 or st[0].stats.npts < 1000:
            print(f" -> skipping {date.isoformat()[:10]}!")
            continue

        st0 = st.select(channel=f"*{config['component']}")

        st0.trim(config['tbeg'], config['tend'])


        ## Pa or hPa
        if config['unit'] == "Pa":
            pass
        elif config['unit'] == "hPa":
            for tr in st0:
                tr.data /= 100


        config['nperseg'] = int(st0[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        ff, psds = __calculate_spectra(st0, config, mode=None)


        __save_to_pickle(psds, f"_{str(date).split(' ')[0].replace('-','')}_hourly")

        dd.append(str(date).split(" ")[0].replace("-",""))

    __save_to_pickle(config, "_config")
    __save_to_pickle(ff, "_frequency_axis")

    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)

# In[] ___________________________________________________________


# st = __get_data(config)

# st.plot()

## End of File
