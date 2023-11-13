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

from obspy import UTCDateTime, read, read_inventory
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


# config['array'] = "FFBI"

config['year'] = 2023

##  O = absolute | F = infrasound
# config['component'] = "F"
# config['name_appendix'] = "_infrasound" ## _infrasound  |  _absolute

config['seed1'] = "BW.FFBI..BDO"
config['seed2'] = "BW.ROMY.10.BJZ"

config['date1'] = UTCDateTime(f"{config['year']}-09-23")
config['date2'] = UTCDateTime(f"{config['year']}-10-23")

config['path_to_data1'] = f"/bay200/mseed_online/archive/"
config['path_to_data2'] = f"/import/freenas-ffb-01-data/romy_archive/"



# config['type'] = "baro"

## specify unit
config['unit'] = None ## hPa or Pa or None

config['interval'] = 3600
config['interval_overlap'] = None # in percent
config['taper'] = 'hanning'

config['tseconds'] = 3600 ## seconds

config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0, 0.05) # in Hz


config['mode'] = "welch"  ## "multitaper" | "welch"

## number of taper for multitaper to use
config['n_taper'] = 5


config['sta1'] = config['seed1'].split(".")[1]
config['sta2'] = config['seed2'].split(".")[1]


config['outname1'] = f"{config['year']}_{config['sta1']}_{config['interval']}"
config['outname2'] = f"{config['year']}_{config['sta2']}_{config['interval']}"

config['outpath1'] = f"/import/kilauea-data/LNM2/PSDS/{config['sta1']}/"
config['outpath2'] = f"/import/kilauea-data/LNM2/PSDS/{config['sta2']}/"

config['path_to_inv1'] = "/home/brotzer/Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml"
config['path_to_inv2'] = "/home/brotzer/Documents/ROMY/stationxml_ringlaser/dataless.seed.BW_ROMY"


# In[] ___________________________________________________________
'''---- define methods ----'''

def __multitaper_psd(arr, dt, n_win=5):

    import multitaper as mt

    out_psd = mt.MTSpec(arr, nw=n_win, kspec=0, dt=dt)

    _f, _psd = out_psd.rspec()

    f = _f.reshape(_f.size)
    psd = _psd.reshape(_psd.size)

    return f, psd




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


def __calculate_spectra(st, config, mode='welch'):

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
    if mode == "welch":
        size_psd = int(config.get('nperseg')/2)+1
        psd = zeros([intervals, size_psd])

        if size_psd >= len(st[0].data):
            print(f"ERROR: reduce nperseg or noverlap or segments! {size_psd} > {len(st[0].data)}")
            return
    elif mode == "multitaper":
        psd = zeros([intervals, 144002])


    for i, tr in enumerate(st):

        # initite variables for while loop
        dt1 = st[0].stats.starttime
        dt2 = st[0].stats.starttime + config['interval']
        n = 0

        while dt2 <= st[0].stats.endtime:

            tr_tmp = tr.copy()
            tr_tmp.trim(starttime = UTCDateTime(dt1), endtime=UTCDateTime(dt2))


            if mode == "welch":

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

            elif mode == "multitaper":

                f, psd0 = __multitaper_psd(tr_tmp.data, tr_tmp.stats.delta, n_win=config.get("n_taper"))

            ## add psd to data matrix
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

    if not Path(config['outpath1']).exists():
        Path(config['outpath1']).mkdir()
        print(f" -> created {config['outpath1']}")

    if not Path(config['outpath2']).exists():
        Path(config['outpath2']).mkdir()
        print(f" -> created {config['outpath2']}")


    minimum_collection = []
    minimal_collection = []
    columns = []
    medians, dd = [], []

    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {str(date)[:10]}...")

        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400

        try:
            st1 = __read_sds(config['path_to_data1'], config['seed1'], config['tbeg'], config['tend'])
            st2 = __read_sds(config['path_to_data2'], config['seed2'], config['tbeg'], config['tend'])
        except:
            print(f" -> failed to load data ...")
            continue

        if len(st1) == 0 or len(st2) == 0:
            print(st1, st2)


        try:
            inv1 = read_inventory(config['path_to_inv1'])
            inv2 = read_inventory(config['path_to_inv2'])
        except:
            print(f" -> failed to load inventory ...")
            continue


        st1 = st1.remove_sensitivity(inv1)
        st2 = st2.remove_sensitivity(inv2)


        ## Pa or hPa
        # if config['component'] == "O":

        #     if config['unit'] == "Pa":
        #         for tr in st0:
        #             tr.data *= 100000
        #     else:
        #         for tr in st0:
        #             tr.data *= 1000
        # if config['component'] == "F":

        #     for tr in st0:
        #         tr.data *= 100


        config['nperseg'] = int(st1[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        ff1, psds1 = __calculate_spectra(st1, config, mode=config['mode'])

        config['outpath'] = config['outpath1']
        config['outname'] = config['outname1']

        __save_to_pickle(psds1, f"_{str(date).split(' ')[0].replace('-','')}_hourly")
        __save_to_pickle(config, "_config")
        __save_to_pickle(ff1, "_frequency_axis")

        ### ____________________________________
        ###

        config['nperseg'] = int(st2[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))

        ff2, psds2 = __calculate_spectra(st2, config, mode=config['mode'])

        config['outpath'] = config['outpath2']
        config['outname'] = config['outname2']

        __save_to_pickle(psds2, f"_{str(date).split(' ')[0].replace('-','')}_hourly")
        __save_to_pickle(config, "_config")
        __save_to_pickle(ff2, "_frequency_axis")


        dd.append(str(date).split(" ")[0].replace("-",""))


    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)

# In[] ___________________________________________________________

st = __get_data(config)

inv = read_inventory("/home/brotzer/Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml")

st = st.remove_sensitivity(inv)

st = st.select(channel=f"*{config['component']}")

## convert to hPa
if config['component'] == "O":
    for tr in st:
        tr.data *= 1000

st.plot()

## End of File
