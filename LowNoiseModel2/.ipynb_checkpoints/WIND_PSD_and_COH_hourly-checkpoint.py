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
import os
import sys
import gc

from tqdm import tqdm
from obspy import UTCDateTime, read, read_inventory
from obspy.signal.rotate import rotate2zne
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path
from scipy.signal import coherence, welch
from multitaper import MTCross, MTSpec
from scipy.fftpack import diff

from andbro__read_sds import __read_sds
from andbro__readYaml import __readYaml
from andbro__querrySeismoData import __querrySeismoData
from andbro__load_FURT_stream import __load_furt_stream

import warnings
warnings.filterwarnings('ignore')


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



# In[] ___________________________________________________________
''' ---- set variables ---- '''

config = {}


config['year'] = 2024



if len(sys.argv) > 1:
    config['seed1'] = sys.argv[1]
else:
    config['seed1'] = "BW.FURT..LAW"  ## F = infrasound | O = absolute



config['date1'] = UTCDateTime(f"{config['year']}-03-23")
config['date2'] = UTCDateTime(f"{config['year']}-03-24")

config['path_to_data1'] = bay_path+f"mseed_online/archive/"
config['path_to_inv1'] = root_path+"Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml"

config['path_to_status_data'] = archive_path+f"temp_archive/"


## specify unit
config['unit'] = "Pa" ## hPa or Pa or None

config['interval_seconds'] = 3600 ## in seconds
config['interval_overlap'] = 0  ## in seconds

## __________________________
## choose psd method
config['mode'] = "multitaper"  ## "multitaper" | "welch"

## __________________________
## set welch and coherence settings

config['taper'] = 'hann'
config['tseconds'] = 3600 ## seconds
config['toverlap'] = 0 ## 0.75
config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0, 0.05) # in Hz

## __________________________
## set multitaper settings

## number of taper for multitaper to use
config['n_taper'] = 5
config['time_bandwith'] = 3.5
config['mt_method'] = 2 ## 0 = adaptive, 1 = unweighted, 2 = weighted with eigenvalues


config['sta1'] = config['seed1'].split(".")[1]

config['cha1'] = config['seed1'].split(".")[3]

config['outname1'] = f"{config['year']}_{config['sta1']}_{config['cha1']}_{config['interval_seconds']}"

config['outpath1'] = data_path+f"LNM2/PSDS/{config['sta1']}/"

## tiltmeter configurations
confTilt = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/", "tiltmeter.conf")





# In[] ___________________________________________________________
'''---- define methods ----'''

def __multitaper_psd(arr, dt, n_win=5, time_bandwidth=4.0):

    import multitaper as mt

    out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=config['mt_method'])

    _f, _psd = out_psd.rspec()

    f = _f.reshape(_f.size)
    psd = _psd.reshape(_psd.size)


    return f, psd


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


def __save_to_pickle(obj, path, name):

    ofile = open(path+name+".pkl", 'wb')
    pickle.dump(obj, ofile)

    if Path(path+name+".pkl").exists():
        print(f"\n -> created:  {path}{name}.pkl")


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


def __interpolate_nan(array_like):

    from numpy import isnan, interp

    array = array_like.copy()

    nans = isnan(array)

    def get_x(a):
        return a.nonzero()[0]

    array[nans] = interp(get_x(nans), get_x(~nans), array[~nans])

    return array



# In[] ___________________________________________________________

def main(config):

    days = int((config['date2'] - config['date1'])/86400)+1

    if not Path(config['outpath1']).exists():
        Path(config['outpath1']).mkdir()
        print(f" -> created {config['outpath1']}")



    # minimum_collection = []
    # minimal_collection = []
    # columns = []
    # medians, dd = [], []

    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {str(date)[:10]}...")

        offset_sec = 10800  ## seconds

        ## load data for the entire day
        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400

        try:
            furt = __load_furt_stream(config['tbeg']-10, config['tend']+10, path_to_archive=bay_path+'gif_online/FURT/WETTER/')

            st1 = furt.select(channel=config['seed1'].split(".")[3])

        except:
            print(f" -> failed to load data for {config['seed1']}...")
            continue

#         try:
#             st1, inv1 = __querrySeismoData(
#                                         seed_id=config['seed1'],
#                                         starttime=config['tbeg']-offset_sec,
#                                         endtime=config['tend']+offset_sec,
#                                         repository="online",
#                                         path=None,
#                                         restitute=False,
#                                         detail=None,
#                                         fill_value=None,
#             )

#             st1 = st1.remove_response(inv1, type="VEL")

#         except:
#             print(f" -> failed to load data for {config['seed1']}...")
#             continue


        if len(st1) > 1:
            st1.merge()

        if len(st1) == 0:
            print(st1)
            continue


        ## Pre-Processing
        try:

            ## interpolate NaN values
            for tr in st1:
                tr.data = __interpolate_nan(tr.data)

            st1 = st1.trim(config['tbeg'], config['tend'])

        except Exception as e:
            print(f" -> pre-processing failed!")
            print(e)
            continue


        ## prepare time intervals
        times = __get_time_intervals(config['tbeg'], config['tend'], config['interval_seconds'], config['interval_overlap'])

        ## prepare psd parameters
        config['nperseg'] = int(st1[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        print(st1)


        ## run operations for time intervals
        for n, (t1, t2) in enumerate(tqdm(times)):

            ## trim streams for current interval
            _st1 = st1.copy().trim(t1, t2, nearest_sample=True)

            _st1 = _st1.detrend("linear").taper(0.05)


            if n == 0:

                ## prepare lists
                if config['mode'] == "welch":
                    psds1 = zeros([len(times), int(config.get('nperseg')/2)+1])

                elif config['mode'] == "multitaper":
                    psds1 = zeros([len(times), int(_st1[0].stats.npts)+1])


            ## compute power spectra
            if config['mode'] == "welch":

                f1, psd1 = welch(
                                _st1[0].data,
                                fs=_st1[0].stats.sampling_rate,
                                window=config.get('taper'),
                                nperseg=config.get('nperseg'),
                                noverlap=config.get('noverlap'),
                                nfft=config.get('nfft'),
                                detrend=config.get('detrend'),
                                return_onesided=config.get('onesided'),
                                scaling=config.get('scaling'),
                               )



            elif config['mode'] == "multitaper":

                psd_st1 = MTSpec(_st1[0].data,
                                 dt=_st1[0].stats.delta,
                                 nw=config['time_bandwith'],
                                 kspec=config.get("n_taper"),
                                 iadapt=config['mt_method'],
                                )

                _f1, _psd1 = psd_st1.rspec()
                f1, psd1 = _f1.reshape(_f1.size), _psd1.reshape(_psd1.size)


            psds1[n] = psd1


        ## save psds
        out1 = {}
        out1['frequencies'] = f1
        out1['psd'] = psds1

        __save_to_pickle(out1, config['outpath1'], f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        # __save_to_pickle(psds1, config['outpath1'],f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")


    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":

    main(config)

    gc.collect();

## End of File
