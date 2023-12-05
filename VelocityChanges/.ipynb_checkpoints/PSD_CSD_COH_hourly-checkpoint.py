# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__   = '2023'


# In[] ___________________________________________________________
'''---- import libraries ----'''

import matplotlib.pyplot as plt
import pickle
import os
import sys

from obspy import UTCDateTime, read, read_inventory
from obspy.signal.rotate import rotate2zne
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path
from scipy.signal import coherence, welch, csd

from andbro__read_sds import __read_sds
from andbro__readYaml import __readYaml

from functions.get_xwt import __compute_cross_wavelet_transform

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


config['year'] = 2023


config['seed1'] = "BW.RLAS..BJZ"

config['seed2'] = "BW.ROMY.10.BJZ"

# if len(sys.argv) > 1:
#     config['seed2'] = sys.argv[1]
# else:
#     # config['seed2'] = "GR.FUR..BHZ"
#     # config['seed2'] = "GR.FUR..BHN"
#     # config['seed2'] = "GR.FUR..BHE"
#     # config['seed2'] = "BW.ROMY.10.BJZ"
#     # config['seed2'] = "BW.ROMY..BJU"
#     # config['seed2'] = "BW.ROMY..BJV"

config['date1'] = UTCDateTime(f"{config['year']}-09-23")
config['date2'] = UTCDateTime(f"{config['year']}-10-23")

config['path_to_data1'] = archive_path+f"romy_archive/"
config['path_to_inv1'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless.seed.BW_RLAS"

config['path_to_data2'] = archive_path+f"romy_archive/"
config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless.seed.BW_ROMY"


config['interval_seconds'] = 3600 ## in seconds
config['interval_overlap'] = 0  ## in seconds

## __________________________
## choose psd method
config['mode'] = "welch"  ## "multitaper" | "welch"

## __________________________
## set welch and coherence settings

config['taper'] = 'hann'
config['tseconds'] = 1800 ## seconds
config['toverlap'] = 0.75
config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0, 0.05) # in Hz

## __________________________
## set multitaper settings

## number of taper for multitaper to use
config['n_taper'] = 5


config['sta1'] = config['seed1'].split(".")[1]
config['sta2'] = config['seed2'].split(".")[1]

config['cha1'] = config['seed1'].split(".")[3]
config['cha2'] = config['seed2'].split(".")[3]

config['outname1'] = f"{config['year']}_{config['sta1']}_{config['interval_seconds']}"
config['outname2'] = f"{config['year']}_{config['sta2']}_{config['cha2']}_{config['interval_seconds']}"
config['outname3'] = f"{config['year']}_{config['sta1']}_{config['sta2']}_coh_{config['interval_seconds']}"
config['outname4'] = f"{config['year']}_{config['sta1']}_{config['sta2']}_csd_{config['interval_seconds']}"

config['outpath1'] = data_path+f"VelocityChanges/data/PSDS/{config['sta1']}/"
config['outpath2'] = data_path+f"VelocityChanges/data/PSDS/{config['sta2']}/"
config['outpath3'] = data_path+f"VelocityChanges/data/PSDS/{config['sta1']}_{config['sta2']}_coh/"
config['outpath4'] = data_path+f"VelocityChanges/data/PSDS/{config['sta1']}_{config['sta2']}_csd/"


# In[] ___________________________________________________________
'''---- define methods ----'''

def __multitaper_psd(arr, dt, n_win=5, time_bandwidth=4.0):

    import multitaper as mt

    out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=0)

    _f, _psd = out_psd.rspec()

    f = _f.reshape(_f.size)
    psd = _psd.reshape(_psd.size)

    ## 95% confidence interval
    # _psd95 = out_psd.jackspec()
    # psd95_lower, psd95_upper = psd95[::2, 0], psd95[::2, 1]

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


        ## load data for the entire day
        config['tbeg'] = UTCDateTime(date) - 1
        config['tend'] = UTCDateTime(date) + 86400 + 1

        try:
            st1 = __read_sds(config['path_to_data1'], config['seed1'], config['tbeg'], config['tend'])
            st2 = __read_sds(config['path_to_data2'], config['seed2'], config['tbeg'], config['tend'])
        except:
            print(f" -> failed to load data ...")
            continue

        print(st1, st2)

        if len(st1) == 0:
            print(st1)
        elif len(st2) == 0:
            print(st2)

        if len(st1) > 1:
            st1.merge()
        elif len(st2) > 1:
            st2.merge()


        ## read inventories
        try:
            inv1 = read_inventory(config['path_to_inv1'])
        except:
            print(f" -> failed to load inventory 1 ...")
            continue
        try:
            inv2 = read_inventory(config['path_to_inv2'])
        except:
            print(f" -> failed to load inventory 2 ...")
            continue

        if "ROMY" in config['seed2'] and "Z" not in config['seed2']:
            try:
                _stU = __read_sds(config['path_to_data2'], "BW.ROMY..BJU", config['tbeg'], config['tend'])
                _stV = __read_sds(config['path_to_data2'], "BW.ROMY..BJV", config['tbeg'], config['tend'])
                _stZ = __read_sds(config['path_to_data2'], "BW.ROMY.10.BJZ", config['tbeg'], config['tend'])

                ori_z = inv2.get_orientation("BW.ROMY.10.BJZ")
                ori_u = inv2.get_orientation("BW.ROMY..BJU")
                ori_v = inv2.get_orientation("BW.ROMY..BJV")

                romy_z, romy_n, romy_e = rotate2zne(
                                                   _stZ[0].data, ori_z['azimuth'], ori_z['dip'],
                                                   _stU[0].data, ori_u['azimuth'], ori_u['dip'],
                                                   _stV[0].data, ori_v['azimuth'], ori_v['dip'],
                                                   inverse=False
                                                  )

                if "N" in config['seed2']:
                    _stU[0].data = romy_n
                    st2 = _stU.copy()
                    # st2.select(channel="*U")[0].stats.channel = "BJN"

                elif "E" in config['seed2']:
                    _stV[0].data = romy_e
                    st2 = _stV.copy()
                    # st2.select(channel="*V")[0].stats.channel = "BJE"

            except Exception as e:
                print(e)
                print(f" -> failed to rotate ROMY ...")
                continue


        ## conversion
        if "J" in st1[0].stats.channel:
            st1 = st1.remove_sensitivity(inv1)

        if "J" in st2[0].stats.channel:
            st2 = st2.remove_sensitivity(inv2)

        if "H" in st2[0].stats.channel:
            st2 = st2.remove_response(inv2, output="ACC", water_level=10)

#         ## Pre-Processing
#         try:
#             st1 = st1.split()
#             st2 = st2.split()

#             st1 = st1.detrend("linear")
#             st2 = st2.detrend("linear")

#             st1 = st1.decimate(2, no_filter=False) ## 40 -> 20 Hz

#             if "DROMY" in config['seed2']:

#                 st1 = st1.decimate(2, no_filter=False) ## 20 -> 10 Hz
#                 st1 = st1.decimate(2, no_filter=False) ## 10 -> 5 Hz
#                 st1 = st1.decimate(5, no_filter=False) ## 5 -> 1 Hz

#             # st1 = st1.filter("highpass", freq=1e-4, corners=4, zerophase=True)
#             # st2 = st2.filter("highpass", freq=1e-4, corners=4, zerophase=True)

#             st1 = st1.merge()
#             st2 = st2.merge()

#         except Exception as e:
#             print(e)
#             print(f" -> pre-processing failed!")
#             continue

        # st1.plot(equal_scale=False);
        # st2.plot(equal_scale=False);

        ## prepare time intervals
        times = __get_time_intervals(config['tbeg'], config['tend'], config['interval_seconds'], config['interval_overlap'])

        ## prepare psd parameters
        config['nperseg'] = int(st1[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        # print(st1)
        # print(st2)

        ## run operations for time intervals
        for n, (t1, t2) in enumerate(times):

            ## trim streams for current interval
            _st1 = st1.copy().trim(t1, t2, nearest_sample=False)
            _st2 = st2.copy().trim(t1, t2, nearest_sample=False)

#            print("st: ", _st1[0].data.size, _st2[0].data.size)

            if n == 0:
                ## prepare lists
                if config['mode'] == "welch":
                    psds1 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    psds2 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    cohs = zeros([len(times), int(config.get('nperseg')/2)+1])
                    csds = zeros([len(times), int(config.get('nperseg')/2)+1])
                    xwts = zeros([len(times), int(config.get('nperseg')/2)+1])
                    xwts = []

                elif config['mode'] == "multitaper":
                    # psds1 = zeros([len(times), int((config['interval_seconds']*20))])
                    # psds2 = zeros([len(times), int((config['interval_seconds']*20))])
                    # cohs = zeros([len(times), int(config.get('nperseg')/2)])
                    psds1 = zeros([len(times), int(_st1[0].stats.npts)+1])
                    psds2 = zeros([len(times), int(_st2[0].stats.npts)+1])
                    cohs = zeros([len(times), int(config.get('nperseg')/2)+1])
                    csds = zeros([len(times), int(config.get('nperseg')/2)+1])
                    xwts = zeros([len(times), int(config.get('nperseg')/2)+1])


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

                f2, psd2 = welch(
                                _st2[0].data,
                                fs=_st2[0].stats.sampling_rate,
                                window=config.get('taper'),
                                nperseg=config.get('nperseg'),
                                noverlap=config.get('noverlap'),
                                nfft=config.get('nfft'),
                                detrend=config.get('detrend'),
                                return_onesided=config.get('onesided'),
                                scaling=config.get('scaling'),
                               )

            elif config['mode'] == "multitaper":

                f1, psd1 = __multitaper_psd(_st1[0].data, _st1[0].stats.delta, n_win=config.get("n_taper"))

                f2, psd2 = __multitaper_psd(_st2[0].data, _st2[0].stats.delta, n_win=config.get("n_taper"))

#            print("psd: ", len(psd1), len(psd2))
            psds1[n] = psd1
            psds2[n] = psd2


            ## compute coherence
            _N = len(_st1[0].data)
            df = _st1[0].stats.sampling_rate
            dt = _st1[0].stats.delta
            
            t_seg = config['tseconds']
            n_seg = int(df*t_seg) if int(df*t_seg) < _N else _N
            n_over = int(config['toverlap']*n_seg)

            ## compute coherence
            ff_coh, _coh = coherence(_st1[0].data, _st2[0].data, fs=df, window='hann', nperseg=n_seg, noverlap=n_over)

            cohs[n] = _coh

            ## compute cross spectral density
            ff_csd, _csd = csd(_st1[0].data, _st2[0].data, fs=df, window='hann', nperseg=n_seg, noverlap=n_over)

            csds[n] = _csd

            ## compute cross wavelet transform
            out = __compute_cross_wavelet_transform(_st1[0].times(), _st1[0].data, _st2[0].data, dt,
                                                    datalabels=["ROMY", "RLAS"],
                                                    xscale="log",
                                                    plot=False,
                                                   )
            ff_xwt = out['frequencies']
            xwts.append(out['global_mean_xwt'])
            
            
        ## save psds
        out = {}
        out['frequencies'] = f1
        out['psd'] = psds1
        
        # __save_to_pickle(psds1, config['outpath1'],f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        __save_to_pickle(out, config['outpath1'],f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        
        out = {}
        out['frequencies'] = f2
        out['psd'] = psds2
        
        # __save_to_pickle(psds2, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        __save_to_pickle(out, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## store coherence
        out = {}
        out['frequencies'] = ff_coh
        out['coherence'] = cohs

        __save_to_pickle(out, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        
        ## store cross wavelet transform
        out = {}
        out['frequencies'] = ff_xwt
        out['coherence'] = array(xwts)

        __save_to_pickle(out, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        ## store cross spectral density
        out = {}
        out['frequencies'] = ff_csd
        out['csd'] = csds

        __save_to_pickle(out, config['outpath4'], f"{config['outname4']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## add date to dates
        dd.append(str(date).split(" ")[0].replace("-", ""))

    ## save config and frequencies
#     __save_to_pickle(config, config['outpath1'], f"{config['outname1']}_config")
#     __save_to_pickle(f1, config['outpath1'], f"{config['outname1']}_frequency_axis")

#     __save_to_pickle(config, config['outpath2'], f"{config['outname2']}_config")
#     __save_to_pickle(f2, config['outpath2'], f"{config['outname2']}_frequency_axis")

#     __save_to_pickle(config, config['outpath3'], f"{config['outname3']}_config")
#     __save_to_pickle(ff_coh, config['outpath3'], f"{config['outname3']}_frequency_axis")


    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)


## End of File
