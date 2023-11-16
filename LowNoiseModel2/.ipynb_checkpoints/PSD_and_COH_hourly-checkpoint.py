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

from obspy import UTCDateTime, read, read_inventory
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path
from scipy.signal import coherence, welch

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
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'

# In[] ___________________________________________________________
''' ---- set variables ---- '''

config= {}


config['year'] = 2023

# config['name_appendix'] = "_infrasound" ## _infrasound  |  _absolute

config['seed1'] = "BW.FFBI..BDO"
# config['seed2'] = "BW.ROMY.10.BJZ"
config['seed2'] = "BW.ROMY..BJU"
# config['seed2'] = "BW.ROMY..BJV"


config['date1'] = UTCDateTime(f"{config['year']}-09-23")
config['date2'] = UTCDateTime(f"{config['year']}-10-23")

config['path_to_data1'] = bay_path+f"mseed_online/archive/"
config['path_to_data2'] = archive_path+f"romy_archive/"



# config['type'] = "baro"

## specify unit
config['unit'] = None ## hPa or Pa or None

config['interval_seconds'] = 3600 ## in seconds
config['interval_overlap'] = 0  ## in seconds

## __________________________
## choose psd method
config['mode'] = "multitaper"  ## "multitaper" | "welch"

## __________________________
## set welch and coherence settings

config['taper'] = 'hann'
config['tseconds'] = 1800 ## seconds
config['toverlap'] = 0.9
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
config['outname2'] = f"{config['year']}_{config['sta2']}_{config['interval_seconds']}"
config['outname3'] = f"{config['year']}_{config['sta2']}_{config['cha2'][2]}_{config['interval_seconds']}"

config['outpath1'] = data_path+f"LNM2/PSDS/{config['sta1']}/"
config['outpath2'] = data_path+f"LNM2/PSDS/{config['sta2']}/"
config['outpath3'] = data_path+f"LNM2/PSDS/{config['sta2']}_coherence/"

config['path_to_inv1'] = root_path+"Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml"
config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless.seed.BW_ROMY"


# In[] ___________________________________________________________
'''---- define methods ----'''

def __multitaper_psd(arr, dt, n_win=5):

    import multitaper as mt

    out_psd = mt.MTSpec(arr, nw=n_win, kspec=0, dt=dt)

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
        config['tbeg'] = UTCDateTime(date) -1
        config['tend'] = UTCDateTime(date) + 86400 +1

        try:
            st1 = __read_sds(config['path_to_data1'], config['seed1'], config['tbeg'], config['tend'])
            st2 = __read_sds(config['path_to_data2'], config['seed2'], config['tbeg'], config['tend'])
        except:
            print(f" -> failed to load data ...")
            continue

        if len(st1) == 0 or len(st2) == 0:
            print(st1, st2)

        if len(st1) > 1:
            st1.merge()
        if len(st2) > 1:
            st2.merge()


        ## read inventories
        try:
            inv1 = read_inventory(config['path_to_inv1'])
            inv2 = read_inventory(config['path_to_inv2'])
        except:
            print(f" -> failed to load inventory ...")
            continue


        # st1 = st1.remove_sensitivity(inv1)
        # st2 = st2.remove_sensitivity(inv2)


        ## conversion to Pa or hPa
        for _st in [st1, st2]:

            if "O" in _st[0].stats.channel:

                if config['unit'] == "Pa":
                    for tr in _st:
                        tr.data = tr.data /1.0 /6.28099e5 /1e-5   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 1 mV/hPa
                elif config['unit'] == "hPa":
                    for tr in _st:
                        tr.data = tr.data /1.0 /6.28099e5 /1e-3   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 1 mV/hPa
            if "F" in _st[0].stats.channel:

                for tr in _st:
                    tr.data = tr.data /1.0 /6.28099e5 /0.02  # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity_mb2005=0.02 VPa

            if "J" in _st[0].stats.channel:
                _st = _st.remove_sensitivity(inv2)


        ## Pre-Processing
        st1 = st1.resample(20.0);
        st2 = st2.resample(20.0);

        # st1.plot();
        # st2.plot();

        ## prepare time intervals
        times = __get_time_intervals(config['tbeg'], config['tend'], config['interval_seconds'], config['interval_overlap'])

        ## prepare psd parameters
        config['nperseg'] = int(st1[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        ## run operations for time intervals
        for n, (t1, t2) in enumerate(times):

            ## trim streams for current interval
            _st1 = st1.copy().trim(t1, t2, nearest_sample=False)
            _st2 = st2.copy().trim(t1, t2, nearest_sample=False)

            print("st: ", _st1[0].data.size, _st2[0].data.size)

            if n == 0:
                ## prepare lists
                if config['mode'] == "welch":
                    psds1 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    psds2 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    cohs = zeros([len(times), int(config.get('nperseg')/2)+1])
                elif config['mode'] == "multitaper":
                    # psds1 = zeros([len(times), int((config['interval_seconds']*20))])
                    # psds2 = zeros([len(times), int((config['interval_seconds']*20))])
                    # cohs = zeros([len(times), int(config.get('nperseg')/2)])
                    psds1 = zeros([len(times), int(_st1[0].stats.npts)+1])
                    psds2 = zeros([len(times), int(_st2[0].stats.npts)+1])
                    cohs = zeros([len(times), int(config.get('nperseg')/2)+1])

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

            print("psd: ", len(psd1), len(psd2))
            psds1[n] = psd1
            psds2[n] = psd2


            ## compute coherence
            _N = len(_st1[0].data)
            df = _st1[0].stats.sampling_rate

            t_seg = config['tseconds']
            n_seg = int(df*t_seg) if int(df*t_seg) < _N else _N
            n_over = int(config['toverlap']*n_seg)

            ff_coh, coh = coherence(_st1[0].data, _st2[0].data, fs=df, window='hann', nperseg=n_seg, noverlap=n_over)

            cohs[n] = coh



        ## save psds
        __save_to_pickle(psds1, config['outpath1'],f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        __save_to_pickle(psds2, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## save coherence
#         out = {}
#         out['frequencies'] = ff_coh
#         out['coherence'] = cohs

#         __save_to_pickle(out, config['outpath3'], f"Coherence_{str(date).split(' ')[0].replace('-','')}_hourly")
        __save_to_pickle(cohs, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## add date to dates
        dd.append(str(date).split(" ")[0].replace("-", ""))

    ## save config and frequencies
    __save_to_pickle(config, config['outpath1'], f"{config['outname1']}_config")
    __save_to_pickle(f1, config['outpath1'], f"{config['outname1']}_frequency_axis")

    __save_to_pickle(config, config['outpath2'], f"{config['outname2']}_config")
    __save_to_pickle(f2, config['outpath2'], f"{config['outname2']}_frequency_axis")

    __save_to_pickle(config, config['outpath3'], f"{config['outname3']}_config")
    __save_to_pickle(ff_coh, config['outpath3'], f"{config['outname3']}_frequency_axis")


    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)


## End of File
