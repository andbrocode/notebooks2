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

from andbro__read_sds import __read_sds
from andbro__readYaml import __readYaml

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



if len(sys.argv) > 1:
    config['seed1'] = sys.argv[1]
    config['seed2'] = sys.argv[2]
else:
    config['seed1'] = "BW.FFBI..BDO"  ## F = infrasound | O = absolute
    config['seed2'] = "GR.FUR..BHZ"
    # config['seed2'] = "GR.FUR..BHN"
    # config['seed2'] = "GR.FUR..BHE"
    # config['seed2'] = "BW.ROMY.10.BJZ"
    # config['seed2'] = "BW.ROMY..BJU"
    # config['seed2'] = "BW.ROMY..BJV"

config['date1'] = UTCDateTime(f"{config['year']}-10-01")
config['date2'] = UTCDateTime(f"{config['year']}-10-10")

config['path_to_data1'] = bay_path+f"mseed_online/archive/"
config['path_to_inv1'] = root_path+"Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml"

if "FUR" in config['seed2']:
    config['path_to_data2'] = bay_path+f"mseed_online/archive/"
    config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless.seed.GR_FUR"
elif "ROMY" in config['seed2']:
    config['path_to_data2'] = archive_path+f"romy_archive/"
    config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless.seed.BW_ROMY"


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
config['time_bandwith'] = 4.0
config['mt_method'] = 2 ## 0 = adaptive, 1 = unweighted, 2 = weighted with eigenvalues


config['sta1'] = config['seed1'].split(".")[1]
config['sta2'] = config['seed2'].split(".")[1]

config['cha1'] = config['seed1'].split(".")[3]
config['cha2'] = config['seed2'].split(".")[3]

config['outname1'] = f"{config['year']}_{config['sta1']}_{config['cha1']}_{config['interval_seconds']}"
config['outname2'] = f"{config['year']}_{config['sta2']}_{config['cha2']}_{config['interval_seconds']}"
config['outname3'] = f"{config['year']}_{config['sta1']}_{config['cha1']}_{config['sta2']}_{config['cha2']}_{config['interval_seconds']}"

if "BW.DROMY" in config['seed2']:
    config['outpath1'] = data_path+f"LNM2/PSDS/{config['sta1']}I/"
else:
    config['outpath1'] = data_path+f"LNM2/PSDS/{config['sta1']}/"

config['outpath2'] = data_path+f"LNM2/PSDS/{config['sta2']}/"
config['outpath3'] = data_path+f"LNM2/PSDS/{config['sta2']}_coherence/"

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


def __conversion_to_tilt(st, conf):

    st0 = st.copy()

    def convertTemp(trace):
        Tvolt = trace.data * conf.get('gainTemp')
        coeff = conf.get('calcTempCoefficients')
        return coeff[0] + coeff[1]*Tvolt + coeff[2]*Tvolt**2 + coeff[3]*Tvolt**3

    def convertTilt(trace, conversion, sensitivity):
        return trace.data * conversion * sensitivity

    for tr in st0:
        if tr.stats.channel[-1] == 'T':
            tr.data = convertTemp(tr)
        elif tr.stats.channel[-1] == 'N':
            tr.data = convertTilt(tr, conf['convTN'], conf['gainTilt'])
        elif tr.stats.channel[-1] == 'E':
            tr.data = convertTilt(tr, conf['convTE'], conf['gainTilt'])
        else:
            print("no match")

    print(f"  -> converted data of {st[0].stats.station}")
    return st0


# In[] ___________________________________________________________

def main(config):

    days = int((config['date2'] - config['date1'])/86400)+1

    if not Path(config['outpath1']).exists():
        Path(config['outpath1']).mkdir()
        print(f" -> created {config['outpath1']}")

    if not Path(config['outpath2']).exists():
        Path(config['outpath2']).mkdir()
        print(f" -> created {config['outpath2']}")


    # minimum_collection = []
    # minimal_collection = []
    # columns = []
    # medians, dd = [], []

    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {str(date)[:10]}...")


        ## load data for the entire day
        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400

        try:
            st1 = __read_sds(config['path_to_data1'], config['seed1'], config['tbeg']-1800, config['tend']+1800)
        except:
            print(f" -> failed to load data for {config['seed1']}...")
            continue
        try:
            st2 = __read_sds(config['path_to_data2'], config['seed2'], config['tbeg']-1800, config['tend']+1800)
        except:
            print(f" -> failed to load data for {config['seed2']} ...")
            continue

        ## read inventories
        try:
            inv1 = read_inventory(config['path_to_inv1'])
        except:
            print(f" -> failed to load inventory {config['path_to_inv1']}...")
            continue

        try:
            inv2 = read_inventory(config['path_to_inv2'])
        except:
            print(f" -> failed to load inventory {config['path_to_inv2']}...")
            continue


        if "BW.ROMY" in config['seed2'] and "Z" not in config['seed2']:
            try:
                _stU = __read_sds(config['path_to_data2'], "BW.ROMY..BJU", config['tbeg']-1800, config['tend']+1800)
                _stV = __read_sds(config['path_to_data2'], "BW.ROMY..BJV", config['tbeg']-1800, config['tend']+1800)
                _stZ = __read_sds(config['path_to_data2'], "BW.ROMY.10.BJZ", config['tbeg']-1800, config['tend']+1800)

                print(_stU, _stV, _stZ)

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


        if len(st1) > 1:
            st1.merge()
        if len(st2) > 1:
            st2.merge()

        if len(st1) == 0 or len(st2) == 0:
            print(st1)
            print(st2)
            continue

        ## conversion
        if "O" in st1[0].stats.channel:

            if config['unit'] == "Pa":
                for tr in st1:
                    tr.data = tr.data *1.589e-6 *1e5   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity_mb2005 = 1 mV/hPa
            elif config['unit'] == "hPa":
                for tr in st1:
                    tr.data = tr.data *1.589e-6 *1e3   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity_mb2005 = 1 mV/hPa



        elif "F" in st1[0].stats.channel:
#            for tr in st1:
#                tr.data = tr.data *1.589e-6 /0.02  # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity_mb2005=0.02 V/Pa
            st1 = st1.remove_response(inv1, water_level=10)

        if "J" in st2[0].stats.channel:
            st2 = st2.remove_sensitivity(inv2)

        elif "H" in st2[0].stats.channel:
            st2 = st2.remove_response(inv2, output="ACC", water_level=10)

        elif "A" in st2[0].stats.channel:
            st2 = __conversion_to_tilt(st2, confTilt["BROMY"])

        ## Pre-Processing
        try:
            st1 = st1.split()
            st2 = st2.split()


            if "BW.DROMY" in config['seed2']:
                st2 = st2.filter("lowpass", freq=0.25, corners=4, zerophase=True)
                st2 = st2.decimate(2, no_filter=True) ## 1 -> 0.5 Hz

                st1 = st1.filter("lowpass", freq=0.25, corners=4, zerophase=True)
                st1 = st1.decimate(2, no_filter=True) ## 40 -> 20 Hz
                st1 = st1.decimate(2, no_filter=True) ## 20 -> 10 Hz
                st1 = st1.decimate(2, no_filter=True) ## 10 -> 5 Hz
                st1 = st1.decimate(5, no_filter=True) ## 5 -> 1 Hz
                st1 = st1.decimate(2, no_filter=True) ## 1 -> 0.5 Hz

                ## convert tilt to acceleration
                for tr in st2:
                    tr.data = tr.data*9.81

            else:
                st1 = st1.filter("lowpass", freq=5, corners=4, zerophase=True)
                st2 = st2.filter("lowpass", freq=5, corners=4, zerophase=True)

                st1 = st1.decimate(2, no_filter=True) ## 40 -> 20 Hz
                st1 = st1.decimate(2, no_filter=True) ## 40 -> 20 Hz
                st2 = st2.decimate(2, no_filter=True) ## 40 -> 20 Hz
                # st1 = st1.resample(20.0, no_filter=False)
                # st2 = st2.resample(20.0, no_filter=False)


            st1 = st1.merge()
            st2 = st2.merge()

            st1 = st1.trim(config['tbeg'], config['tend'])
            st2 = st2.trim(config['tbeg'], config['tend'])

        except Exception as e:
            print(e)
            print(f" -> pre-processing failed!")
            continue

        # st1.plot(equal_scale=False);
        # st2.plot(equal_scale=False);

        ## prepare time intervals
        times = __get_time_intervals(config['tbeg'], config['tend'], config['interval_seconds'], config['interval_overlap'])

        ## prepare psd parameters
        config['nperseg'] = int(st1[0].stats.sampling_rate*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))


        print(st1)
        print(st2)


        if len(st1[0].data) != len(st2[0].data):
            print(" -> not sampe amount of samples!")
            continue


        ## run operations for time intervals
        for n, (t1, t2) in enumerate(tqdm(times)):

            ## trim streams for current interval
            # _st1 = st1.copy().trim(t1, t2, nearest_sample=False)
            # _st2 = st2.copy().trim(t1, t2, nearest_sample=False)
            _st1 = st1.copy().trim(t1, t2, nearest_sample=True)
            _st2 = st2.copy().trim(t1, t2, nearest_sample=True)


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
                    cohs = zeros([len(times), int(_st2[0].stats.npts)+1])


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

                ## compute coherence
                ff_coh, coh = coherence(_st1[0].data,
                                        _st2[0].data,
                                        fs=_st2[0].stats.sampling_rate,
                                        window=config.get('taper'),
                                        nperseg=config.get('nperseg'),
                                        noverlap=config.get('noverlap')
                                       )

                cohs[n] = coh

            elif config['mode'] == "multitaper":

                psd_st1 = MTSpec(_st1[0].data,
                                 dt=_st1[0].stats.delta,
                                 nw=config['time_bandwith'],
                                 kspec=config.get("n_taper"),
                                 iadapt=config['mt_method'],
                                )

                _f1, _psd1 = psd_st1.rspec()
                f1, psd1 = _f1.reshape(_f1.size), _psd1.reshape(_psd1.size)


                psd_st2 = MTSpec(_st2[0].data,
                                 dt=_st2[0].stats.delta,
                                 nw=config['time_bandwith'],
                                 kspec=config.get("n_taper"),
                                 iadapt=config['mt_method'],
                                )

                _f2, _psd2 = psd_st2.rspec()
                f2, psd2 = _f2.reshape(_f2.size), _psd2.reshape(_psd2.size)

                if psd1.size == psd2.size:
                    Pxy  = MTCross(psd_st1, psd_st2, wl=0.001)
                    N = Pxy.freq.size
                    ff_coh, coh = Pxy.freq[:,0][:N//2], Pxy.cohe[:,0][:N//2]
                else:
                    print(_st1[0].data.size, _st2[0].data.size, psd1.size, psd2.size)
                    continue

                # print(ff_coh.size, coh.size, _st1[0].data.size, psd1.size)
                # print(ff_coh[0], ff_coh[-1], f1[0], f1[-1])

            psds1[n] = psd1
            psds2[n] = psd2
            cohs[n] = coh




        # plt.figure()
        # plt.plot(_st1[0].times(), _st1[0].data)
        # plt.figure()
        # plt.plot(_st2[0].times(), _st2[0].data)
        # plt.figure()
        # plt.loglog(f1, psd1)
        # plt.figure()
        # plt.loglog(f2, psd2)
        # plt.figure()
        # plt.semilogx(ff_coh, coh)
        # plt.show()

        ## save psds
        out1 = {}
        out1['frequencies'] = f1
        out1['psd'] = psds1

        __save_to_pickle(out1, config['outpath1'], f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        # __save_to_pickle(psds1, config['outpath1'],f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        out2 = {}
        out2['frequencies'] = f2
        out2['psd'] = psds2

        __save_to_pickle(out2, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        # __save_to_pickle(psds2, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## save coherence
        out3 = {}
        out3['frequencies'] = ff_coh
        out3['coherence'] = cohs

        __save_to_pickle(out3, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")
        # __save_to_pickle(cohs, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")


        ## add date to dates
        # dd.append(str(date).split(" ")[0].replace("-", ""))

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

    gc.collect()

## End of File
