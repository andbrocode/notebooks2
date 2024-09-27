# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__ = '2023'

# In[] ___________________________________________________________
'''---- import libraries ----'''

import matplotlib.pyplot as plt
import pickle
import os
import sys
import gc

from tqdm import tqdm
from obspy import UTCDateTime, read, Stream, read_inventory
from obspy.signal.rotate import rotate2zne
from numpy import log10, zeros, append, linspace, mean, median, array, where, transpose, shape, histogram, nan
from pandas import DataFrame, concat, Series, date_range, to_pickle
from pathlib import Path
from scipy.signal import coherence, welch
from multitaper import MTCross, MTSpec
from scipy.fftpack import diff
from numpy import ma

from andbro__read_sds import __read_sds
from andbro__readYaml import __readYaml

import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/ontap-ffb-bay200/'
    lamont_path = '/home/andbro/lamont/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
    lamont_path = '/lamont/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
    lamont_path = '/lamont/'

# In[] ___________________________________________________________
''' ---- set variables ---- '''

config = {}

# expected argument for seed1 (pressure) and seed2
if len(sys.argv) > 1:
    config['seed1'] = sys.argv[1]
    config['seed2'] = sys.argv[2]
else:
    print(f" missing seed arguments!")
    # config['seed1'] = "BW.FFBI.30.BDO"  ## F = infrasound | O = absolute
    # config['seed2'] = "GR.FUR..BHZ"
    # config['seed2'] = "GR.FUR..BHN"
    # config['seed2'] = "GR.FUR..BHE"
    # config['seed2'] = "BW.ROMY.10.BJZ"
    # config['seed2'] = "BW.ROMY..BJU"
    # config['seed2'] = "BW.ROMY..BJV"

# define a project name
config['project'] = "2"

# specify year
config['year'] = 2024

# define time period to analyze
config['date1'] = UTCDateTime(f"{config['year']}-04-01")
config['date2'] = UTCDateTime(f"{config['year']}-06-30")

# config['path_to_data1'] = bay_path+f"mseed_online/archive/"
config['path_to_data1'] = archive_path+f"temp_archive/"
config['path_to_inv1'] = root_path+"Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml"

# specify path to data and metadata
if "FUR" in config['seed2']:
    config['path_to_data2'] = bay_path+f"mseed_online/archive/"
    config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/dataless.seed.GR_FUR"
elif "ROMY" in config['seed2']:
    config['path_to_data2'] = archive_path+f"temp_archive/"
    config['path_to_inv2'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/dataless.seed.BW_ROMY"

# specify path to status files
config['path_to_status_data'] = archive_path+f"temp_archive/"

# specify unit for pressure
config['unit'] = "Pa" ## hPa or Pa or None

# define the interval and overlap to analyze (default is one hour = 3600s)
config['interval_seconds'] = 3600 ## in seconds
config['interval_overlap'] = 0  ## in seconds

config['sampling_rate'] = 5.0 ## Hz

# __________________________
# choose psd method
config['mode'] = "multitaper"  ## "multitaper" | "welch"

# __________________________
# set welch and coherence settings
config['taper'] = 'hann'
config['tseconds'] = 3600 ## seconds
config['toverlap'] = 0 ## 0.75
config['nfft'] = None
config['detrend'] = 'constant'
config['scaling'] = 'density'
config['onesided'] = True
config['frequency_limits'] = None # (0, 0.05) # in Hz

# __________________________
# set multitaper settings

# number of taper for multitaper to use
config['n_taper'] = 5
config['time_bandwith'] = 3.5
config['mt_method'] = 2 ## 0 = adaptive, 1 = unweighted, 2 = weighted with eigenvalues

config['sta1'] = config['seed1'].split(".")[1]
config['sta2'] = config['seed2'].split(".")[1]

config['cha1'] = config['seed1'].split(".")[3]
config['cha2'] = config['seed2'].split(".")[3]

config['outname1'] = f"{config['year']}_{config['sta1']}_{config['cha1']}_{config['interval_seconds']}"
config['outname2'] = f"{config['year']}_{config['sta2']}_{config['cha2']}_{config['interval_seconds']}"
config['outname3'] = f"{config['year']}_{config['sta1']}_{config['cha1']}_{config['sta2']}_{config['cha2']}_{config['interval_seconds']}"

if "BW.DROMY" in config['seed2']:
    config['outpath1'] = data_path+f"LNM2/PSDS{config['project']}/{config['sta1']}I/"
else:
    config['outpath1'] = data_path+f"LNM2/PSDS{config['project']}/{config['sta1']}/"

config['outpath2'] = data_path+f"LNM2/PSDS{config['project']}/{config['sta2']}/"
config['outpath3'] = data_path+f"LNM2/PSDS{config['project']}/{config['sta2']}_coherence/"

# tiltmeter configurations
confTilt = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/", "tiltmeter.conf")

# adjustments if ROMY is to be integrated
if "BW.ROMY" in config['seed2'] and "BA" in config['seed2']:
    config['seed2'] = config['seed2'].replace("A", "J")
    integrate = True
else:
    integrate = False

path_to_figs = data_path+"LNM2/testfigs/"

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


def __load_status(tbeg, tend, ring, path_to_data):

    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range
    from obspy import UTCDateTime
    from os.path import isfile
    from numpy import array, arange, ones, nan

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    # dummy
    def __make_dummy(date):
        NN = 1440
        df_dummy = DataFrame()
        df_dummy['times_utc'] = array([UTCDateTime(date) + _t for _t in arange(30, 86400, 60)])
        for col in ["quality", "fsagnac", "mlti", "ac_threshold", "dc_threshold"]:
            df_dummy[col] = ones(NN)*nan

        return df_dummy

    df = DataFrame()
    for dat in date_range(dd1, dd2):
        file = f"{str(dat)[:4]}/BW/R{ring}/R{ring}_"+str(dat)[:10]+"_status.pkl"

        try:

            if not isfile(f"{path_to_data}{file}"):
                print(f" -> no such file: {file}")
                df = concat([df, __make_dummy(dat)])
            else:
                df0 = read_pickle(path_to_data+file)
                df = concat([df, df0])
        except:
            print(f" -> error for {file}")

    if df.empty:
        print(" -> empty dataframe!")
        return df

    ## trim to defined times
    df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]

    ## correct seconds
    df['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df['times_utc']]

    return df


def __load_lxx(tbeg, tend, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{year}/logfiles/LXX_maintenance.log"

        lxx = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

    else:

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tbeg.year}/logfiles/LXX_maintenance.log"
        lxx1 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tend.year}/logfiles/LXX_maintenance.log"
        lxx2 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        lxx = concat([lxx1, lxx2])

    lxx = lxx[(lxx.datetime > tbeg) & (lxx.datetime < tend)]

    return lxx


# In[] ___________________________________________________________

def main(config):

    days = int((config['date2'] - config['date1'])/86400)+1

    if not Path(config['outpath1']).exists():
        Path(config['outpath1']).mkdir()
        print(f" -> created {config['outpath1']}")

    if not Path(config['outpath2']).exists():
        Path(config['outpath2']).mkdir()
        print(f" -> created {config['outpath2']}")


    ## set counter
    size_counter = 0
    mask_counter = 0
    mltiU_counter = 0
    mltiV_counter = 0
    mltiZ_counter = 0
    lxx_counter = 0

    # error status
    error = False

    for date in date_range(str(config['date1'].date), str(config['date2'].date), days):

        print(f"\nprocessing  {str(date)[:10]}...")

        offset_sec = 10800  ## seconds

        ## load data for the entire day
        config['tbeg'] = UTCDateTime(date)
        config['tend'] = UTCDateTime(date) + 86400

        try:
            st1 = __read_sds(config['path_to_data1'], config['seed1'], config['tbeg']-offset_sec, config['tend']+offset_sec)
        except:
            print(f" -> failed to load data for {config['seed1']}...")
            # continue
            error = True
        try:
            st2 = __read_sds(config['path_to_data2'], config['seed2'], config['tbeg']-offset_sec, config['tend']+offset_sec)
        except:
            print(f" -> failed to load data for {config['seed2']} ...")
            # continue
            error = True

        ## read inventories
        try:
            inv1 = read_inventory(config['path_to_inv1'])
        except:
            print(f" -> failed to load inventory {config['path_to_inv1']}...")
            # continue
            error = True

        try:
            inv2 = read_inventory(config['path_to_inv2'])
        except:
            print(f" -> failed to load inventory {config['path_to_inv2']}...")
            # continue
            error = True

        if len(st1) > 1:
            st1.merge(fill_value=0)

        if len(st2) > 1:
            st2.merge(fill_value=0)

        # integrate romy data from rad/s to rad
        if integrate:
            print(f" -> integrating ...")
            st2 = st2.detrend("linear")
            st2 = st2.integrate(method='spline');
            st2 = st2.detrend("linear")

            # tilt to acceleration
            for tr in st2:
                tr.data *= 9.81 ## m/s^2

        # Data Processing
        try:
            # pressure from hPa to Pa
            if "O" in st1[0].stats.channel:
                st1[0].data = st1[0].data*100

            if "J" in st2[0].stats.channel:
                # st2 = st2.remove_sensitivity(inv2)
                pass

            if "H" in st2[0].stats.channel:
                st2 = st2.remove_response(inv2, output="ACC", water_level=60)

            if "A" in st2[0].stats.channel:
                if st2[0].stats.station == "DROMY":
                    st2 = __conversion_to_tilt(st2, confTilt["BROMY"])
                elif st2[0].stats.station == "ROMYT":
                    st2 = __conversion_to_tilt(st2, confTilt["ROMYT"])
                else:
                    print(" -> not defined")
        except:
            error = True

        # Pre-Processing
        try:
            st1 = st1.split()
            st2 = st2.split()

            if "BW.DROMY" in config['seed2'] or "BW.ROMYT" in config['seed2']:

                # remove mean, trend and taper trace
                st1 = st1.detrend("linear").detrend("demean").taper(0.05)
                st2 = st2.detrend("linear").detrend("demean").taper(0.05)

                # set a filter for resampling
                # st1 = st1.filter("lowpass", freq=0.25, corners=4, zerophase=True)
                st1 = st1.filter("bandpass", freqmin=1e-4, freqmax=0.25, corners=4, zerophase=True)

                # st2 = st2.filter("lowpass", freq=0.25, corners=4, zerophase=True)
                st2 = st2.filter("bandpass", freqmin=1e-4, freqmax=0.25, corners=4, zerophase=True)

                # resampling
                st1 = st1.decimate(2, no_filter=True) ## 40 -> 20 Hz
                st1 = st1.decimate(2, no_filter=True) ## 20 -> 10 Hz
                st1 = st1.decimate(2, no_filter=True) ## 10 -> 5 Hz
                st1 = st1.decimate(5, no_filter=True) ## 5 -> 1 Hz
                st1 = st1.decimate(2, no_filter=True) ## 1 -> 0.5 Hz

                if "BW.DROMY" in config['seed2']:
                    st2 = st2.decimate(2, no_filter=True) ## 1 -> 0.5 Hz
                elif "BW.ROMYT" in config['seed2']:
                    st2 = st2.decimate(5, no_filter=True) ## 5 -> 1 Hz
                    st2 = st2.decimate(2, no_filter=True) ## 1 -> 0.5 Hz

                # convert tilt to acceleration
                for tr in st2:
                    tr.data = tr.data*9.81

            else:
                st1 = st1.detrend("linear").detrend("demean").taper(0.05)
                st2 = st2.detrend("linear").detrend("demean").taper(0.05)

                st1 = st1.filter("bandpass", freqmin=5e-4, freqmax=1, corners=4, zerophase=True)
                st2 = st2.filter("bandpass", freqmin=5e-4, freqmax=1, corners=4, zerophase=True)

                st1 = st1.resample(config.get('sampling_rate'), no_filter=True)
                st2 = st2.resample(config.get('sampling_rate'), no_filter=True)

            st1 = st1.merge()
            st2 = st2.merge()

            st1 = st1.trim(config['tbeg'], config['tend'], nearest_sample=True)
            st2 = st2.trim(config['tbeg'], config['tend'], nearest_sample=True)

            for tr in st1+st2:
                if len(tr.data) > 86401:
                    tr.data = tr.data[:86401]

        except Exception as e:
            print(f" -> pre-processing failed!")
            print(e)
            # continue
            error = True

        if len(st1) == 0:
            print(f" -> st1 empty!")
            # continue
            error = True

        if len(st2) == 0:
            print(f" -> st2 empty!")
            # continue
            error = True

        try:
            st1.plot(equal_scale=False, outfile=path_to_figs+f"all/st1_{st1[0].stats.channel}_all.png")
            st2.plot(equal_scale=False, outfile=path_to_figs+f"all/st2_{st2[0].stats.channel}_all.png")
        except:
            print(f" -> waveform plotting failed!")

        # prepare time intervals
        times = __get_time_intervals(config['tbeg'], config['tend'], config['interval_seconds'], config['interval_overlap'])

        # prepare psd parameters
        config['nperseg'] = int(config.get('sampling_rate')*config.get('tseconds'))
        config['noverlap'] = int(0.5*config.get('nperseg'))

        print(st1)
        print(st2)


        # ________________________________________________________________________________

        config['n_pds'] = int(config.get('tseconds')*config.get('sampling_rate'))

        default_f1 = zeros(int(config.get('n_pds'))+2)
        default_f2 = zeros(int(config.get('n_pds'))+2)
        default_ff_coh = zeros(int(config.get('n_pds'))+2)

        set_default = False

        # run operations for time intervals
        for n, (t1, t2) in enumerate(tqdm(times)):


            # trim streams for current interval
            _st1 = st1.copy().trim(t1, t2, nearest_sample=True)
            _st2 = st2.copy().trim(t1, t2, nearest_sample=True)

            # prepare data arrays
            if n == 0:
                if config['mode'] == "welch":
                    psds1 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    psds2 = zeros([len(times), int(config.get('nperseg')/2)+1])
                    cohs = zeros([len(times), int(config.get('nperseg')/2)+1])

                elif config['mode'] == "multitaper":
                    psds1 = zeros([len(times), int(config.get('n_pds'))+2])
                    psds2 = zeros([len(times), int(config.get('n_pds'))+2])
                    cohs = zeros([len(times), int(config.get('n_pds'))+2])

            print(psds1.shape)

            # check length
            if len(_st1) == 0 or len(_st2) == 0:
                error = True
                # continue

            # check for same length
            if not error:
                if len(_st1[0].data) != len(_st2[0].data):
                    print(f" -> size difference! {len(_st1[0].data)} != {len(_st2[0].data)}")
                    error = True
                    # continue

            # check if masked array
            if not error:
                if ma.is_masked(_st1[0].data) or ma.is_masked(_st2[0].data):
                    print(" -> masked array found")
                    mask_counter += 1
                    # continue
                    error = True

            try:
                _st1 = _st1.detrend("linear")
                _st2 = _st2.detrend("linear")

                # _st1 = _st1.filter("bandpass", freqmin=8e-4, freqmax=5, corners=4, zerophase=True)
                # _st2 = _st2.filter("bandpass", freqmin=8e-4, freqmax=5, corners=4, zerophase=True)

                _st1.plot(equal_scale=False, outfile=path_to_figs+f"{n}_st1_{st1[0].stats.channel}.png")
                _st2.plot(equal_scale=False, outfile=path_to_figs+f"{n}_st2_{st2[0].stats.channel}.png")
            except:
                error = True

            # compute power spectra
            if config['mode'] == "welch" and not error:

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

                # compute coherence
                ff_coh, coh = coherence(_st1[0].data,
                                        _st2[0].data,
                                        fs=_st2[0].stats.sampling_rate,
                                        window=config.get('taper'),
                                        nperseg=config.get('nperseg'),
                                        noverlap=config.get('noverlap')
                                       )

                cohs[n] = coh

            elif config['mode'] == "multitaper" and not error:

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
                    ff_coh, coh = Pxy.freq[:,0][:N//2+1], Pxy.cohe[:,0][:N//2+1]
                else:
                    print(_st1[0].data.size, _st2[0].data.size, psd1.size, psd2.size)
                    continue

                # update default arrays
                if not set_default:
                    print("set defaults")
                    print(len(f1))
                    default_f1 = f1
                    default_f2 = f2
                    default_ff_coh = ff_coh
                    set_default = True

            # set arrays to dummy or default if compuation failed due to error
            else:
                print("use defaults")
                print(f1)
                psd1 = zeros(int(config.get('n_pds'))+2)
                psd2 = zeros(int(config.get('n_pds'))+2)
                coh = zeros(int(config.get('n_pds'))+2)
                f1 = default_f1
                f2 = default_f2
                ff_coh = default_ff_coh

            # load maintenance file
            try:
                lxx = __load_lxx(t1, t2, archive_path)

                if lxx[lxx.sum_all.eq(1)].sum_all.size > 0 and not error:
                    print(f" -> maintenance period! Skipping...")
                    lxx_counter += 1
                    psd1, psd2, coh = psd1*nan, psd2*nan, coh*nan
            except:
                pass

            # check data quality
            max_num_of_bad_quality = 10

            if "BW.ROMY." in config['seed2'] and "Z" not in config['seed2'] and not error:
                try:
                    statusU = __load_status(t1, t2, "U", config['path_to_status_data'])
                    statusV = __load_status(t1, t2, "V", config['path_to_status_data'])
                except:
                    print(f" -> cannot load status file!")
                    continue

                if statusU[statusU.quality.eq(0)].quality.size > max_num_of_bad_quality:
                    print(f" -> U: bad quality status detected!")
                    mltiU_counter += 1
                    # psd1, psd2, coh = psd1*nan, psd2*nan, coh*nan
                    psd2, coh = psd2*nan, coh*nan

                if statusV[statusV.quality.eq(0)].quality.size > max_num_of_bad_quality:
                    print(f" -> V: bad quality status detected!")
                    mltiV_counter += 1
                    # psd1, psd2, coh = psd1*nan, psd2*nan, coh*nan
                    psd2, coh = psd2*nan, coh*nan

            if "BW.ROMY." in config['seed2'] and "Z" in config['seed2'] and not error:
                try:
                    statusZ = __load_status(t1, t2, "Z", config['path_to_status_data'])
                except:
                    print(f" -> cannot load status file!")
                    continue

                if statusZ[statusZ.quality.eq(0)].quality.size > max_num_of_bad_quality:
                    print(f" -> Z: bad quality status detected!")
                    mltiZ_counter += 1
                    # psd1, psd2, coh = psd1*nan, psd2*nan, coh*nan
                    psd2, coh = psd2*nan, coh*nan

            print(len(f1), len(psd1), len(f2), len(psd2))

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

        # save psds
        out1 = {}
        out1['frequencies'] = f1
        out1['psd'] = psds1

        __save_to_pickle(out1, config['outpath1'], f"{config['outname1']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        # save psds
        out2 = {}
        out2['frequencies'] = f2
        out2['psd'] = psds2

        __save_to_pickle(out2, config['outpath2'], f"{config['outname2']}_{str(date).split(' ')[0].replace('-','')}_hourly")

        # save coherence
        out3 = {}
        out3['frequencies'] = ff_coh
        out3['coherence'] = cohs

        __save_to_pickle(out3, config['outpath3'], f"{config['outname3']}_{str(date).split(' ')[0].replace('-','')}_hourly")

    print(f" MLTI-U count: {mltiU_counter}")
    print(f" MLTI-V count: {mltiV_counter}")
    print(f" MLTI-Z count: {mltiZ_counter}")

    print(f" Size count: {size_counter}")
    print(f" Mask count: {mask_counter}")
    print(f" LXX count: {lxx_counter}")

    print("\nDone\n")

# In[] ___________________________________________________________

if __name__ == "__main__":
    main(config)

    gc.collect()

# End of File
