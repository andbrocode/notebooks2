#!/bin/python3

import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from obspy import UTCDateTime, read_inventory

from functions.rotate_romy_ZUV_ZNE import __rotate_romy_ZUV_ZNE
from functions.get_time_intervals import __get_time_intervals
from functions.compute_beamforming_ROMY import __compute_beamforming_ROMY
from functions.compute_backazimuth_and_velocity_noise import __compute_backazimuth_and_velocity_noise
from functions.load_lxx import __load_lxx
from functions.read_sds import __read_sds

# ______________________________________________________

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

# ______________________________________________________

config = {}

# argument for date is expected
if len(sys.argv) > 1:
    config['tbeg'] = UTCDateTime(sys.argv[1])
    config['tend'] = config['tbeg'] + 86400
else:
    print(f" -> missing date argument (e.g. 2024-09-01)!")
    sys.exit()

# config['tbeg'] = UTCDateTime("2023-09-20 21:00")
# config['tend'] = UTCDateTime("2023-09-20 22:00")

config['path_to_sds1'] = archive_path+"temp_archive/"

config['path_to_sds2'] = bay_path+f"mseed_online/archive/"

config['path_to_figures'] = data_path+f"VelocityChanges/figures/autoplots/"

config['path_to_figures_status'] = data_path+f"VelocityChanges/figures/autoplots_status/"

config['path_to_inv'] = root_path+"Documents/ROMY/stationxml_ringlaser/"

config['path_to_data_out'] = data_path+f"VelocityChanges/data/"

# set maximum number of MLTI in time interval. Otherwise skip interval.
config['num_mlti'] = 3

# set frequency band
config['fmin'], config['fmax'] = 1/11, 1/6 # 1/10, 1/7

# set cross-correlation threshold
config['cc_threshold'] = 0.5 # 0.5

# set interval and overlap of data ( in seconds )
config['interval_seconds'] = 3600 # 1800
config['interval_overlap'] = 0

# set window lenght for computations ( in seconds )
config['window_length_sec'] = 3/config['fmin']  #2/config['fmin']

# set window overlap for computations ( in seconds )
config['window_overlap'] = 75 # 90

# set sampling rate
config['sps'] = 20 # Hz

# expected samples
config['samples'] = config['sps'] * config['interval_seconds']

# get size of arrays
config['arr_size'] = (config['interval_seconds'] * config['sps']) // (int(config['sps'] * config['window_length_sec']))

# ______________________________________________________

def __load_mlti(tbeg, tend, ring, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    year = tbeg.year

    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    path_to_mlti = path_to_archive+f"romy_archive/{year}/BW/CROMY/{year}_romy_{rings[ring]}_mlti.log"

    mlti = read_csv(path_to_mlti, names=["time_utc","Action","ERROR"])

    mlti = mlti[(mlti.time_utc > tbeg) & (mlti.time_utc < tend)]

    return mlti

def __store_as_pickle(object, name):

    import os, pickle

    ofile = open(name+".pkl", 'wb')
    pickle.dump(object, ofile)

    if os.path.isfile(name+".pkl"):
        print(f"created: {name}.pkl")

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

def __to_array(arr_in):

    arr_out = []

    for _t in arr_in:
        _t = np.array(_t)
        if _t.size > 1:
            for _x in _t:
                arr_out.append(_x)
        elif _t.size == 1:
            arr_out.append(np.nan)

    return np.array(arr_out)

# ______________________________________________________

def main(config):

    # prepare time intervals for loop
    times = __get_time_intervals(config['tbeg'],
                                 config['tend'],
                                 interval_seconds=config['interval_seconds'],
                                 interval_overlap=config['interval_overlap']
                                )

    # set sample size
    NN = len(times)

    # prepare dummy trace
    dummy_size = config['arr_size']
    nan_dummy = np.ones(dummy_size)*np.nan

    # prepare arrays
    baz_tangent = np.ones(NN)*np.nan
    baz_rayleigh = np.ones(NN)*np.nan
    baz_love = np.ones(NN)*np.nan

    baz_tangent_std = np.ones(NN)*np.nan
    baz_rayleigh_std = np.ones(NN)*np.nan
    baz_love_std = np.ones(NN)*np.nan

    baz_tangent_all = np.ones([NN, dummy_size])*np.nan
    baz_rayleigh_all = np.ones([NN, dummy_size])*np.nan
    baz_love_all = np.ones([NN, dummy_size])*np.nan
    baz_bf_all = np.ones([NN, dummy_size])*np.nan

    cc_tangent_all = np.ones([NN, dummy_size])*np.nan
    cc_rayleigh_all = np.ones([NN, dummy_size])*np.nan
    cc_love_all = np.ones([NN, dummy_size])*np.nan

    vel_love_max = np.ones(NN)*np.nan
    vel_love_std = np.ones(NN)*np.nan
    vel_rayleigh_max = np.ones(NN)*np.nan
    vel_rayleigh_std = np.ones(NN)*np.nan

    vel_rayleigh_all = np.ones([NN, dummy_size])*np.nan
    vel_love_all = np.ones([NN, dummy_size])*np.nan
    vel_bf_all = np.ones([NN, dummy_size])*np.nan

    times_relative = np.ones([NN, dummy_size])*np.nan
    times_all = np.ones([NN, dummy_size])*np.nan

    baz_bf = np.ones(NN)*np.nan
    baz_bf_std = np.ones(NN)*np.nan

    vel_bf = np.ones(NN)*np.nan
    time_bf = np.ones(NN)*np.nan

    ttime = []
    ttime_bf = []

    num_stations_used = np.ones(NN)*np.nan

    # prepare status variable
    status = np.zeros((2, len(times)))

    # ______________________________________________________

    for _n, (t1, t2) in enumerate(tqdm(times)):

        # print(_n, t1, t2)

        # load maintenance file
        lxx = __load_lxx(t1, t2, archive_path)

        try:
            print(f"\nloading data ...")

            # inv1 = read_inventory(config['path_to_inv']+"dataless/dataless.seed.BW_ROMY")
            # inv2 = read_inventory(config['path_to_inv']+"dataless/dataless.seed.GR_FUR")

            inv1 = read_inventory(config['path_to_inv']+"station_BW_ROMY.xml")
            inv2 = read_inventory(config['path_to_inv']+"station_GR_FUR.xml")

            st1 = __read_sds(config['path_to_sds1'], "BW.ROMY.30.BJZ", t1, t2);
            st1 += __read_sds(config['path_to_sds1'], "BW.ROMY.30.BJN", t1, t2);
            st1 += __read_sds(config['path_to_sds1'], "BW.ROMY.30.BJE", t1, t2);

            st2 = __read_sds(config['path_to_sds2'], "GR.FUR..BHZ", t1, t2);
            st2 += __read_sds(config['path_to_sds2'], "GR.FUR..BHN", t1, t2);
            st2 += __read_sds(config['path_to_sds2'], "GR.FUR..BHE", t1, t2);

            # remove sensitivity for ROMY
            # st1.remove_sensitivity(inv1);

            # remove response for FUR
            st2 = st2.remove_response(inv2, output="ACC", water_level=60);

            # get length of streams
            N_Z = len(st1.select(channel="*Z"))
            N_N = len(st1.select(channel="*N"))
            N_E = len(st1.select(channel="*E"))

            # check if merging is necessary
            if len(st1) > 3:
                print(f" -> merging required: rot")
                # st1 = st1.merge(fill_value="interpolate")
                st1 = st1.merge(fill_value=0)

            if len(st2) > 3:
                print(f" -> merging required: acc")
                # st2 = st2.merge(fill_value="interpolate")
                st2 = st2.merge(fill_value=0)

            print(st1)
            print(st2)

            # check if data has same length
            Nexpected = int((t2 - t1)*config['sps'])
            for tr in st1:
                Nreal = len(tr.data)
                if Nreal != Nexpected:
                    tr.data = tr.data[:Nexpected]
                    # print(f" -> adjust length: {tr.stats.station}.{tr.stats.channel}:  {Nreal} -> {Nexpected}")
            for tr in st2:
                Nreal = len(tr.data)
                if Nreal != Nexpected:
                    tr.data = tr.data[:Nexpected]
                    # print(f" -> adjust length: {tr.stats.station}.{tr.stats.channel}:  {Nreal} -> {Nexpected}")

            # get amplitude levels
            levels = {}
            for tr in st1:
                ring = tr.stats.channel[-1]
                levels[ring] = np.percentile(abs(tr.data), 90)

            # rotate UVZ to ZNE
            # st1 = __rotate_romy_ZUV_ZNE(st1, inv1, keep_z=True);

        except Exception as e:
            print(f" -> data loading failed!")
            print(e)

        # ______________________________________________________
        # pre-processing

        try:
            st1 = st1.detrend("linear");
            st2 = st2.detrend("linear");

            acc = st2.copy();
            rot = st1.copy();

            acc = acc.detrend("linear");
            acc = acc.taper(0.01, type="cosine");
            acc = acc.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=8, zerophase=True);

            rot = rot.detrend("linear");
            rot = rot.taper(0.01, type="cosine");
            rot = rot.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=8, zerophase=True);

        except:
            print(f" -> processing failed !")

        try:
            # check if data is all zero
            for tr in rot:
                if np.count_nonzero(tr.data) == 0:
                    print(f" -> all zero: {tr.stats.station}.{tr.stats.channel}")
            for tr in acc:
                if np.count_nonzero(tr.data) == 0:
                    print(f" -> all zero: {tr.stats.station}.{tr.stats.channel}")

            # rot.plot(equal_scale=False);
            # acc.plot(equal_scale=False);

            print(rot)
            print(acc)
        except:
            pass

        for tr in rot+acc:
            if tr.stats.npts != config['samples']:
                print(f"shorter {tr.stats.npts} != {config['samples']}")

        # ______________________________________________________
        # configurations

        conf = {}

        conf['eventtime'] = config['tbeg']

        conf['tbeg'] = t1
        conf['tend'] = t2

        conf['station_longitude'] = 11.275501
        conf['station_latitude'] = 48.162941

        # specify window length for baz estimation in seconds
        conf['win_length_sec'] = config['window_length_sec']

        # define an overlap for the windows in percent (50 -> 50%)
        conf['overlap'] = config['window_overlap']

        # specify steps for degrees of baz
        conf['step'] = 1

        conf['path_to_figs'] = config['path_to_figures']

        conf['cc_thres'] = config['cc_threshold']

        # ______________________________________________________
        # compute backatzimuths

        try:
            print(f"\ncompute backazimuth estimation ...")
            out = __compute_backazimuth_and_velocity_noise(conf, rot, acc,
                                                           config['fmin'], config['fmax'],
                                                           plot=False,
                                                           save=True
                                                          );
            # change status to success
            status[0, _n] = 1
            baz_computed = True

        except Exception as e:
            print(f" -> baz computation failed!")
            baz_computed = False
            print(e)

        # ______________________________________________________
        # check for MTLI launches

        try:
            print(f"\ncheckup for MLTI ...")

            # check maintenance periods
            maintenance = lxx[lxx.sum_all.eq(1)].sum_all.size > 0

            if N_N > 1 or N_E > 1 or levels["N"] > 1e-6 or levels["E"] > 1e-6 or maintenance:
                print(" -> to many MLTI (horizontal)")
                baz_computed = False

            if N_Z > 1 or levels["Z"] > 1e-6 or maintenance:
                print(" -> to many MLTI (vertical)")
                baz_computed = False

        except Exception as e:
            print(f" -> chekup failed!")
            print(e)

        # ______________________________________________________
        # compute beamforming for array

        try:
            print(f"\ncompute beamforming ...")

            out_bf = __compute_beamforming_ROMY(
                                                conf['tbeg'],
                                                conf['tend'],
                                                submask="outer",
                                                fmin=config['fmin'],
                                                fmax=config['fmax'],
                                                component="Z",
                                                bandpass=True,
                                                plot=False
                                               )

            # change status to success
            status[1, _n] = 1
            bf_computed = True

        except Exception as e:
            print(f" -> beamforming computation failed!")
            print(e)
            bf_computed = False

        # ______________________________________________________
        # assign values to arrays

        # always assign time values
        # ttime[_n] = t1
        # ttime_bf[_n] = t1
        ttime.append(t1)
        ttime_bf.append(t1)

        print(ttime)

        if baz_computed:

            try:
                baz_tangent[_n] = out['baz_tangent_max']
                baz_tangent_std[_n] = out['baz_tangent_std']

                baz_rayleigh[_n] = out['baz_rayleigh_max']
                baz_rayleigh_std[_n] = out['baz_rayleigh_std']

                vel_rayleigh_max[_n] = out['vel_rayleigh_max']
                vel_rayleigh_std[_n] = out['vel_rayleigh_std']

                baz_love[_n] = out['baz_love_max']
                baz_love_std[_n] = out['baz_love_std']

                vel_love_max[_n] = out['vel_love_max']
                vel_love_std[_n] = out['vel_love_std']
            except:
                pass

            try:
                baz_love_all[_n] = out['baz_love_all']
                baz_rayleigh_all[_n] = out['baz_rayleigh_all']
                baz_tangent_all[_n] = out['baz_tangent_all']

                vel_love_all[_n] = out['vel_love_all']
                vel_rayleigh_all[_n] = out['vel_rayleigh_all']

                cc_tangent_all[_n] = out['cc_tangent_all']
                cc_love_all[_n] = out['cc_love_all']
                cc_rayleigh_all[_n] = out['cc_rayleigh_all']

                times_relative[_n] = out['times_relative']

                times_absolute = [t1 + float(_t) for _t in out['times_relative']]
                times_all[_n] = times_absolute

            except Exception as e:
                print(f" -> failed to assign ({dummy_size}) != {len(out['times_relative'])}")
                print(e)
                pass


        if bf_computed:

            try:
                baz_bf[_n] = out_bf['baz_bf_max']
                baz_bf_std[_n] = out_bf['baz_bf_std']

                num_stations_used[_n] = out_bf['num_stations_used']
            except Exception as e:
                print(e):
                pass

            try:
                vel_bf_all[_n] = out_bf['slow']
                baz_bf_all[_n] = out_bf['baz']

                times_abs = np.array([t1 + int(_t) for _t in out_bf['time']])
                time_bf[_n] = times_abs
            except Exception as e:
                print(e)
                pass

        # ______________________________________________________
        # store plots

        try:
            print(f"\ncreate plot ...")

            t1_t2 = f"{t1.date}_{str(t1.time).split('.')[0]}_{t2.date}_{str(t2.time).split('.')[0]}"
            out['fig3'].savefig(config['path_to_figures']+f"VC_BAZ_{t1_t2}.png",
                                format="png", dpi=100, bbox_inches='tight')
            print(f" -> stored: {config['path_to_figures']}VC_BAZ_{t1_t2}.png")

        except Exception as e:
            print(f" -> plotting failed!")
            print(e)

        print("\n_______________________________________________\n")

    # ______________________________________________________
    # reshape arrays
    def reshaping(_arr):
        return _arr.reshape(NN*dummy_size)

    # ______________________________________________________
    # prepare output dictionary

    output = {}

    output['time'] = np.array(ttime)
    output['baz_tangent'] = np.array(baz_tangent)
    output['baz_rayleigh'] = np.array(baz_rayleigh)
    output['baz_love'] = np.array(baz_love)
    output['baz_tangent_std'] = np.array(baz_tangent_std)
    output['baz_rayleigh_std'] = np.array(baz_rayleigh_std)
    output['baz_love_std'] = np.array(baz_love_std)
    output['baz_bf'] = np.array(baz_bf)
    output['baz_bf_std'] = np.array(baz_bf_std)

    output['vel_bf'] = np.array(vel_bf)
    output['vel_love_max'] = np.array(vel_love_max)
    output['vel_rayleigh_max'] = np.array(vel_rayleigh_max)

    output['vel_love_std'] = np.array(vel_love_std)
    output['vel_rayleigh_std'] = np.array(vel_rayleigh_std)

    output['num_stations_used'] = num_stations_used

    # ______________________________________________________
    # store output to file

    print(f"-> store: {config['path_to_data_out']}statistics/VC_BAZ_{config['tbeg'].date}.pkl")
    __store_as_pickle(output, config['path_to_data_out']+"statistics/"+f"VC_BAZ_{config['tbeg'].date}")

    # ______________________________________________________
    # prepare output dictionary 1

    output1 = {}

    output1['time'] = np.array(times_all)
    output1['time_bf'] = np.array(time_bf)

    output1['baz_tangent_all'] = reshaping(baz_tangent_all)
    output1['baz_rayleigh_all'] = reshaping(baz_rayleigh_all)
    output1['baz_love_all'] = reshaping(baz_love_all)
    output1['baz_bf_all'] = reshaping(baz_bf_all)

    output1['cc_tangent_all'] = reshaping(cc_tangent_all)
    output1['cc_rayleigh_all'] = reshaping(cc_rayleigh_all)
    output1['cc_love_all'] = reshaping(cc_love_all)

    output1['vel_rayleigh_all'] = reshaping(vel_rayleigh_all)
    output1['vel_love_all'] = reshaping(vel_love_all)
    output1['vel_bf_all'] = reshaping(vel_bf_all)

    # ______________________________________________________
    # store output to file

    print(f"-> store: {config['path_to_data_out']}all/VC_BAZ_{config['tbeg'].date}_all.pkl")
    __store_as_pickle(output1, config['path_to_data_out']+"all/"+f"VC_BAZ_{config['tbeg'].date}_all")

    # ______________________________________________________
    # status plot

    import matplotlib.colors
    cmap = matplotlib.colors.ListedColormap(['red', 'green'])

    fig = plt.figure()

    c = plt.pcolormesh(np.arange(0, status.shape[1]), ["BAZ", "BF"], status, edgecolors='k', linewidths=1, cmap=cmap)

    fig.savefig(config['path_to_figures_status']+f"VC_BAZ_{config['tbeg'].date}_status.png",
                format="png", dpi=100, bbox_inches='tight')

    print(f" -> stored: {config['path_to_figures_status']}VC_BAZ_{config['tbeg'].date}.png")

    del fig

    print("\n")

# ______________________________________________________

if __name__ == "__main__":
    main(config)

# End of File
