#!/bin/python3

import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from obspy import UTCDateTime, read_inventory
from scipy.signal import welch

from functions.get_fband_average import __get_fband_average
from functions.get_median_psd import __get_median_psd
from functions.compute_backazimuth_noise import __compute_backazimuth_noise
from functions.rotate_romy_ZUV_ZNE import __rotate_romy_ZUV_ZNE
from functions.get_time_intervals import __get_time_intervals

from andbro__read_sds import __read_sds
from andbro__save_to_pickle import __save_to_pickle


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





## ---------------------------------------

config = {}


config['station1'] = "BW.ROMY.10.BJZ"
config['station2'] = "GR.FUR..BHN"

config['tbeg'] = UTCDateTime("2023-09-23 00:00")
config['tend'] = UTCDateTime("2023-09-30 00:00")

config['path_to_sds1'] = archive_path+"romy_archive/"
config['path_to_sds2'] = bay_path+f"mseed_online/archive/"

config['path_to_figures'] = data_path+f"VelocityChanges/figures/"

config['path_to_inv'] = root_path+"Documents/ROMY/stationxml_ringlaser/"

config['path_to_data_out'] = data_path+f"VelocityChanges/data/"


times = __get_time_intervals(config['tbeg'], config['tend'], interval_seconds=3600, interval_overlap=0)

baz_tangent = []
baz_rayleigh = []
baz_love = []

baz_tangent_std = []
baz_rayleigh_std = []
baz_love_std = []

for t1, t2 in tqdm(times):

    try:
        inv1 = read_inventory(config['path_to_inv']+"dataless.seed.BW_ROMY")
        inv2 = read_inventory(config['path_to_inv']+"dataless.seed.GR_FUR")

        st1 =  __read_sds(config['path_to_sds1'], "BW.ROMY.10.BJZ", t1, t2);
        st1 += __read_sds(config['path_to_sds1'], "BW.ROMY..BJU", t1, t2);
        st1 += __read_sds(config['path_to_sds1'], "BW.ROMY..BJV", t1, t2);


        st2 =  __read_sds(config['path_to_sds2'], "GR.FUR..BHZ", t1, t2);
        st2 += __read_sds(config['path_to_sds2'], "GR.FUR..BHN", t1, t2);
        st2 += __read_sds(config['path_to_sds2'], "GR.FUR..BHE", t1, t2);

        st1.remove_sensitivity(inv1);
        st2.remove_response(inv2, output="ACC", water_level=10);

        st1 = __rotate_romy_ZUV_ZNE(st1, inv1)
    except:
        print(f" -> data loading failed !")
        continue


    st1.detrend("linear");
    st2.detrend("linear");

    acc = st2.copy();
    rot = st1.copy();

    fmin, fmax = 1/10, 1/7

    acc = acc.detrend("linear");
    acc = acc.taper(0.01);
    acc = acc.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True);

    rot = rot.detrend("linear");
    rot = rot.taper(0.01);
    rot = rot.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True);

    conf = {}

    conf['eventtime'] = config['tbeg']

    conf['tbeg'] = t1
    conf['tend'] = t2

    conf['station_longitude'] = 11.275501
    conf['station_latitude']  = 48.162941

    ## specify window length for baz estimation in seconds
    conf['win_length_sec'] = 2/fmin

    ## define an overlap for the windows in percent (50 -> 50%)
    conf['overlap'] = 50

    ## specify steps for degrees of baz
    conf['step'] = 1

    try:
        out = __compute_backazimuth_noise(rot, acc, None, fmin, fmax, cc_thres=0.2, plot=False);
    except:
        print(f" -> baz computation failed!")
        continue

    baz_tangent.append(out['baz_tangent_max'])
    baz_rayleigh.append(out['baz_rayleigh_max'])
    baz_love.append(out['baz_love_max'])

    baz_tangent_std.append(out['baz_tangent_std'])
    baz_rayleigh_std.append(out['baz_rayleigh_std'])
    baz_love_std.append(out['baz_love_std'])

out = {}
out['baz_tangent'] = np.array(baz_tangent)
out['baz_rayleigh'] = np.array(baz_rayleigh)
out['baz_love'] = np.array(baz_love)
out['baz_tangent_std'] = np.array(baz_tangent_std)
out['baz_rayleigh_std'] = np.array(baz_rayleigh_std)
out['baz_love_std'] = np.array(baz_love_std)

__save_to_pickle(out, config['path_to_data_out'], f"VC_BAZ_{config['tbeg'].date}_{config['tend'].date}")

## End of File