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

from andbro__read_sds import __read_sds
from andbro__save_to_pickle import __save_to_pickle

## ---------------------------------------


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

# config['station1'] = "BW.ROMY.10.BJZ"
# config['station2'] = "GR.FUR..BHN"

# if len(sys.argv) > 1:
#     config['tbeg'] = UTCDateTime(sys.argv[1])
#     config['tend'] = config['tbeg'] + 86400
# else:
#     config['tbeg'] = UTCDateTime("2023-09-23 00:00")
#     config['tend'] = UTCDateTime("2023-09-23 01:00")

config['tbeg'] = UTCDateTime("2023-09-23 00:00")
config['tend'] = UTCDateTime("2023-09-23 01:00")

config['path_to_sds1'] = archive_path+"romy_archive/"

config['path_to_sds2'] = bay_path+f"mseed_online/archive/"

config['path_to_figures'] = data_path+f"VelocityChanges/figures/autoplots/"

config['path_to_inv'] = root_path+"Documents/ROMY/stationxml_ringlaser/"

config['path_to_data_out'] = data_path+f"VelocityChanges/data/"

config['fmin'], config['fmax'] = 1/10, 1/7

config['cc_threshold'] = 0.5

config['interval_seconds'] = 1800

config['window_overlap'] = 90

config['window_length_sec'] = 2/config['fmin']

## ---------------------------------------


times = __get_time_intervals(config['tbeg'], config['tend'], interval_seconds=config['interval_seconds'], interval_overlap=0)

baz_tangent = []
baz_rayleigh = []
baz_love = []

baz_tangent_std = []
baz_rayleigh_std = []
baz_love_std = []

vel_love_max = []
vel_love_std = []
vel_rayleigh_std = []
vel_rayleigh_std = []

baz_bf = []
baz_bf_std = []

vel_bf = []

ttime = []

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
        pass

## ---------------------------------------

    try:
        st1.detrend("linear");
        st2.detrend("linear");

        acc = st2.copy();
        rot = st1.copy();


        acc = acc.detrend("linear");
        acc = acc.taper(0.01);
        acc = acc.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True);

        rot = rot.detrend("linear");
        rot = rot.taper(0.01);
        rot = rot.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True);

    except:
        print(f" -> processing failed !")
        pass

## ---------------------------------------

    conf = {}

    conf['eventtime'] = config['tbeg']

    conf['tbeg'] = t1
    conf['tend'] = t2

    conf['station_longitude'] = 11.275501
    conf['station_latitude']  = 48.162941

    ## specify window length for baz estimation in seconds
    conf['win_length_sec'] = config['window_length_sec']

    ## define an overlap for the windows in percent (50 -> 50%)
    conf['overlap'] = config['window_overlap']

    ## specify steps for degrees of baz
    conf['step'] = 1

    conf['path_to_figs'] = config['path_to_figures']

    conf['cc_thres'] = config['cc_threshold']

    try:
        out = __compute_backazimuth_and_velocity_noise(conf, rot, acc, config['fmin'], config['fmax'], plot=False, save=True);

        baz_tangent.append(out['baz_tangent_max'])
        baz_rayleigh.append(out['baz_rayleigh_max'])
        baz_love.append(out['baz_love_max'])

        baz_tangent_std.append(out['baz_tangent_std'])
        baz_rayleigh_std.append(out['baz_rayleigh_std'])
        baz_love_std.append(out['baz_love_std'])

        vel_love_max.append(out['vel_love_max'])
        vel_love_std.append(out['vel_love_std'])
        vel_rayleigh_std.append(out['vel_rayleigh_max'])
        vel_rayleigh_std.append(out['vel_rayleigh_std'])

        ttime.append(t1)

    except Exception as e:
        print(e)
        print(f" -> baz computation failed!")

        baz_tangent.append(np.nan)
        baz_rayleigh.append(np.nan)
        baz_love.append(np.nan)

        baz_tangent_std.append(np.nan)
        baz_rayleigh_std.append(np.nan)
        baz_love_std.append(np.nan)

        vel_love_max.append(np.nan)
        vel_love_std.append(np.nan)
        vel_rayleigh_std.append(np.nan)
        vel_rayleigh_std.append(np.nan)

        ttime.append(t1)

    try:
        out_bf = __compute_beamforming_ROMY(
                                            conf['tbeg'],
                                            conf['tend'],
                                            submask=None,
                                            fmin=config['fmin'],
                                            fmax=config['fmax'],
                                            component="Z",
                                            bandpass=True,
                                            plot=False
                                           )

        baz_bf.append(out_bf['baz_bf_max'])
        baz_bf_std.append(out_bf['baz_bf_std'])
        vel_bf.append(out_bf['slow'])


    except Exception as e:
        print(e)
        print(f" -> baz computation failed!")

        baz_bf.append(np.nan)
        baz_bf_std.append(np.nan)
        vel_bf.append(np.nan)



## ---------------------------------------


## prepare output dictionary
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

## store output dictionary
__save_to_pickle(output, config['path_to_data_out'], f"VC_BAZ_{config['tbeg'].date}_{config['tend'].date}")

## store plot
out['fig3'].savefig(config['path_to_figures']+f"VC_BAZ_{config['tbeg'].date}_{config['tend'].date}.png", format="png", dpi=150, bbox_inches='tight')
print(f" -> stored: {config['path_to_figures']}VC_BAZ_{config['tbeg'].date}_{config['tend'].date}.png")

## End of File