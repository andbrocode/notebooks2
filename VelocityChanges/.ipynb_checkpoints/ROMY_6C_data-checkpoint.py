#!/usr/bin/env python
# coding: utf-8

# # Get 6C Data for ROMY


import os
import obspy as obs
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client

from andbro__read_sds import __read_sds
from andbro__write_stream_to_sds import __write_stream_to_sds

from functions.rotate_romy_ZUV_ZNE import __rotate_romy_ZUV_ZNE


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


# ## Configurations

config = {}

config['path_to_outdata'] = input("Enter path for output data: ")

config['tbeg'] = obs.UTCDateTime(input("Enter starttime: "))
config['tend'] = obs.UTCDateTime(input("Enter endtime:   "))

config['seis'] = input("Select Seismometer: [FUR] or DROMY") or "FUR"

config['onet'] = input("Enter output network [XX]: ") or "XX"
config['osta'] = input("Enter output station [VROMY]: ") or "VROMY"
config['oloc'] = input("Enter output location []: ") or ""

config['path_to_sds'] = archive_path+"romy_archive/"

config['path_to_fur'] = bay_path+"mseed_online/archive/"

config['path_to_inventory'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/"

## BSPF coordinates
config['ROMY_lon'] = 11.275501
config['ROMY_lat'] = 48.162941



# ## Load Data


# #### Load inventory

romy_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.BW_ROMY")

if "FUR" in config['seis']:
    seis_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.GR_FUR")

elif "DROMY" in config['seis']:
    seis_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.BW_DROMY")


### Load Seismometer Data


rot = obs.Stream()

rot += __read_sds(config['path_to_sds'], "BW.ROMY.10.BJZ", config['tbeg']-60, config['tend']+60)
rot += __read_sds(config['path_to_sds'], "BW.ROMY..BJU", config['tbeg']-60, config['tend']+60)
rot += __read_sds(config['path_to_sds'], "BW.ROMY..BJV", config['tbeg']-60, config['tend']+60)


if len(rot) > 3:
    print(" -> merging required")
    rot = rot.merge(fill_value="interpolate")


## remove ring laser sensitivtiy
rot = rot.remove_sensitivity(romy_inv)

# detrend
rot = rot.detrend("demean")

# rotate data to ZNE
rot = __rotate_romy_ZUV_ZNE(rot, romy_inv, keep_z=True)

rot = rot.trim(config['tbeg'], config['tend'])

for tr in rot:
    tr.stats.network = config['onet']
    tr.stats.station = config['osta']
    tr.stats.location = config['oloc']

# print(rot)

### Load Seismometer Data


if "FUR" in config['seis']:
    acc = __read_sds(config['path_to_fur'], "GR.FUR..BH*", config['tbeg']-60, config['tend']+60)

elif "DROMY" in config['seis']:
    acc = __read_sds(config['path_to_sds'], "BW.DROMY..BH*", config['tbeg']-60, config['tend']+60)

# remove seismometer response
acc.remove_response(seis_inv, output="ACC")

## detrend
acc.detrend("demean")

if len(acc) > 3:
    print(" -> merging required")
    acc = acc.merge(fill_value="interpolate")

acc = acc.trim(config['tbeg'], config['tend'])

for tr in acc:
    tr.stats.network = config['onet']
    tr.stats.station = config['osta']
    tr.stats.location = config['oloc']

# print(acc)

### Write Data to SDS

__write_stream_to_sds(rot, config['path_to_outdata'])

__write_stream_to_sds(acc, config['path_to_outdata'])


## End of File