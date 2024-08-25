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
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'


# ## Configurations

config = {}

config['path_to_outdata'] = input("Enter path for output data: ")

config['tbeg'] = obs.UTCDateTime(input("Enter starttime: "))
config['tend'] = obs.UTCDateTime(input("Enter endtime: "))

config['path_to_sds'] = archive_path+"romy_archive/"

config['path_to_inventory'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/"

## BSPF coordinates
config['ROMY_lon'] = 11.275501
config['ROMY_lat'] = 48.162941



# ## Load Data


# #### Load inventory

romy_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.BW_ROMY")
seis_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.BW_DROMY")


### Load Seismometer Data


rot = obs.Stream()

rot += __read_sds(config['path_to_sds'], "BW.ROMY.10.BJZ", config['tbeg']-1, config['tend']+1)
rot += __read_sds(config['path_to_sds'], "BW.ROMY..BJU", config['tbeg']-1, config['tend']+1)
rot += __read_sds(config['path_to_sds'], "BW.ROMY..BJV", config['tbeg']-1, config['tend']+1)


if len(rot) > 3:
    print(" -> merging required")
    rot = rot.merge(fill_value="interpolate")


## remove ring laser sensitivtiy
rot = rot.remove_sensitivity(romy_inv)

# detrend
rot = rot.detrend("demean")

# rotate data to ZNE
rot = __rotate_romy_ZUV_ZNE(rot, romy_inv, keep_z=True)


### Load Seismometer Data

acc = __read_sds(config['path_to_sds'], "BW.DROMY..HH*", config['tbeg']-1, config['tend']+1)

# remove seismometer response
acc.remove_response(seis_inv, output="ACC")

## detrend
acc.detrend("demean")

if len(acc) > 3:
    print(" -> merging required")
    acc = acc.merge(fill_value="interpolate")


### Write Data to SDS

__write_stream_to_sds(rot, config['path_to_outdata'])

__write_stream_to_sds(acc, config['path_to_outdata'])


## End of File