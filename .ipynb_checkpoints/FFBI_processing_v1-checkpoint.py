#!/bin/python3

'''
FFBI Processing

- load FFBI data
- remove sensitivity
- resample to 1 Hz
- store as streams

'''

import os
import sys
import obspy as obs
import numpy.ma as ma

from numpy import where
from andbro__read_sds import __read_sds


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


config = {}

config['path_to_sds'] = archive_path+"romy_archive/"

config['path_to_sds_out'] = archive_path+"temp_archive/"

config['path_to_inventory'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/"

if len(sys.argv) > 1:
    config['tbeg'] = obs.UTCDateTime(sys.argv[1])
config['tend'] = config['tbeg'] + 86400

config['sampling_rate'] = 20 # Hz

config['time_offset'] = 60 # seconds

config['t1'] = config['tbeg']-config['time_offset']
config['t2'] = config['tend']+config['time_offset']

config['Nexpected'] = int((config['t2'] - config['t1']) * config['sampling_rate'])





def __write_stream_to_sds(st, cha, path_to_sds):

    import os

    # check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    tr = st.select(channel=cha)[0]

    nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
    yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

    if not os.path.exists(path_to_sds+f"{yy}/"):
        os.mkdir(path_to_sds+f"{yy}/")
        print(f"creating: {path_to_sds}{yy}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/")
        print(f"creating: {path_to_sds}{yy}/{nn}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
        print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
        print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

    st.write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")

    print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")


def __get_trace(seed):

    from numpy import zeros

    net, sta, loc, cha = seed.split('.')

    trr = obs.Trace()
    trr.stats.starttime = config['t1']
    trr.data = zeros(config['Nexpected'])
    trr.stats.network = net
    trr.stats.station = sta
    trr.stats.location = loc
    trr.stats.channel = cha
    trr.stats.sampling_rate = config['sampling_rate']

    return trr


def main(config):

    # load inventory
    ffbi_inv = obs.read_inventory(root_path+"/Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml")

    # load data
    ffbi = __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDF", config['tbeg'], config['tend'])
    ffbi += __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDO", config['tbeg'], config['tend'])

    # check if merging is required
    if len(ffbi) != 2:
        print(f" -> merging required!")
        ffbi = ffbi.merge(fill_value="interpolate")

    # convert data
    for tr in ffbi:
        if "O" in tr.stats.channel:
            # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 100 mV/hPa
            tr.data = tr.data /1.0 /6.28099e5 /100e-3
        if "F" in tr.stats.channel:
            tr.remove_response(ffbi_inv, water_level=10)


        ffbi = ffbi.resample(1.0, no_filter=True)

    ffbi.merge(fill_value="interpolate")

    # ffbi.plot(equal_scale=False);

    ffbi = ffbi.resample(1.0, no_filter=False)

    # write output O
    out = obs.Stream()

    out += ffbi.select(component="O").copy()
    out.select(component="O")[0].stats.location = "30"
    out.select(component="O")[0].stats.channel = "LDO"

    # adjust time period
    out = out.trim(config['tbeg'], config['tend'], nearest_sample=False)

    # split into several traces since masked array cannot be stored as mseed
    out = out.split()

    __write_stream_to_sds(out, "LDO", config['path_to_sds_out'])

    # write output F
    out = obs.Stream()

    out += ffbi.select(component="F").copy()
    out.select(component="F")[0].stats.location = "30"
    out.select(component="F")[0].stats.channel = "LDF"

    # adjust time period
    out = out.trim(config['tbeg'], config['tend'], nearest_sample=False)

    # split into several traces since masked array cannot be stored as mseed
    out = out.split()

    __write_stream_to_sds(out, "LDF", config['path_to_sds_out'])


if __name__ == "__main__":
    main(config)

# End of File