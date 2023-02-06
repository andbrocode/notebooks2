#!/usr/bin/env python

import os
import subprocess
import optparse
import matplotlib
matplotlib.use("agg")
from obspy import *
from obspy.core import AttribDict
import obspy.signal.array_analysis as AA
import obspy.signal.util as util
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
from obspy.signal.rotate import rotate2zne
from obspy.geodetics import gps2dist_azimuth
from obspy.clients.fdsn import Client
from obspy.clients.filesystem import sds
import scipy as sp
import scipy.odr as odr
import math

"""
USAGE: george_adr_post.py <path2strore>

"""

def getParameters(path2store):
    par = AttribDict()
    par.path2store = path2store
    par.freqmin = 0.01
    return par


###################################################################
# Set parameters here:
###################################################################
def run(time, par):
    start = time
    end = time + (24*60 * 60 ) # 24h 
    print(start,"-",end)

# Data Base
    sdsclient = sds.Client(sds_root="/bay200/mseed_online/archive")
    cl = Client(base_url="http://jane",timeout=300)
    array_stations = ['GR.FUR.HH*','BW.FFB1.HH*','BW.FFB2.HH*','BW.FFB3.HH*']

    subarray = [0,1,2,3]

    res = []


    tsz = []
    tsn = []
    tse = []
    coo = []
    first = True
    for station in array_stations:
        net,sta,stream = station.split(".")
        stats = sdsclient.get_waveforms(network=net,station=sta,location="",channel=stream, starttime=start-10, endtime=end+10)
        inventory = cl.get_stations(network=net,station=sta,location="", channel=stream,level='response')
        stats.merge(method=1,fill_value="latest")
        stats.attach_response(inventory)
        stats.remove_sensitivity()
        stats.sort()
        stats.reverse()
        stats.trim(start,end)
        stats.rotate(method="->ZNE",inventory=inventory,components=['Z21'])

        print(stats)

        l_lon = float(inventory[0][0][0].longitude)
        l_lat = float(inventory[0][0][0].latitude)
        height = float(inventory[0][0][0].elevation) - float(inventory[0][0][0].depth)
        stats.resample(sampling_rate=20)
        fs = stats[0].stats.sampling_rate
        stats.detrend("linear")

        stats.detrend("simple")
        stats[0].filter('highpass',freq=par.freqmin)
        stats[1].filter('highpass',freq=par.freqmin)
        stats[2].filter('highpass',freq=par.freqmin)


        if first:
            first = False
            o_lon = l_lon
            o_lat = l_lat
            o_height = height


        lon,lat = util.util_geo_km(o_lon,o_lat,l_lon,l_lat)
        coo.append([lon*1000,lat*1000,height-o_height])


        tsz.append(stats[0].data)
        tsn.append(stats[1].data)
        tse.append(stats[2].data)


    ttse  =  np.array(tse)
    ttsn  =  np.array(tsn)
    ttsz  =  np.array(tsz)

    subarray = np.array(subarray)
    vp = 1000.
    vs = 560.
    sigmau = 0.0000001
    result = AA.array_rotation_strain(subarray, np.transpose(ttse), np.transpose(ttsn), np.transpose(ttsz), vp, vs, np.array(coo), sigmau)

    rotz = result['ts_w3']
    rotn = result['ts_w2']
    rote = result['ts_w1']

    straine = result['ts_e'][:,0,0]
    strainn = result['ts_e'][:,1,1]
    strainz = result['ts_e'][:,2,2]
    strainv = result['ts_d']

    rots = stats.copy()
    rots[0].stats.station = "ROMY"
    rots[0].stats.channel = "BJZ"
    rots[0].stats.location = "20"
    rots[0].data = rotz
    rots[1].stats.station = "ROMY"
    rots[1].stats.channel = "BJN"
    rots[1].stats.location = "20"
    rots[1].data = rotn
    rots[2].stats.station = "ROMY"
    rots[2].stats.channel = "BJE"
    rots[2].stats.location = "20"
    rots[2].data = rote
    rots.detrend("simple")

    strains = stats.copy()
    rots += strains
    add = stats[0].copy() 
    rots += add
    rots[3].stats.station = "ROMY"
    rots[3].stats.channel = "BSZ"
    rots[3].stats.location = "20"
    rots[3].data = strainz
    rots[4].stats.station = "ROMY"
    rots[4].stats.channel = "BSN"
    rots[4].stats.location = "20"
    rots[4].data = strainn
    rots[5].stats.station = "ROMY"
    rots[5].stats.channel = "BSE"
    rots[5].stats.location = "20"
    rots[5].data = straine
    rots[6].stats.station = "ROMY"
    rots[6].stats.channel = "BV"
    rots[6].stats.location = "20"
    rots[6].data = strainv

    rots.detrend("simple")


    # and the parameter periods_per_window
    rots.trim(start,end)

    myday = str(rots[0].stats.starttime.julday)

    pathyear = str(rots[0].stats.starttime.year)
    # open catalog file in read and write mode in case we are continuing d/l,
    # so we can append to the file
    mydatapath = os.path.join(par.path2store, pathyear)

    # create datapath 
    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)

    mydatapath = os.path.join(mydatapath, rots[0].stats.network)
    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)

    mydatapath = os.path.join(mydatapath, rots[0].stats.station)

    # create datapath 
    if not os.path.exists(mydatapath):
                os.mkdir(mydatapath)

    for i,tr in enumerate(rots):
        print(rots)
        mydatapathchannel = os.path.join(mydatapath,rots[i].stats.channel + ".D")

        if not os.path.exists(mydatapathchannel):
            os.mkdir(mydatapathchannel)

        netFile = rots[i].stats.network + "." + rots[i].stats.station +  "." + rots[i].stats.location + "." + rots[i].stats.channel+ ".D." + pathyear + "." + myday
        netFileout = os.path.join(mydatapathchannel, netFile)

        # for post processing we overwrite existing files
        #netFileout = open(netFileout, 'w')

        # header of the stream object which contains the output of the ADR
        tr.write(netFileout , format='MSEED', encoding="FLOAT64")


def main():
    parser = optparse.OptionParser()
    (options, args) = parser.parse_args()
    # if no time is specified, use now (rounded to hour) - 3 hours
    # assume we do this 4 hr in the moring
    if len(args) == 1:
        t = UTCDateTime()
        t = UTCDateTime(t.year, t.month, t.day-1)
        times = [t]
    elif len(args) == 2:
        t = UTCDateTime(args[1])
        times = [t]
    elif len(args) == 3:
        t1 = int(UTCDateTime(args[1]).timestamp)
        t2 = int(UTCDateTime(args[2]).timestamp)
        times = [UTCDateTime(t) for t in np.arange(t1, t2, 24*3600)]
    else:
        parser.print_usage()
        return

    path2store = args[0]

    par = getParameters(path2store)
  #  templates = returnTemplates(par.filter.freqmin, par.filter.freqmax,
  #                              par.filter.corners, par.filter.zerophase)
  #  par.coincidence.event_templates = templates
  #  par.coincidence.similarity_threshold = 0.5

    for t in times:
        try:
            run(t, par)
        except:
            print("failed to compute ADR")
            continue


if __name__ == '__main__':
        main()

