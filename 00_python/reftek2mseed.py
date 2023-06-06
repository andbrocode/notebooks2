#!/usr/bin/env python
# coding: utf-8

# ## REFTEK 2 MSEED



import os
import obspy

from numpy import nan, sort
from numpy.ma import filled, isMaskedArray

import warnings
warnings.filterwarnings('ignore')




ipath = input("\nEnter path: ")

reftek = input("\nEnter REFTEK ID (e.g. 9E52): ")

networkcode = input("\nEnter network code (e.g. BW): ")

stationname = input("\nEnter station name (e.g. ROMY): ")

channelcode = input("\nEnter channel code (e.g. BHZ,BHN,BHE): ")

amount_of_channels = int(input("\nEnter amount of channels (default 3): ") or 3)

channelcode = channelcode.split(",")




import subprocess

if ipath[-1] == "/":
    ipath = ipath[:-1]

## prepare directories for writing
if not os.path.isdir(ipath+"/mseed"):
    for i in range(3):
        os.makedirs(ipath+f"/mseed/{channelcode[i]}.D")

## get list of recorded days
# days = !ls $ipaths
#days = os.listdir(ipath)
#days = [x for x in os.listdir(ipath) if x not in ['mseed']]
days = [x for x in os.listdir(ipath) if len(x) == 7]

days = sort(days)

## loop over days as stored by REFTEK
for day in days:

    print(f"\nprocessing {day} ...\n")

    ## extract year and doy
    year = day[0:4]
    doy  = day[4:]

    ## read REFTEK raw files
    st = obspy.read(ipath+f"/{day}/{reftek}/1/*")

    st.merge()
    
    ## check if merging worked
    if len(st) < amount_of_channels:
        print("seems like a channel is missing !")
    if len(st) > amount_of_channels:
        print("seems like merging failed !")

    npts=[]

    if len(st) == amount_of_channels:

        ## loop over channels for writing
        for i in range(amount_of_channels):

            ## add meta data
            st[i].stats.network = networkcode
            st[i].stats.station = stationname
            st[i].stats.channel = channelcode[i]

            ## get amount of samples for channel i
            npts.append(st[i].stats.npts)

            ## check if any channel is masked and add NaN values if this is the case
            if isMaskedArray(st[i].data):
                print(f"masked array {st[i].stats.channel[-1]} filled with 0")
                st[i].data = filled(st[i].data, fill_value=0)
                st
            ## write data as MSEED format
            st[i].write(ipath+f"/mseed/{channelcode[i]}.D/{networkcode}.{stationname}..{channelcode[i]}.D.{year}.{doy}", format="MSEED")

        ## check amount of samples across channels
        if npts[0] != npts[1] or npts[0] != npts[2]:
            print(f"Number of samples in channels do not match: 1:{npts[0]} 2:{npts[1]} 3:{npts[2]}")

    else:
        print(f"error for day: {day}")

print("\nDONE\n")

## END OF FILE
