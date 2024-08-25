#!/bin/python3

from andbro__write_stream_to_sds import __write_stream_to_sds
from andbro__read_sds import __read_sds
from obspy import UTCDateTime
from pandas import date_range

seed = input("Enter seed: ")

date1 = input("Enter date1: ")

date2 = input("Enter date2: ")

outpath = input("Enter outpath: ")


dates = date_range(date1, date2)

path_to_archive1 = "/import/freenas-ffb-01-data/romy_archive/"
path_to_archive2 = "/bay200/mseed_online/archive/"


for date in dates:

    print("")
    print(UTCDateTime(date).date)

    try:
        st = __read_sds(path_to_archive1, seed, UTCDateTime(date), UTCDateTime(date)+86400)
    except:
        print(f" -> failed to load from {path_to_archive1}!")

    try:
        st = __read_sds(path_to_archive2, seed, UTCDateTime(date), UTCDateTime(date)+86400)
    except:
        print(f" -> failed to load from {path_to_archive2}!")

    print(st)
    
    try:
        __write_stream_to_sds(st, outpath)
    except:
        print(f" -> failed to write data!")

# End of File