#!/usr/bin/env python
# coding: utf-8

# # Convert GEORG CSV to Mseed

import os

from pandas import read_csv
from obspy import UTCDateTime
from functions.get_stream import __get_stream
from functions.write_stream_to_sds import __write_stream_to_sds
from functions.read_sds import __read_sds

# ### Configurations

# path_to_data = "/home/andbro/kilauea-data/sagnac_frequency/bonn/"
path_to_data = "/import/kilauea-data/GEORG/tmp/"

path_to_data_out = "/import/kilauea-data/GEORG/data/"

# filename = "4h_GEORG_Data.csv"
filename = "Ringlaser26-12_00-00-00_27-12_00-00-00_MEZ.csv"

sps = 7000

starttime = "2024-12-26 00:00"

seed_code = "XX.GEORG..FJZ"

chunksize = 7000*2*3600

for n, filename in enumerate(sorted(os.listdir(path_to_data))):

    print(n, filename, starttime)

    if n < 0:
        continue

    # read csv data to dataframe
    df = read_csv(path_to_data+filename, names=["time", "data"])

    # create stream
    st = __get_stream(df['data'].values, seed_code, starttime, sps=sps)

    if n != 0:
        st += __read_sds(path_to_data_out, seed_code, "2024-12-26 00:00", "2024-12-27 00:00")

    st.merge()
    print(st)

    # write data as mseed to SDS archive
    __write_stream_to_sds(st, path_to_data_out)

    # update starttime
    starttime = UTCDateTime(starttime) + chunksize / sps

    del st
