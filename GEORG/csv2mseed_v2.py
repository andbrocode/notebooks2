#!/usr/bin/env python
# coding: utf-8

# # Convert GEORG CSV to Mseed

import os
import gc
import csv

from tqdm import tqdm
from pandas import read_csv
from obspy import UTCDateTime
from functions.get_stream import __get_stream
from functions.write_stream_to_sds import __write_stream_to_sds
from functions.read_sds import __read_sds


def write_csv(path, filename, data, header=None):

    if not os.path.isdir(path+"tmp/"):
        os.mkdir(path+"tmp/")

    with open(path+"tmp/"+filename, 'w') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def split_csv(pathname, filename, num_rows, has_header=True):
    name, extension = filename.split('.')
    file_no = 1
    chunk = []
    row_count = 0
    header = ''

    with open(pathname+filename, 'r') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            if has_header:
                header = row
                has_header = False
                continue
            chunk.append(row)
            row_count += 1
            if row_count > num_rows:
                print(f"writing {pathname}tmp/{name}-{file_no}.{extension}...")
                write_csv(f'{pathname}', f'{name}-{file_no}.{extension}', chunk, header)
                chunk = []
                file_no += 1
                row_count = 0
        if chunk:
            write_csv(f'{pathname}', f'{name}-{file_no}.{extension}', chunk, header)

# ### Configurations

# path_to_data = "/home/andbro/kilauea-data/sagnac_frequency/bonn/"
path_to_data = "./"

path_to_sds = "./"

# filename = "4h_GEORG_Data.csv"
filename = "Ringlaser26-12_00-00-00_27-12_00-00-00_MEZ.csv"

sps = 7000

starttime = "2024-12-26 00:00"

seed_code = "XX.GEORG..FJZ"

chunksize = 7000*1*1800 # equals 2 hours of data

# split large csv file into smaller ones
split_csv(path_to_data, filename, chunksize, has_header=False)

# read smaller csv files and add it to mseed day file
for n, filename in enumerate(sorted(os.listdir(path_to_data))):

    print(n, filename, starttime)

    # read csv data to dataframe
    df = read_csv(path_to_data+filename, names=["time", "data"])

    # create stream
    st = __get_stream(df['data'].values, seed_code, starttime, sps=sps)

    print(st)

        for tr in st:

            # downsample to reduce memory
            tr = tr.resample(3500, no_filter=True)

            # scaling up to store integer instead of floats (reduce memory)
            tr.data = array([int(x*1e6) for x in tr.data])

    # write data as mseed to SDS archive
    __write_stream_to_sds(st, path_to_sds)

    del st, df
    gc.collect()

    # update starttime
    starttime = UTCDateTime(starttime) + chunksize / sps

