#!/usr/bin/env python
# coding: utf-8

# ## Convert and Resample WROMY Data

import os
import pandas as pd
import logging

from tqdm import tqdm
from numpy import nan
from pathlib import Path
from obspy import UTCDateTime
from datetime import datetime

# ### Configurations


config = {}

# config['channel'] = "WS8"
config['channels'] = ["WS1","WS4","WS5","WS6","WS7","WS8","WS9"]

config['resample'] = 600 ## seconds

config['tbeg'] = UTCDateTime("2022-09-01")
config['tend'] = UTCDateTime("2023-05-31")

config['date_range']  = pd.date_range(config['tbeg'].date, config['tend'].date)

config['pathToData'] = f"/import/freenas-ffb-01-data/romy_archive/"
# config['pathToData'] = f"/home/andbro/Downloads/tmp/wromy/"

config['pathToOutput'] = f"/import/kilauea-data/wromy/"
# config['pathToOutput'] = f"/home/andbro/Downloads/tmp/wromy/"


# ### Methods


def __logging_setup(logpath, logfile):

    import logging

    logging.basicConfig(filename=f'{logpath}{logfile}', filemode='w', format='%(asctime)s, %(levelname)s, %(message)s')


def __read_wromy_data(config, date):
    '''
    reads data from T1 to T2
    '''
    doy = str(date.timetuple().tm_yday).rjust(3,"0")

    path = f"{config['pathToData']}{date.year}/BW/WROMY/{config['channel']}.D/"

    if not Path(path).exists():
        logging.error(f"Path: {path}, does not exists!")
        return None


    fileName = f"BW.WROMY.{config['channel']}.D.{date.year}.{doy}"

    try:
        df0 = pd.read_csv(path+fileName)
        ## replace error indicating values (-9999, 999.9) with NaN values
        df0.replace(to_replace=-9999, value=nan, inplace=True)
        df0.replace(to_replace=999.9, value=nan, inplace=True)

#             ## change time from in to 6 character string
        df0.iloc[:,2] = [str(ttt).rjust(6,"0") for ttt in df0.iloc[:,2]]

    except:
        logging.error(f"File: {fileName}, does not exists!")
        return None

    df0.reset_index(inplace=True, drop=True)

    ## add columns with total seconds

    if 'Seconds' in df0.columns:
        time_reference = datetime(2019,1,1)
        time_offset_seconds = (datetime(date.year,date.month,date.day) - time_reference).total_seconds()
        df0['totalSeconds'] = time_offset_seconds + df0['Seconds']

    return df0


# ### MAIN ####################################

__logging_setup(config['pathToOutput'], 'logfile.txt')

for cha in config['channels']:
    config['channel'] = cha
    logging.error(cha)

    for n, date in enumerate(tqdm(config['date_range'])):

        df_data = __read_wromy_data(config, date)

        if df_data is None:
            logging.error(f"skipping date: {date}")
            continue

        df_data['TimeStamp'] = pd.to_datetime(df_data['Date'].astype(str)+df_data['Time (UTC)'].astype(str))

        df_resampled = df_data.resample(f"{config['resample']/60}T", on="TimeStamp").mean()

        df_resampled.reset_index(inplace=True)

        df_resampled.pop("Date")
        df_resampled.pop("Seconds")

        if not os.path.isdir(config['pathToOutput']+config['channel']):
            os.mkdir(config['pathToOutput']+config['channel'])

        filename = config['pathToOutput']+config['channel']+"/"+str(config['channel'])+"_"+str(date)[:10].replace("-","")
#         logging.error(f" writing {filename}.pkl")
#         df_resampled.to_csv(filename+".csv")
        df_resampled.to_pickle(filename+".pkl")

print("Done")

## End Of File
