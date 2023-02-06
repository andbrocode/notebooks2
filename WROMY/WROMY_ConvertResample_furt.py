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

config['channel'] = "FURT"

config['resample'] = 600 ## seconds

config['tbeg'] = UTCDateTime("2021-09-01")
config['tend'] = UTCDateTime("2022-08-31")

config['date_range']  = pd.date_range(config['tbeg'].date, config['tend'].date)

config['pathToData'] = f"/import/freenas-ffb-01-data/romy_archive/"
# config['pathToData'] = f"/home/andbro/Downloads/tmp/wromy/"

config['pathToOutput'] = f"/import/kilauea-data/wromy/"
# config['pathToOutput'] = f"/home/andbro/Downloads/tmp/wromy/"


# ### Methods


def __logging_setup(logpath, logfile):

    import logging

    logging.basicConfig(filename=f'{logpath}{logfile}', filemode='w', format='%(asctime)s, %(levelname)s, %(message)s')


def __read_furt_data(config, date, path_to_archive = '/bay200/gif_online/FURT/WETTER/'):
    '''
    Load a selection of data of FURT weather station for time period
    
    PARAMETERS:
        - config:    configuration dictionary
        - show_raw:  bool (True/False) -> shows raw data FURT head

    RETURN:
        - dataframe
        
    '''
    
    from pathlib import Path
    from obspy import UTCDateTime
    from tqdm.notebook import tqdm_notebook
   
    if not Path(path_to_archive).exists():
        logging.error(f" -> Path: {path_to_archive}, does not exists!")
        return None
    
    ## declare empyt dataframe
    df = pd.DataFrame()
            
    date = UTCDateTime(date)
    filename = f'FURT.WSX.D.{str(date.day).rjust(2,"0")}{str(date.month).rjust(2,"0")}{str(date.year).rjust(2,"0")[-2:]}.0000'

    try:

        df0 = pd.read_csv(path_to_archive+filename, usecols=[0,1,10,12,13,14], names=['date', 'time', 'T', 'H', 'P','Rc'])            
        ## substitute strings with floats
        df0['T']  = df0['T'].str.split("=", expand=True)[1].str.split("C", expand=True)[0].astype(float)
        df0['P']  = df0['P'].str.split("=", expand=True)[1].str.split("H", expand=True)[0].astype(float)
        df0['H']  = df0['H'].str.split("=", expand=True)[1].str.split("P", expand=True)[0].astype(float)
        df0['Rc'] = df0['Rc'].str.split("=", expand=True)[1].str.split("M", expand=True)[0].astype(float)


        ## replace error indicating values (-9999, 999.9) with NaN values
        df0.replace(to_replace=-9999, value=nan, inplace=True)
        df0.replace(to_replace=999.9, value=nan, inplace=True)


        if df.empty:
            df = df0
        else: 
            df = pd.concat([df, df0])
    except:
        logging.error(f"  -> File: {filename}, does not exists!")
        return None

    ## add seconds properly 
    times = [str(t).rjust(6,"0") for t in df['time']]
    df['Seconds'] = [int(t[:2])*60*24+int(t[2:4])*60+int(t[4:6]) for t in times]
    
    ## add total seconds
    time_reference = datetime(2019,1,1)
    time_offset_seconds = (datetime(date.year,date.month,date.day) - time_reference).total_seconds()
    df['totalSeconds'] = time_offset_seconds + df['Seconds']
    
    ## add timestamp
    df['TimeStamp'] = pd.to_datetime([datetime(date.year, date.month, date.day, int(t[:2]),int(t[2:4]),int(t[4:6]))  for t in times])
   
    df.reset_index(inplace=True, drop=True)
        
    return df


# ### MAIN ####################################

__logging_setup(config['pathToOutput'], 'logfile_furt.txt')


for n, date in enumerate(tqdm(config['date_range'])):

#         df_data = __read_wromy_data(config, date)
    df_data = __read_furt_data(config, date)

    if df_data is None:
        logging.error(f"skipping date: {date}")
        continue

    df_resampled = df_data.resample(f"{config['resample']/60}T", on="TimeStamp").mean()

    df_resampled.reset_index(inplace=True)

    ## clean up dataframe
    df_resampled.pop("date")
    df_resampled.pop("time")
    df_resampled.pop("Seconds")

    if not os.path.isdir(config['pathToOutput']+config['channel']):
        os.mkdir(config['pathToOutput']+config['channel'])

    filename = config['pathToOutput']+config['channel']+"/"+str(config['channel'])+"_"+str(date)[:10].replace("-","")
#     logging.error(f" writing {filename}.pkl")
#     df_resampled.to_csv(filename+".csv")
    df_resampled.to_pickle(filename+".pkl")

print("Done")

## End Of File
