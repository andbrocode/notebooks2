# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__   = '2022'

##_____________________________________________________________
'''---- import libraries ----'''

from obspy import *
from pandas import *
from os import listdir

import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
##_____________________________________________________________
''' ---- set variables ---- '''

config = {}

config['workdir'] = "/export/data/LNM/data/"

config['year'] = "2019"

config['datadir_spectra'] = config['workdir']+config['year']+"/"

config['spectra_files'] = listdir(config['datadir_spectra'])

config['outdir'] = config['workdir']+"STATS/"

config['limits'] = {
    "GNI":  [1e-20, 1e-12],
    "KDAK": [1e-20, 1e-9],    
    "RER":  [1e-20, 1e-9],
    "TIXI": [1e-18, 1e-8],    
    "GUMO": [1e-19, 1e-9],    
    "WAKE": [1e-18, 1e-8],    
    "RAO":  [1e-21, 1e-9],
    "AIS":  [1e-20, 1e-9],    
    "TARA": [1e-20, 1e-8],
    "MAKZ": [1e-20, 1e-9],    
    "DAV":  [1e-19, 1e-11],    
    "FOMA": [1e-19, 1e-11],
    "CTAO": [1e-19, 1e-11],    
    "KIP":  [1e-20, 1e-9],    
    "TATO": [1e-20, 1e-9],    
    "TRIS": [1e-20, 1e-10],    
    "RAO":  [1e-20, 1e-9],    
    "FUNA": [1e-19, 1e-9],    
    "MIDW": [1e-17, 1e-9],    
    "MPG":  [1e-19, 1e-10],    
    "HKT":  [1e-20, 1e-12],    
    "KWJN": [1e-20, 1e-9],    
    "TARA": [1e-19, 1e-8],    
    "ROCAM":[1e-20, 1e-9],    
    "PAYG": [1e-18, 1e-11],    
    "JOHN": [1e-20, 1e-9],    
    "OTAV": [1e-19, 1e-11],    
    "PPTF": [1e-20, 1e-9],    
    "YSS":  [1e-19, 1e-11],    
    "RAR":  [1e-20, 1e-9],    
    "KMBO": [1e-18, 1e-12],    
    "ADK":  [1e-19, 1e-10],    
    "BORG": [1e-20, 1e-8],    
    "COCO": [1e-20, 1e-8],    
    "DGAR": [1e-20, 1e-9],    
    "PTCN": [1e-19, 1e-8],    
    "BILL": [1e-20, 1e-10],    
    "CMLA": [1e-20, 1e-10],    
    "BRKV": [1e-20, 1e-12],    
    "PAF":  [1e-19, 1e-9],    
}

##_____________________________________________________________
'''---- define methods ----'''


def __read_spectra_nc(path, fname):

    print(f"\nreading {path}{fname}")

    f = nc.Dataset(str(path)+str(fname),'r')

    # for key in f.variables.keys():
    #        print(key)

    sta = f.variables["trace_id"][:][0].split(".")[1]
    loc = f.variables["trace_id"][:][0].split(".")[0]
    name = f"{sta}.{loc}"

    ff = f.variables["frequency"][:]
    ss = f.variables["spectrogram"][:]

    return name, ff, ss


def __convert_to_decibel(psds):

    from numpy import zeros, count_nonzero, ones, nan, log10, sqrt, isinf, array, mean, min

    psds_db = zeros(psds.shape)
    rows_with_zeros = 0
    for i, psd in enumerate(psds):
        if count_nonzero(psd) != len(psd):
            rows_with_zeros +=1
            psd = [nan for val in psd if val == 0]
        psds_db[i,:] = 10*log10(psd)
        if isinf(psds_db[i,:]).any():
            psds_db[i,:] = nan * ones(len(psds_db[i,:]))
    if rows_with_zeros != 0:
        print(f" -> found {rows_with_zeros} rows with zeros!")

    return psds_db


def __get_stats(arr, axis=0):

    from numpy import array
    
    med, men, std = [],[],[]
    nan_found = False

    for fcross in range(arr.shape[axis]):
        if axis == 0:
            data = arr[fcross,:]
        elif axis == 1:
            data = arr[:,fcross]
        if np.isnan(data).any():
            nan_found = True
            data = data[~np.isnan(data)]
        men.append(np.mean(data))
        med.append(np.median(data))
        std.append(np.std(data))

    if nan_found:
        print(" -> NaN values were detected and ignored!")

    return array(med), array(men), array(std)

def __filter_psds(psds, name, config):
    
    from numpy import array, mean, min
    
    station_name = name.split(".")[0]
    
    if station_name in config['limits'].keys():
        print(f"special limits for {station_name}")
        limits = config['limits'][station_name]
    else:
        limits = [1e-20, 1e-11]    
        
    psds_selected = []
    for n, psd in enumerate(psds):
        if mean(psd) > limits[1] or mean(psd[-150:-15]) <  limits[0] or min(psd) < 1e-24:
            break
        else:
            psds_selected.append(psd)

    return array(psds_selected)


def main(config):


    path = config['datadir_spectra']
    # fname = "ESACCI-SEASTATE-L2-MSSPEC-IU_YAK_LHZ_00_2018-fv01.nc"

    medians = DataFrame()
    means = DataFrame()
    deviations = DataFrame()


    for i, fname in enumerate(config['spectra_files']):

        sta, ff, ss = __read_spectra_nc(path, fname)
        
        ss_selected = __filter_psds(ss.T, sta, config)
        
       
        if ss_selected.shape[0] == 0: 
            print("ss_selected is empty! Aborting ...")
            continue
        else:
            print(f"selected: {ss_selected.shape[0]} psds from {ss.shape[1]}")
            
        ss_db = __convert_to_decibel(ss_selected.T)

#         med, men, std = __get_stats(ss_selected.T, axis=0)
        med, men, std = __get_stats(ss_db, axis=0)


        if i == 0:
            medians['frequencies'] = ff
            means['frequencies'] = ff
            deviations['frequencies'] = ff

        medians[sta] = med
        means[sta] = men
        deviations[sta] = std

    ## write dataframes to files as pickle
    print(f"writing data to {config['outdir']+config['year']}")
    medians.to_pickle(config['outdir']+config['year']+"_medians.pkl")
    means.to_pickle(config['outdir']+config['year']+"_means.pkl")
    deviations.to_pickle(config['outdir']+config['year']+"_deviations.pkl")


##_____________________________________________________________
if __name__ == "__main__":
    main(config)

##_____________________________________________________________
# End of File
