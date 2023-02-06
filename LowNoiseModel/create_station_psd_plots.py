# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
__author__ = 'AndreasBrotzer'
__year__   = '2022'

##_____________________________________________________________
'''---- import libraries ----'''

from pandas import *
from os import listdir, uname, path

import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

##_____________________________________________________________
'''---- methods ----'''

def __read_spectra_nc(path, fname):

    print(f"\n -> reading {path}{fname}")

    f = nc.Dataset(str(path)+str(fname),'r')

    # for key in f.variables.keys():
    #        print(key)

    sta = f.variables["trace_id"][:][0].split(".")[1]
    loc = f.variables["trace_id"][:][0].split(".")[0]
    name = f"{sta}.{loc}"

    ff = f.variables["frequency"][:]
    ss = f.variables["spectrogram"][:]

    return name, ff, ss

def __makeplot_colorlines(config, ff, psds, station):

    from numpy import isnan, median, mean, std, array, zeros, linspace
    from scipy.stats import median_abs_deviation as mad
    
#     psds_minimal = __get_minimal_psd(array(psds))
    
        
    ##____________________________
    
    fig, ax = plt.subplots(1,1, figsize=(15,10), sharey=False, sharex=True)

    font = 12

    N = psds.shape[0]
    colors = plt.cm.rainbow(linspace(0, 1, N))

    psds_selected = []
    for n, psd in enumerate(psds):
        
        station_name = station.split(".")[0]
        if station_name in config['limits'].keys():
            limits = config['limits'][station_name]
        else:
            limits = [1e-20, 1e-11]
            
        if np.mean(psd) > limits[1] or np.mean(psd[-150:-15]) <  limits[0] or np.min(psd) < 1e-24:
            ax.loglog(ff, psd, color="grey", alpha=0.2, zorder=1)
        else:
            ax.loglog(ff, psd, color=colors[n], alpha=0.7, zorder=2)
            psds_selected.append(psd)
            
    ## turn list to array
    psds_selected = array(psds_selected)
    
    ## add scatter for colorbar object only
    for n, psd in enumerate(psds):
        p2 = ax.scatter(ff[0], psd[0], s=0., c=n, cmap='rainbow', vmin=0, vmax=N)

    ## calculate median of all psds
    psds_median = __get_median_psd(psds)
    ax.loglog(ff, psds_median, 'black', zorder=3, alpha=0.9)

    ## calculate median of selected psds
    psds_selected_median = __get_median_psd(psds_selected)
    ax.loglog(ff, psds_selected_median, 'white', zorder=3, alpha=0.9)
    

    ax.set_title(f"{station} ({len(psds_selected)})", fontsize=font)

    ax.set_xlabel("Frequency (Hz)", fontsize=font)

    ax.set_ylabel(r"PSD (rad$^2$/s$^2$/$Hz)$", fontsize=font)
    
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    ax.set_xlim(min(ff), max(ff))
    
    ## set colorbar at bottom
    cbar = fig.colorbar(p2, orientation='horizontal', ax=ax, aspect=50)

#     plt.show();

    return fig

def __get_minimal_psd(psds):

    from numpy import nanmin, array, nonzero, zeros
    
    min_psd = zeros(psds.shape[1])
    
    for f in range(psds.shape[1]):
        a = psds[:,f]
        min_psd[f] = nanmin(a[nonzero(a)])
    
    return min_psd

def __get_median_psd(psds):

    from numpy import median, zeros, isnan

    med_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        med_psd[f] = median(a[~isnan(a)])

    return med_psd


### CONFIGURATIONS ###

config = {}

config['hostname'] = uname()[1]

if config['hostname'] == 'kilauea':
    config['workdir'] = "/export/data/LNM/data/"
    config['outdir_figures'] = "/home/brotzer/Documents/ROMY/LowNoiseModel/figures/"    
elif config['hostname'] == 'lighthouse':
    config['workdir'] = "/home/andbro/kilauea-data/LNM/data/"
    config['outdir_figures'] = "/home/andbro/Documents/ROMY/LowNoiseModel/figures/"
else: 
    print(f"Hostname: {config['hostname']} not known!")


config['year'] = "2019"

    
config['love_phase_nc'] = "PHASE_VELOCITY_MODEL/LovePhaseVelocity.nc"
config['rayleigh_phase_nc'] = "PHASE_VELOCITY_MODEL/RayleighPhaseVelocity.nc"

config['datadir_spectra'] = config['workdir']+config['year']+"/"

config['datadir_stats'] = config['workdir']+"STATS/"

config['spectra_files'] = listdir(config['datadir_spectra'])

config['noise_models'] =  config['workdir']+"MODELS/""noise_models.npz"


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



def main(config):

    num = len(config['spectra_files'])

    for i, filename in enumerate(config['spectra_files']):

           
        name, ff ,ss = __read_spectra_nc(config['workdir']+config['year']+"/", filename)
        
        if path.isfile(f"/home/brotzer/Downloads/tmp/{name}.png"):
            print(" -> file already exists!")
            continue
            
        print(f" -> plotting ({i} / {num}) ...")
        fig = __makeplot_colorlines(config, ff, ss.T, name)

        print(" -> saving ...")
        fig.savefig(f"/home/brotzer/Downloads/tmp/{name}.png", transparent=False, dpi=300)



##_____________________________________________________________
if __name__ == "__main__":
    main(config)

##_____________________________________________________________
# End of File
