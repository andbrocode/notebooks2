#!/usr/bin/env python
# coding: utf-8

# # Automatic BSPF Eventplots

# Creates automatic event plots based on catalog 


## _______________________________________

import os 
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from andbro__savefig import __savefig
from tqdm.notebook import tqdm

from functions.request_data import __request_data
from functions.add_distances_and_backazimuth import __add_distances_and_backazimuth

## _______________________________________

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'


## _______________________________________
## Configurations

config = {}


## location of BSPF
config['BSPF_lon'] = -116.455439
config['BSPF_lat'] = 33.610643

## path for figures to store
config['outpath_figs'] = data_path+"BSPF/figures/triggered_all/"

## blueSeis sensor (@200Hz)
config['seed_blueseis'] = "PY.BSPF..HJ*"

## Trillium 240 next to BlueSeis on Pier (@40Hz)
config['seed_seismometer1'] = "II.PFO.10.BH*" 

## STS2 next to BlueSeis (@200Hz)
config['seed_seismometer2'] = "PY.PFOIX..HH*" 

config['path_to_catalog'] = data_path+"BSPF/data/catalogs/"
config['catalog'] = "BSPF_catalog_20221001_20230615_triggered.pkl"


## _______________________________________


def __process_xpfo(config, st, inv):

    ii_pfo = st.copy()

    ## demean
    ii_pfo.detrend("demean")

    ## taper 
    ii_pfo.taper(0.1)

    ## bandpass
    ii_filter = ii_pfo.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True)

    ## adjust channel names
    for tr in ii_pfo:
        if tr.stats.channel[-1] == "1":
            tr.stats.channel = str(tr.stats.channel).replace("1","N")
        if tr.stats.channel[-1] == "2":
            tr.stats.channel = str(tr.stats.channel).replace("2","E")
    
    return ii_pfo



def __makeplot(config, st):

    
    st_in = st.copy()
    
    fig, ax = plt.subplots(6,1, figsize=(15,10), sharex=True)

    font = 14

    time_scaling, time_unit = 1, "sec"
    rot_scaling = 1e9
    trans_scaling = 1e6
    
    for i, tr in enumerate(st_in):
        
        if i in [0,1,2]:
            ax[i].set_ylabel(r"$\omega$ (nrad/s)", fontsize=font)
            ax[i].plot(tr.times()/time_scaling, tr.data*rot_scaling, 'k', label=tr.stats.station+"."+tr.stats.channel)

        elif i in [3,4,5]:
            ax[i].set_ylabel(r"u ($\mu$m/s)", fontsize=font)
            ax[i].plot(tr.times()/time_scaling, tr.data*trans_scaling, 'k', label=tr.stats.station+"."+tr.stats.channel)
            
        ax[i].legend(loc=1)
        
    ax[5].set_xlabel(f"Time ({time_unit}) from {st[0].stats.starttime.date} {str(st[0].stats.starttime.time).split('.')[0]} UTC", fontsize=font)
    ax[0].set_title(config['title']+f" | {config['fmin']} - {config['fmax']} Hz", fontsize=font, pad=10)
    
#     plt.show();
    del st_in
    return fig



def __makeplotStreamSpectra2(st, config, fscale=None):

    from scipy import fftpack
    from andbro__fft import __fft
    import matplotlib.pyplot as plt

    st_in = st.copy()
    
    NN = len(st_in)
    rot_scaling, rot_unit = 1e9, r"nrad/s"
    trans_scaling, trans_unit = 1e6, r"$\mu$m/s"
        
    fig, axes = plt.subplots(NN,2,figsize=(15,int(NN*2)), sharex='col')

    font = 14
    
    plt.subplots_adjust(hspace=0.3)

    ## _______________________________________________

    st.sort(keys=['channel'], reverse=True)
    
    for i, tr in enumerate(st_in):

#         comp_fft = abs(fftpack.fft(tr.data))
#         ff       = fftpack.fftfreq(comp_fft.size, d=1/tr.stats.sampling_rate)
#         comp_fft = fftpack.fftshift(comp_fft)
#         ff, spec = ff[1:len(ff)//2], abs(fftpack.fft(tr.data)[1:len(ff)//2])

        if tr.stats.channel[-2] == "J":
            scaling = rot_scaling
        elif tr.stats.channel[-2] == "H":
            scaling = trans_scaling

        spec, ff, ph = __fft(tr.data*scaling, tr.stats.delta, window=None, normalize=None)
    
    
        ## _________________________________________________________________
        if tr.stats.channel[-2] == "J":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*rot_scaling,
                        color='black',
                        label='{} {}'.format(tr.stats.station, tr.stats.channel),
                        lw=1.0,
                        )

        elif tr.stats.channel[-2] == "H":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*trans_scaling,
                        color='black',
                        label='{} {}'.format(tr.stats.station, tr.stats.channel),
                        lw=1.0,
                        )
        ## _________________________________________________________________
        if fscale == "loglog":
            axes[i,1].loglog(ff, spec, color='black', lw=1.0)
        elif fscale == "loglin":
            axes[i,1].semilogx(ff, spec, color='black', lw=1.0)
        elif fscale == "linlog":
            axes[i,1].semilogy(ff, spec, color='black', lw=1.0)
        else:
            axes[i,1].plot(ff, spec, color='black', lw=1.0)         
        
        
        if tr.stats.channel[1] == "J":
            sym, unit = r"$\Omega$", rot_unit
        elif tr.stats.channel[1] == "H":
            sym, unit = "v", trans_unit
        else:
            unit = "Amplitude", "a.u."
            
        axes[i,0].set_ylabel(f'{sym} ({unit})',fontsize=font)    
        axes[i,1].set_ylabel(f'ASD \n({unit}/Hz)',fontsize=font)        
        axes[i,0].legend(loc='upper left',bbox_to_anchor=(0.8, 1.10), framealpha=1.0)
        
#         axes[i,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         axes[i,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    if "fmin" in config.keys() and "fmax" in config.keys():
        axes[i,1].set_xlim(config['fmin'],config['fmax'])

    axes[NN-1,0].set_xlabel(f"Time from {tr.stats.starttime.date} {str(tr.stats.starttime.time)[:8]} (s)",fontsize=font)     
    axes[NN-1,1].set_xlabel(f"Frequency (Hz)",fontsize=font)     

    del st_in
    return fig



## _______________________________________
## Event Info

events = pd.read_pickle(config['path_to_catalog']+config['catalog'])

events.reset_index(inplace=True)

events.rename(columns = {'index':'origin'}, inplace = True)



# __add_distances_and_backazimuth(config['tbeg'], config['tend'], events)




## _______________________________________
## RUN LOOP



for jj, ev in enumerate(tqdm(events.index)):
# for jj, ev in enumerate([0,1]):
    
    print(f" -> {jj} {events.origin[jj]} ")
    
    ## configurations
    config['title'] = f"{events.origin[jj]} UTC | M{events.magnitude[jj]}"
    config['tbeg'] = obs.UTCDateTime(str(events.origin[jj]))
    config['fmin'], config['fmax'] = 0.02, 18.0
    
    
    if config['tbeg'].date < obs.UTCDateTime("2023-04-01"):
        config['seed_seismometer'] = config['seed_seismometer1']
    else:
        config['seed_seismometer'] = config['seed_seismometer2']
        
        
    if events.distances_km[jj] < 30:
        config['tend'] = obs.UTCDateTime(events.origin[jj])+30
    elif events.distances_km[jj] > 30 and events.distances_km[jj] < 100:
        config['tend'] = obs.UTCDateTime(events.origin[jj])+60
    else:
        config['tend'] = obs.UTCDateTime(events.origin[jj])+180

    
    ## load and process blueSeis data
    try:
        py_bspf0, py_bspf_inv = __request_data(config['seed_blueseis'], config['tbeg'], config['tend'])
        
#         py_bspf0.merge(fill_value="interpolate")
#         py_bspf0.trim(config['tbeg'], config['tend'])
#         py_bspf0.remove_sensitivity(py_bspf_inv)
#         py_bspf0.detrend('demean')
#         py_bspf0.resample(40)
        
    except Exception as e:
        print(f" -> failed to request BSPF for event: {ev}")
        print(e)
        continue
        
        
    ## load and process seismometer data
    try:        
        ii_pfo0, ii_pfo_inv = __request_data(config['seed_seismometer'], config['tbeg'], config['tend'])
        
#         ii_pfo0.merge(fill_value="interpolate")
#         ii_pfo0.trim(config['tbeg'], config['tend'])
#         ii_pfo0.remove_response(inventory=ii_pfo_inv, output="ACC", plot=False)
#         ii_pfo0.detrend('demean')
        
    except Exception as e:
        print(f" -> failed to request PFOX for event: {ev}")   
        print(e)
        continue

    ## processing data
#     py_bspf = __process_bspf(config, py_bspf0, py_bspf_inv)
#     ii_pfo = __process_xpfo(config, ii_pfo0, ii_pfo_inv)
    
    
    if ii_pfo0[0].stats.sampling_rate != py_bspf0[0].stats.sampling_rate:
        py_bspf0.resample(ii_pfo0[0].stats.sampling_rate)
    
    
    ## joining data
    st0 = py_bspf0
    st0 += ii_pfo0    
    
    st = st0.copy() 
    st.detrend("linear")
    st.taper(0.01)
    st.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True)
    
    event_name = str(events.origin[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]
    
    ## plotting
    
    fig1 = st0.plot(equal_scale=False, show=False);

    fig2 = __makeplot(config, st)
    
    fig3 = __makeplotStreamSpectra2(st, config, fscale="linlin");
    
    ## saving figures
    fig1.savefig(config['outpath_figs']+"raw/"+f"{event_name}_raw.png", format='png', dpi=200, bbox_inches='tight', pad_inches=0.05)

    fig2.savefig(config['outpath_figs']+"filtered/"+f"{event_name}_filtered.png", format='png', dpi=200, bbox_inches='tight', pad_inches=0.05)

    fig3.savefig(config['outpath_figs']+"spectra/"+f"{event_name}_spectra.png", format='png', dpi=200, bbox_inches='tight', pad_inches=0.05)

## End of File
