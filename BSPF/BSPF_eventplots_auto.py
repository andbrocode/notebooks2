#!/usr/bin/env python
# coding: utf-8

# # Automatic BSPF Eventplots

# Creates automatic event plots based on catalog

# In[1]:


import os
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from pprint import pprint

from functions.request_data import __request_data
from functions.add_distances_and_backazimuth import __add_distances_and_backazimuth
from functions.compute_adr_pfo import __compute_adr_pfo


# In[2]:


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
else:
    data_path = '/export/dump/abrotzer/'



# In[3]:


def __process_xpfo(config, st, inv):

    ii_pfo = st.copy()

#     pre_filt = [0.005, 0.01, 19, 20]

    ## cut properly
#     ii_pfo.trim(config['tbeg'], config['tend'])

    ## demean
    ii_pfo.detrend("demean")

    ## remove response
#     ii_pfo.remove_response(inventory=inv,
#     #                        pre_filt=pre_filt,
#                            output="VEL",
#     #                        water_level=60,
#                            plot=False)

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


# In[7]:


def __empty_stream(reference_stream):

    from numpy import ones
    from obspy import Stream, Trace

    t_ref = reference_stream[0]

    empty = Stream()

    for cha in ["BHZ", "BHN", "BHE"]:
        t = Trace()
        t.data = ones(len(t_ref))
        t.stats.sampling_rate = t_ref.stats.sampling_rate
        t.stats.starttime = t_ref.stats.starttime
        t.stats.network, t.stats.station, t.stats.channel = "PY", "RPFO", cha
        empty += t

    return empty




# In[9]:
# ## Configurations


config = {}

## location of BSPF
config['BSPF_lon'] = -116.455439
config['BSPF_lat'] = 33.610643

## path for figures to store
config['outpath_figs'] = data_path+"BSPF/figures/triggered_all/"

## 
config['translation_type'] = "ACC" ## ACC | DISP | VEL

## path for output data
config['outpath_data'] = data_path+f"BSPF/data/waveforms/{config['translation_type']}/"

## blueSeis sensor (@200Hz)
config['seed_blueseis'] = "PY.BSPF..HJ*"

## Trillium 240 next to BlueSeis on Pier (@40Hz)
config['seed_seismometer1'] = "II.PFO.10.BH*"

## STS2 next to BlueSeis (@200Hz)
config['seed_seismometer2'] = "PY.PFOIX..H*"

config['path_to_catalog'] = data_path+"BSPF/data/catalogs/"

config['catalog'] = "BSPF_catalog_20221001_20230930_triggered.pkl"



# In[10]:
# ## Event Info


events = pd.read_pickle(config['path_to_catalog']+config['catalog'])


#events.reset_index(inplace=True)

#events.rename(columns = {'index':'origin'}, inplace = True)



# In[17]:
# ## RUN LOOP


global errors
errors = []

for jj, ev in enumerate(events.index):

    print(f"\n _____________________________________")
    print(f"\n -> {jj} {events.origin[jj]} ")

    event_name = str(events.origin[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]

    filename=config['outpath_figs']+"raw/"+f"{event_name}_raw.png"

    ## check if file already exists
#     if os.path.isfile(filename):
#         print(f" -> file alread exits for {event_name}")
#         continue

    ## configuration adjustments
    config['title'] = f"{events.origin[jj]} UTC | M{events.magnitude[jj]}"
    config['tbeg'] = obs.UTCDateTime(str(events.origin[jj]))-60


    ## select appropriate seismometer
    # if config['tbeg'] < obs.UTCDateTime("2023-04-01"):
    #     config['seed_seismometer'] = config['seed_seismometer1']
    #     config['fmin'], config['fmax'] = 0.01, 18.0
    # else:
    #     config['seed_seismometer'] = config['seed_seismometer2']
    #     config['fmin'], config['fmax'] = 0.01, 90.0


    ## same endtime for all
    config['tend'] = obs.UTCDateTime(events.origin[jj])+120


    from functions.get_stream import __get_stream

    st0 = __get_stream(config['tbeg'], config['tend'])


#     ## load and process blueSeis data
#     try:
#         py_bspf0, py_bspf_inv = __request_data(config['seed_blueseis'], config['tbeg'], config['tend'], config['translation_type'])

#     except Exception as e:
#         print(e)
#         print(f" -> failed to request BSPF for event: {ev}")
#         continue


#     ## load and process seismometer data
#     try:
#         ii_pfo0, ii_pfo_inv = __request_data(config['seed_seismometer'], config['tbeg'], config['tend'], config['translation_type'])

#     except Exception as e:
#         print(e)
#         print(f" -> failed to request PFO for event: {ev}")
#         continue

#     ## continue if either one stream ist empty
#     if py_bspf0 is None or ii_pfo0 is None:
#         continue

#     ## processing data
# #    if ii_pfo0[0].stats.sampling_rate != py_bspf0[0].stats.sampling_rate:
# #        py_bspf0.resample(ii_pfo0[0].stats.sampling_rate)


#     ## joining data
#     st0 = py_bspf0
#     st0 += ii_pfo0

#     ## apply bandpass filter for BSPF and PFO
#     st0 = st0.detrend("linear")
#     st0 = st0.taper(0.01)
#     st0 = st0.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True)


#     ## compute ADR

#     ## for complete array
#     try:
#         pfo_adr = __compute_adr_pfo(config['tbeg']-100, config['tend']+100, submask="all")
#         for tr in pfo_adr:
#             tr.stats.location = "all"
#         st0 += pfo_adr
#     except:
#         print(" -> failed to compute all ADR ...")
#         pfo_adr = __empty_stream(st0)

#     ## for inner array
#     try:
#         pfo_adr = __compute_adr_pfo(config['tbeg']-100, config['tend']+100, submask="inner")
#         for tr in pfo_adr:
#             tr.stats.location = "inn"
#         st0 += pfo_adr
#     except:
#         print(" -> failed to compute inner ADR ...")
#         pfo_adr = __empty_stream(st0)

#     st0 = st0.resample(40, no_filter=False)

#     st0 = st0.sort()

#     st0 = st0.trim(config['tbeg'], config['tend'])


    ## processing data stream
#    st = st0.copy()
#    st.trim(config['tbeg'], config['tend'])


    ## store waveform data
    num = str(jj).rjust(3,"0")
    waveform_filename = f"{num}_{str(events.origin[jj]).split('.')[0].replace('-','').replace(':','').replace(' ','_')}.mseed"
    st0.write(config['outpath_data']+waveform_filename, format="MSEED")
    print(f" -> writing {waveform_filename}")

    ## create eventname
    event_name = str(events.origin[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]


    ## plotting figures
    fig1 = st0.plot(equal_scale=False, show=False);
#     fig1 = st0.plot(equal_scale=False, show=False);

#     fig2 = __makeplot(config, st)

#     fig3 = __makeplotStreamSpectra2(st, config, fscale="linlin");

    ## saving figures
    fig1.savefig(config['outpath_figs']+"raw/"+f"{event_name}_raw.png", dpi=200, bbox_inches='tight', pad_inches=0.05)

#     fig2.savefig(config['outpath_figs']+"filtered/"+f"{event_name}_filtered.png", dpi=200, bbox_inches='tight', pad_inches=0.05)

#     fig3.savefig(config['outpath_figs']+"spectra/"+f"{event_name}_spectra.png", dpi=200, bbox_inches='tight', pad_inches=0.05)


## End of File
