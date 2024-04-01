#!/usr/bin/env python
# coding: utf-8

# # RomyEvents - Automatic Eventplots

# Creates automatic event plots based on catalog



import os
import gc
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.ma as ma

from tqdm import tqdm
from pprint import pprint

from functions.add_distances_and_backazimuth import __add_distances_and_backazimuth

from andbro__querrySeismoData import __querrySeismoData
from andbro__read_sds import __read_sds

import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'



def __makeplot(config, st):


    st_in = st.copy()

    try:
        acc_min, acc_max = -max(abs(st_in.select(station="FUR")[0].data)), max(abs(st_in.select(station="FUR")[0].data))
    except:
        acc_min, acc_max = -1e-6, 1e-6

    try:
        rot_min, rot_max = -3*max(abs(st_in.select(station="RLAS")[0].data)), 3*max(abs(st_in.select(station="RLAS")[0].data))
    except:
        rot_min, rot_max = -1e-9, 1e-9


    fig, ax = plt.subplots(len(st_in), 1, figsize=(15, 10), sharex=True)

    font = 14

    for i, tr in enumerate(st_in):

        ax[i].plot(tr.times(), tr.data, 'k', label=tr.stats.station+"."+tr.stats.channel)

        ax[i].legend(loc=1)

        if "FUR" in tr.stats.station:
            ax[i].set_ylim(acc_min*1.2, acc_max*1.2)
        else:
            ax[i].set_ylim(rot_min*1.2, rot_max*1.2)

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

    for i, tr in enumerate(tqdm(st_in)):


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


# ## Configurations


config = {}

## ROMY coordinates
config['ROMY_lon'] = 11.275501
config['ROMY_lat'] = 48.162941

config['duration'] = 3600*3 # in seconds

config['fmin'] = 0.01
config['fmax'] = 0.1

## path for figures to store
config['outpath_figs'] = data_path+"romy_events/figures/"

## path for output data
config['outpath_data'] = data_path+"romy_events/data/waveforms/"


config['seeds'] = ["BW.ROMY.10.BJZ", "BW.ROMY..BJU", "BW.ROMY..BJV", "BW.ROMY..BJW",
                   "BW.RLAS..BJZ",
                   "GR.FUR..BHZ", "GR.FUR..BHN", "GR.FUR..BHE",
                   "GR.WET..BHZ", "GR.WET..BHN", "GR.WET..BHE",
                  ]

config['path_to_catalog'] = data_path+"romy_events/data/catalogs/"

config['catalog'] = "ROMY_global_catalog_20200101_20231231.pkl"

# specify output unit of translation
config['tra_output'] = "ACC"

# ## Load Events

# In[50]:


events = pd.read_pickle(config['path_to_catalog']+config['catalog'])


# In[51]:


events['origin'] = events.timestamp


# In[52]:


events


# In[53]:

## select events for minmal magnitude
events = events[events.magnitude > 6]
events



## RUN LOOP


global errors

errors = []
adr_status = []


for jj in range(events.shape[0]):

    # specify event number
    num = str(jj).rjust(3, "0")

    print(f"\n -> {num} {events.origin.iloc[jj]} ")

    try:
        event_name = str(events.origin.iloc[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]
    except:
        print(f" -> {num}: error for {events.origin.iloc[jj]}")
        continue


    # check if file already exists
    # filename = config['outpath_figs']+"raw/"+f"{event_name}_raw.png"
    # if os.path.isfile(filename):
    #     print(f" -> file alread exits for {event_name}")
    #     continue

    # waveform_filename = f"{num}_{str(events.origin.iloc[jj]).split('.')[0].replace('-','').replace(':','').replace(' ','_')}.mseed"
    # if os.path.isfile(config['outpath_data']+config['tra_output']+"/"+waveform_filename):
    #     print(f" -> mseed file alread exits!")
    #     continue


    # configuration adjustments
    config['title'] = f"{num}_{events.origin.iloc[jj]} UTC | M{events.magnitude.iloc[jj]}"
    config['tbeg'] = obs.UTCDateTime(str(events.origin.iloc[jj]))


    # same endtime for all
    config['tend'] = obs.UTCDateTime(events.origin.iloc[jj]) + config['duration']

    # prepare stream object
    st0 = obs.Stream()


    for seed in tqdm(config['seeds']):

        if "FUR" in seed:
            repo = "IRIS"
        elif "WET" in seed:
            repo = "online"
        else:
            repo = "george"

        net, sta, loc, cha = seed.split(".")

        try:
            try:
                stx, invx = __querrySeismoData(
                                                seed_id=seed,
                                                starttime=config['tbeg'],
                                                endtime=config['tend'],
                                                repository=repo,
                                                path=None,
                                                restitute=False,
                                                detail=None,
                                                fill_value=None,
                                            )
                if "FUR" in seed or "WET" in seed:
                    stx = stx.remove_response(invx, output=config['tra_output'].upper())
                else:
                    stx = stx.remove_sensitivity(invx)

                st0 += stx

            except:
                print(f"  -> {repo} failed")

            try:
                stx = __read_sds(archive_path+"romy_archive/", seed, config['tbeg'], config['tend'])

                # invx = obs.read_inventory(root_path+f"Documents/ROMY/stationxml_ringlaser/station_{net}_{sta}.xml", format="STATIONXML")
                invx = obs.read_inventory(root_path+f"Documents/ROMY/stationxml_ringlaser/dataless/dataless.seed.{net}_{sta}", format="SEED")

                if "J" in cha:
                    stx = stx.remove_sensitivity(invx)

                st0 += stx
            except:
                print(f"  -> archive failed")

        except Exception as e:
            print(e)
            print(f" -> failed to request {seed} for event: {events.origin.iloc[jj]}")
            errors.append(f" -> failed to request {seed} for event: {events.origin.iloc[jj]}")
            continue

    # ______________________________________________________________
    # processing data stream

    st0 = st0.sort()

    st0 = st0.merge();

    print(st0)

    # check for masked arrays
    for tr in st0:
        arr = tr.data
        if ma.is_masked(arr):
            print(f"  -> masked array: {tr.stats.station}.{tr.stats.channel}")
            tr.data = arr.filled(fill_value=0)

    st1 = st0.copy();
    st1 = st1.detrend("linear");
    st1 = st1.taper(0.05);
    st1 = st1.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True);

    st0 = st0.trim(config['tbeg'], config['tend']);
    st1 = st1.trim(config['tbeg'], config['tend']);

    st0.sort();
    st1.sort();

    # st.plot(equal_scale=False);

    # ______________________________________________________________
    # store waveform data
    try:

        # create subdir
        if not os.path.isdir(config['outpath_data']+config['tra_output']):
            os.mkdir(config['outpath_data']+config['tra_output'])

        # prepare filename
        waveform_filename = f"{num}_{str(events.origin.iloc[jj]).split('.')[0].replace('-','').replace(':','').replace(' ','_')}.mseed"

        # store data
        st0.write(config['outpath_data']+config['tra_output']+"/"+waveform_filename, format="MSEED");

        print(f"  -> stored: {waveform_filename}")

    except Exception as e:
        print(f"  -> failed to store waveforms!")
        print(e)

    # ______________________________________________________________
    # plotting figures
#     try:
#         fig1 = st0.plot(equal_scale=False, show=False);
#         fig2 = st1.plot(equal_scale=False, show=False);

#         fig2 = __makeplot(config, st1)

#         # saving figures
#         fig1.savefig(config['outpath_figs']+"raw/"+f"{num}_{event_name}_raw.png", dpi=150, bbox_inches='tight', pad_inches=0.05)
#         print(f"  -> stored: {num}_{event_name}_raw.png")

#         fig2.savefig(config['outpath_figs']+"filtered/"+f"{num}_{event_name}_filtered.png", dpi=150, bbox_inches='tight', pad_inches=0.05)
#         print(f"  -> stored: {num}_{event_name}_filtered.png")
#     except:
#         print(f"  -> failed to store figures!")

    gc.collect()


pprint(errors)

# ______________________________________________________________
# Make StatusPlot

import matplotlib.colors

fig3, ax = plt.subplots(1, 1, figsize=(15, 5))

cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

ax.pcolormesh(np.array(adr_status).T, cmap=cmap, edgecolors="k", lw=0.5)

ax.set_yticks(np.arange(0, len(config['seeds']))+0.5, labels=config['seeds'])

ax.set_xlabel("Event No.", fontsize=12)

fig3.savefig(config['outpath_figs']+f"status.png", dpi=150, bbox_inches='tight', pad_inches=0.05)

# End of File