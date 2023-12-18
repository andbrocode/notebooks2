#!/usr/bin/env python
# coding: utf-8

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
from functions.add_radial_and_transverse_channel import __add_radial_and_transverse_channel


# In[2]:


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'


# In[3]:


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

    plt.show();
    del st_in
    return fig


# In[4]:


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


# In[5]:


def __stream_to_dataframe(st):

    dff = pd.DataFrame()

    for tr in st:
        name = f"{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}"
        dff[name] = tr.data

    return dff


# In[6]:


config = {}

## location of BSPF
config['BSPF_lon'] = -116.455439
config['BSPF_lat'] = 33.610643

## path for figures to store
config['outpath_figs'] = data_path+"BSPF/figures/SNR/"

##
config['translation_type'] = "ACC"  ## ACC / DISP

## path for output data
config['outpath_data'] = data_path+"BSPF/data/"

config['path_to_mseed'] = data_path+f"BSPF/data/waveforms/{config['translation_type'].upper()}/"

## blueSeis sensor (@200Hz)
# config['seed_blueseis'] = "PY.BSPF..HJ*"

## Trillium 240 next to BlueSeis on Pier (@40Hz)
# config['seed_seismometer1'] = "II.PFO.10.BH*"

## STS2 next to BlueSeis (@200Hz)
# config['seed_seismometer2'] = "PY.PFOIX..HH*"

config['path_to_catalog'] = data_path+"BSPF/data/catalogs/"

config['catalog'] = "BSPF_catalog_20221001_20230930_triggered.pkl"
# config['catalog'] = "BSPF_catalog_20221001_20231001_triggered.pkl"


# ## Event Info

# In[ ]:


## read data frame of selected triggered events
events = pd.read_pickle(config['path_to_catalog']+config['catalog'])

## add column for hypocenter distance
events['Hdistance_km'] = np.sqrt(events['distances_km']**2 + (events['depth']/1000)**2)

events


# In[ ]:


events


# ## RUN LOOP

# In[ ]:


def __compute_Amax(header, st_in, out_lst, trigger_time, win_length_sec):

    from numpy import ones, nan, nanargmax, argmax
    from obspy import UTCDateTime

    st_in = st_in.sort()

    st_in = st_in.detrend("demean")
    st_in = st_in.taper(0.01)
    st_in = st_in.filter("bandpass", freqmin=1.0, freqmax=12.0, corners=4, zerophase=True)


    ## get relative samples of signal window
    t_rel_sec = abs(UTCDateTime(trigger_time)-st[0].stats.starttime)

    df = st[0].stats.sampling_rate  ## sampling rate

    NN = int(df * win_length_sec) ## samples

    n_rel_spl = t_rel_sec * df ## samples

    n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)

    ## find index of maximum for PFO Z
    tr_ = st_in.select(station="PFO*", location="10", channel=f"*R")[0]
    max_idx = argmax(abs(tr_.data[n_signal_1:n_signal_2]))

    ## samples offset around maximum
    n_off = int(0.1 * df)

    out = {}
    for ii, h in enumerate(header):
        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]
        try:
            tr = st_in.select(station=sta, location=loc, channel=f"*{cha}")[0]
            out[h+"_amax"] = max(abs(tr.data[n_signal_1+max_idx-n_off:n_signal_1+max_idx+n_off]))
            # out[h+"_amax"] = max(abs(tr.data[n_signal_1+max_idx]))
        except:
            out[h+"_amax"] = nan
            print(f" -> amax: {h} adding nan")

    return out


# In[ ]:


def __compute_SNR(header, st_in, out_lst, trigger_time, win_length_sec=10, plot=False, plot_save=False):

    from numpy import nanmean, sqrt, isnan, ones, nan, nanpercentile, nanargmax, argmax
    from obspy import UTCDateTime

    st_in = st_in.sort()
    st_in = st_in.detrend("demean")
    # st_in = st_in.taper(0.01)
    # st_in = st_in.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=4, zerophase=True)

    if plot or plot_save:
        fig, ax = plt.subplots(len(st_in), 1, figsize=(15, 15), sharex=True)

    out = {}
    for ii, h in enumerate(header):

        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]

        try:
            tr = st_in.select(station=sta, location=loc, channel=f"*{cha}")[0]
        except:
            out[h+"_snr"] = nan

        t_rel_sec = abs(UTCDateTime(trigger_time)-tr.stats.starttime)

        df = tr.stats.sampling_rate

        NN = int(df * win_length_sec) ## samples

        n_rel_spl = t_rel_sec * df ## samples

        n_offset = df * 2 ## samples

        n_noise_1, n_noise_2 = int(n_rel_spl-NN-n_offset), int(n_rel_spl-n_offset)
        n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)

        ## noise, signal and ratio using mean
        # noise = nanmean(tr.data[n_noise_1:n_noise_2]**2)
        # signal = nanmean(tr.data[n_signal_1:n_signal_2]**2)
        # out[h+"_snr"] = sqrt(signal/noise)

        ## find index of maximum for PFO Z
        tr_ = st_in.select(station="PFO*", location="10", channel=f"*R")[0]
        max_idx = argmax(abs(tr_.data[n_signal_1:n_signal_2]))

        ## samples offset around maximum
        n_off = int(1 * df)


        ## noise, signal and ratio using percentile
        try:
            noise = nanpercentile(abs(tr.data[n_noise_1:n_noise_2]), 99.9)
            signal = nanpercentile(abs(tr.data[n_signal_1:n_signal_2]), 99.9)
            out[h+"_snr"] = signal/noise
        except:
            out[h+"_snr"] = nan
            print(f" -> snr: {h} adding nan")


        if plot or plot_save:

            if ii < len(st_in):

                scaling = 1


                ax[ii].plot(abs(tr.data)*scaling, label=f"{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}")

                ax[ii].legend(loc=1)

                ## signal period
                ax[ii].axvline(n_rel_spl, color="g")
                ax[ii].axvline(n_rel_spl+NN, color="g")

                # ax[ii].axhline(signal*scaling, n_rel_spl, n_rel_spl+NN, color="g", ls="--", zorder=3)

                ## noise period
                ax[ii].axvline(n_rel_spl-n_offset, color="r")
                ax[ii].axvline(n_rel_spl-NN-n_offset, color="r")

                # ax[ii].axhline(noise*scaling, n_rel_spl-n_offset, n_rel_spl-NN-n_offset, color="r", ls="--", zorder=3)

                maximal_value = abs(tr.data[n_signal_1+max_idx])
                ax[ii].scatter(n_signal_1+max_idx, maximal_value*scaling, color="tab:orange", alpha=0.7)


                ## OLD window picking
                # maximal_idx = nanargmax(abs(tr.data[n_signal_1+max_idx-n_off:n_signal_1+max_idx+n_off]))
                # maximal_value = abs(tr.data[n_signal_1+max_idx-n_off+maximal_idx])
                # ax[ii].scatter(maximal_idx+n_signal_1+max_idx-n_off, maximal_value*scaling, color="tab:orange", alpha=0.7)

                # ax[ii].axvline(n_signal_1+max_idx-n_off, color="orange")
                # ax[ii].axvline(n_signal_1+max_idx+n_off, color="orange")
                # ax[ii].axvline(n_signal_1+max_idx-n_off+maximal_idx, color="orange", ls="--")

                ax[ii].set_ylim(bottom=-0)

    if plot:
        plt.show();

    if plot_save:
        fig.savefig(config['outpath_figs']+f"SNR_{st_in[0].stats.starttime}.png", format="png", dpi=200, bbox_inches='tight')
        plt.close();

    return out


# In[ ]:


def __compute_adr_max(header, st_in, out_lst, trigger_time, win_length_sec):

    from numpy import ones, nan, argmax
    from obspy import UTCDateTime

    st_in = st_in.copy().sort()

    st_in = st_in.detrend("linear")

    bspf_i = st_in.select(station="BSPF").copy()
    bspf_a = st_in.select(station="BSPF").copy()
    bspf_m = st_in.select(station="BSPF").copy()
    adr_i = st_in.select(station="RPFO", location="in").copy()
    adr_a = st_in.select(station="RPFO", location="al").copy()
    adr_m = st_in.select(station="RPFO", location="mi").copy()

    bspf_a = bspf_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)
    adr_a = adr_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)

    bspf_i = bspf_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)
    adr_i = adr_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)

    bspf_m = bspf_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)
    adr_m = adr_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)

    for tr in bspf_a:
        tr.stats.location = "al"
    for tr in bspf_m:
        tr.stats.location = "mi"
    for tr in bspf_i:
        tr.stats.location = "in"

    bspf_a.detrend("linear")
    adr_a.detrend("linear")

    bspf_i.detrend("linear")
    adr_i.detrend("linear")

    adr_m.detrend("linear")
    adr_m.detrend("linear")

    st0 = adr_i
    st0 += bspf_i

    st0 += bspf_a
    st0 += adr_a

    st0 += bspf_m
    st0 += adr_m


    ## get relative samples of signal window
    t_rel_sec = abs(UTCDateTime(trigger_time)-st[0].stats.starttime)

    df = st[0].stats.sampling_rate  ## sampling rate

    NN = int(df * win_length_sec) ## samples

    n_rel_spl = t_rel_sec * df ## samples

    n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)


    ## find index of maximum for PFO Z
    tr_ = st_in.select(station="PFO*", location="10", channel=f"*T")[0]
    max_idx = argmax(abs(tr_.data[n_signal_1:n_signal_2]))

    ## samples offset around maximum
    n_off = int(0.1 * df)


    out = {}
    for ii, h in enumerate(header):

        if h == "origin":
            continue

        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]
        try:
            tr = st0.select(station=sta, location=loc, channel=f"*{cha}")[0]
            out[h] = max(abs(tr.data[n_signal_1+max_idx-n_off:n_signal_1+max_idx+n_off]))
        except Exception as e:
            out[h] = nan
            print(f" -> adr: {h} adding nan")

    return out


# In[ ]:

def __compute_adr_snr(header, st_in, out_lst, trigger_time, win_length_sec=10):

    from numpy import ones, nan, argmax, nanpercentile
    from obspy import UTCDateTime

    st_in = st_in.copy().sort()

    st_in = st_in.detrend("linear")

    bspf_i = st_in.select(station="BSPF").copy()
    bspf_a = st_in.select(station="BSPF").copy()
    bspf_m = st_in.select(station="BSPF").copy()
    adr_i = st_in.select(station="RPFO", location="in").copy()
    adr_a = st_in.select(station="RPFO", location="al").copy()
    adr_m = st_in.select(station="RPFO", location="mi").copy()

    bspf_a = bspf_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)
    adr_a = adr_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)

    bspf_i = bspf_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)
    adr_i = adr_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)

    bspf_m = bspf_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)
    adr_m = adr_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)

    for tr in bspf_a:
        tr.stats.location = "al"
    for tr in bspf_m:
        tr.stats.location = "mi"
    for tr in bspf_i:
        tr.stats.location = "in"

    bspf_a.detrend("linear")
    adr_a.detrend("linear")

    bspf_i.detrend("linear")
    adr_i.detrend("linear")

    adr_m.detrend("linear")
    adr_m.detrend("linear")

    st0 = adr_i
    st0 += bspf_i

    st0 += bspf_a
    st0 += adr_a

    st0 += bspf_m
    st0 += adr_m



    ## get relative samples of signal window
    t_rel_sec = abs(UTCDateTime(trigger_time)-st[0].stats.starttime)

    df = st[0].stats.sampling_rate  ## sampling rate

    NN = int(df * win_length_sec) ## samples

    n_rel_spl = t_rel_sec * df ## samples


    n_offset = df * 2 ## samples

    n_noise_1, n_noise_2 = int(n_rel_spl-NN-n_offset), int(n_rel_spl-n_offset)
    n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)


    out = {}
    for ii, h in enumerate(header):

        if h == "origin":
            continue

        ## noise, signal and ratio using percentile
        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]
        try:
            tr = st0.select(station=sta, location=loc, channel=f"*{cha}")[0]

            noise = nanpercentile(abs(tr.data[n_noise_1:n_noise_2]), 99)
            signal = nanpercentile(abs(tr.data[n_signal_1:n_signal_2]), 99)
            out[h] = signal/noise
        except:
            out[h] = nan
            print(f" -> snr: {h} adding nan")

    return out

# In[ ]:


def __compute_adr_cc(header, st_in, out_lst, trigger_time, win_length_sec):

    from numpy import ones, nan, argmax
    from obspy import UTCDateTime
    from obspy.signal.cross_correlation import correlate

    st_in = st_in.copy().sort()

    st_in = st_in.detrend("linear")

    bspf_i = st_in.select(station="BSPF").copy()
    bspf_a = st_in.select(station="BSPF").copy()
    bspf_m = st_in.select(station="BSPF").copy()
    adr_i = st_in.select(station="RPFO", location="in").copy()
    adr_a = st_in.select(station="RPFO", location="al").copy()
    adr_m = st_in.select(station="RPFO", location="mi").copy()


    bspf_a = bspf_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)
    adr_a = adr_a.filter("bandpass", freqmin=0.1, freqmax=0.5, corners=8, zerophase=True)

    bspf_i = bspf_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)
    adr_i = adr_i.filter("bandpass", freqmin=1.0, freqmax=6.0, corners=8, zerophase=True)

    bspf_m = bspf_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)
    adr_m = adr_m.filter("bandpass", freqmin=0.5, freqmax=1.0, corners=8, zerophase=True)

    for tr in bspf_a:
        tr.stats.location = "al"
    for tr in bspf_m:
        tr.stats.location = "mi"
    for tr in bspf_i:
        tr.stats.location = "in"

    bspf_a.detrend("linear")
    adr_a.detrend("linear")

    bspf_i.detrend("linear")
    adr_i.detrend("linear")

    adr_m.detrend("linear")
    adr_m.detrend("linear")

    st0 = adr_i
    st0 += bspf_i

    st0 += bspf_a
    st0 += adr_a

    st0 += bspf_m
    st0 += adr_m


    ## get relative samples of signal window
    t_rel_sec = abs(UTCDateTime(trigger_time)-st[0].stats.starttime)

    df = st[0].stats.sampling_rate  ## sampling rate

    NN = int(df * win_length_sec) ## samples

    n_rel_spl = t_rel_sec * df ## samples

    n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)


    ## find index of maximum for PFO
    tr_ = st_in.select(station="PFO*", location="10", channel=f"*T")[0]
    max_idx = argmax(abs(tr_.data[n_signal_1:n_signal_2]))

    ## samples offset around maximum
    n_off = int(1 * df)

    _i1, _i2 = n_signal_1+max_idx-n_off, n_signal_1+max_idx+n_off

    # print(st0.__str__(extended=True))

    out = {}
    for ii, h in enumerate(header):

        if h == "origin":
            continue

        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]

        try:
            tr_ref = st0.select(station="BSPF", location=loc, channel=f"*{cha}")[0]
            tr = st0.select(station=sta, location=loc, channel=f"*{cha}")[0]
            # out[h] = round(max(correlate(tr_ref.data[n_signal_1:n_signal_2], tr.data[n_signal_1:n_signal_2], 0)), 1)
            out[h] = round(max(correlate(tr_ref.data[_i1:_i2], tr.data[_i1:_i2], 0)), 1)

        except Exception as e:
            # print("cc error:" + e)
            out[h] = nan
            print(f" -> cc: {h} adding nan")

    return out


# In[ ]:


config['mseed_files'] = sorted([file for file in os.listdir(config['path_to_mseed'])])

len(config['mseed_files'])


# In[ ]:


## create dataframe for output
out_df = pd.DataFrame()

out_df["Torigin"] = events.origin
out_df["Magnitude"] = events.magnitude
out_df["CoincidenceSum"] = events.cosum
out_df["Mag_type"] = events.type
out_df["BAZ"] = events.backazimuth
out_df["Edistance_km"] = events.distances_km
out_df["Hdistance_km"] = events.Hdistance_km

tmp_events = [str(ee).split(".")[0] for ee in events.origin]

data_amax, data_adr_snr, data_adr, skipped, nan_row = [], [], [], 0, 0

## pre-define header for data frames
header = ['BSPF__E','BSPF__N','BSPF__R','BSPF__T','BSPF__Z',
          'PFO_10_E','PFO_10_N','PFO_10_R','PFO_10_T','PFO_10_Z',
          'RPFO_al_E','RPFO_al_N','RPFO_al_Z',
          'RPFO_in_E','RPFO_in_N','RPFO_in_Z',
          'RPFO_mi_E','RPFO_mi_N','RPFO_mi_Z'
         ]

## prepare dataframes
header_amax = [h+"_amax" for h in header]
header_amax.insert(0, "origin")
df_amax = pd.DataFrame(columns=header_amax)

# header_snr = [h+"_snr" for h in header]
# header_snr.insert(0, "origin")
# df_snr = pd.DataFrame(columns=header_snr)

header_adr_snr = []
for hh in ["BSPF_al_N_snr", "BSPF_al_E_snr", "BSPF_al_Z_snr", "BSPF_al_T_snr", "BSPF_al_R_snr",
           "BSPF_mi_N_snr", "BSPF_mi_E_snr", "BSPF_mi_Z_snr", "BSPF_mi_T_snr", "BSPF_mi_R_snr",
           "BSPF_in_N_snr", "BSPF_in_E_snr", "BSPF_in_Z_snr", "BSPF_in_T_snr", "BSPF_in_R_snr",
           "RPFO_in_N_snr", "RPFO_in_E_snr", "RPFO_in_Z_snr",
           "RPFO_al_N_snr", "RPFO_al_E_snr", "RPFO_al_Z_snr",
           "RPFO_mi_N_snr", "RPFO_mi_E_snr", "RPFO_mi_Z_snr"]:
    header_adr_snr.append(hh)


header_adr = []
for hh in ["BSPF_al_N_adr", "BSPF_al_E_adr", "BSPF_al_Z_adr", "BSPF_al_T_adr", "BSPF_al_R_adr",
           "BSPF_mi_N_adr", "BSPF_mi_E_adr", "BSPF_mi_Z_adr", "BSPF_mi_T_adr", "BSPF_mi_R_adr",
           "BSPF_in_N_adr", "BSPF_in_E_adr", "BSPF_in_Z_adr", "BSPF_in_T_adr", "BSPF_in_R_adr",
           "RPFO_in_N_adr", "RPFO_in_E_adr", "RPFO_in_Z_adr",
           "RPFO_al_N_adr", "RPFO_al_E_adr", "RPFO_al_Z_adr",
           "RPFO_mi_N_adr", "RPFO_mi_E_adr", "RPFO_mi_Z_adr"]:
    header_adr.append(hh)

# header_adr = [h+"_adr" for h in header]
header_adr.insert(0, "origin")
df_adr = pd.DataFrame(columns=header_adr)

df_adr_cc = pd.DataFrame(columns=header_adr)

header_adr_snr.insert(0, "origin")
df_adr_snr = pd.DataFrame(columns=header_adr_snr)


# for event in tqdm(config['mseed_files'][:50]):
for event in tqdm(config['mseed_files']):

    yy = int(event.replace(".","_").split("_")[1][:4])
    mm = int(event.replace(".","_").split("_")[1][4:6])
    dd = int(event.replace(".","_").split("_")[1][6:8])
    h = int(event.replace(".","_").split("_")[2][0:2])
    m = int(event.replace(".","_").split("_")[2][2:4])
    s = int(event.replace(".","_").split("_")[2][4:6])

    otime = f"{yy}-{mm:02d}-{dd:02d} {h:02d}:{m:02d}:{s:02d}"


    if otime not in tmp_events:
        skipped += 1
        continue
    else:
        jj = tmp_events.index(otime)

#     print(f"\n -> {jj} {events.origin[jj]} ")

    event_name = str(events.origin[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]


    try:

        st = obs.read(config['path_to_mseed']+event)

        ## resample all to 40 Hz
        st = st.resample(40, no_filter=False)

        st = st.sort();

        ## rename PFOIX events to make compatible
        for tr in st:
            if "PFOIX" in tr.stats.station:
                tr.stats.station = "PFO"
                tr.stats.location = "10"

        # for tr in st:
        #     if "BSPF" in tr.stats.station:
        #         tr.data = np.roll(tr.data, 4)

        ## add radial and transverse channels
        st = __add_radial_and_transverse_channel(st, "PFO*", events.backazimuth[jj])
        st = __add_radial_and_transverse_channel(st, "BSPF", events.backazimuth[jj])

        if len(st) != 19:
            print(f"{len(st)} is not 19")

        ## processing data stream
        st = st.detrend("linear")
        st = st.taper(0.01)
        st = st.filter("highpass", freq=0.01, corners=4, zerophase=True)


        ## compute maximal amplitude values (PGA and PGR)
        data_amax = __compute_Amax(header, st, data_amax, events.trigger_time[jj], win_length_sec=15)
        row = {**{"origin":events.origin[jj]}, **data_amax}
        df_amax.loc[len(df_amax)] = row

        # ## compute signal-to-noise ratios
        # data_snr = __compute_SNR(header, st, data_snr, events.trigger_time[jj], win_length_sec=15, plot=False, plot_save=True)
        # row = {**{"origin":events.origin[jj]}, **data_snr}
        # df_snr.loc[len(df_snr)] = row

        ## compute signal-to-noise ratios
        data_adr_snr = __compute_adr_snr(header_adr_snr, st, data_adr_snr, events.trigger_time[jj], win_length_sec=15)
        row = {**{"origin":events.origin[jj]}, **data_adr_snr}
        df_adr_snr.loc[len(df_adr_snr)] = row

        ## compute adr maxima
        data_adr = __compute_adr_max(header_adr, st, data_adr, events.trigger_time[jj], win_length_sec=15)
        row = {**{"origin":events.origin[jj]}, **data_adr}
        df_adr.loc[len(df_adr)] = row

        ## compute bspf vs adr cross correlation coefficients
        data_adr_cc = __compute_adr_cc(header_adr, st, data_adr, events.trigger_time[jj], win_length_sec=15)
        row = {**{"origin":events.origin[jj]}, **data_adr_cc}
        df_adr_cc.loc[len(df_adr)] = row


    except Exception as e:
        print(e)
        print(f" -> failed for {event}")
        skipped += 1
        continue


        
        
## _______________________________________

out_df['origin'] = out_df['Torigin']

df_amax_out = pd.merge(out_df, df_amax, on=["origin"])

df_amax_out.to_pickle(config['outpath_data']+f"bspf_analysisdata_amax_{config['translation_type']}.pkl")
print(f"\n -> writing: {config['outpath_data']}bspf_analysisdata_amax_{config['translation_type']}.pkl")

# ## _______________________________________

df_snr_out = pd.merge(out_df, df_adr_snr, on=["origin"])
df_snr_out.to_pickle(config['outpath_data']+f"bspf_analysisdata_snr_{config['translation_type']}.pkl")
print(f"\n -> writing: {config['outpath_data']}bspf_analysisdata_snr_{config['translation_type']}.pkl")

# ## _______________________________________

df_adr_out = pd.merge(out_df, df_adr, on=["origin"])

df_adr_out.to_pickle(config['outpath_data']+f"bspf_analysisdata_adr_{config['translation_type']}.pkl")
print(f"\n -> writing: {config['outpath_data']}bspf_analysisdata_adr_{config['translation_type']}.pkl")

# ## _______________________________________

df_adr_cc_out = pd.merge(out_df, df_adr_cc, on=["origin"])

df_adr_cc_out.to_pickle(config['outpath_data']+f"bspf_analysisdata_adr_cc_{config['translation_type']}.pkl")
print(f"\n -> writing: {config['outpath_data']}bspf_analysisdata_adr_cc_{config['translation_type']}.pkl")



print(f"\n -> skipped: {skipped}")

# ## Testing Signal-to-Noise ratios

# In[ ]:


def __compute_SNR(header, st_in, out_lst, trigger_time, win_length_sec=10, plot=False, plot_save=False):

    from numpy import nanmean, sqrt, isnan, ones, nan, nanpercentile, nanargmax, argmax
    from obspy import UTCDateTime

    st_in = st_in.sort()
    # st_in = st_in.detrend("demean").taper(0.01).filter("bandpass", freqmin=0.5, freqmax=15.0, corners=4, zerophase=True)

    if plot or plot_save:
        fig, ax = plt.subplots(len(st_in), 1, figsize=(15, 15), sharex=True)

    out = {}
    for ii, h in enumerate(header):

        sta, loc, cha = h.split("_")[0], h.split("_")[1], h.split("_")[2]

        try:
            tr = st_in.select(station=sta, location=loc, channel=f"*{cha}")[0]
        except:
            out[h+"_snr"] = nan

        t_rel_sec = abs(UTCDateTime(trigger_time)-tr.stats.starttime)

        df = tr.stats.sampling_rate

        NN = int(df * win_length_sec) ## samples

        n_rel_spl = t_rel_sec * df ## samples

        n_offset = df * 2 ## samples

        n_noise_1, n_noise_2 = int(n_rel_spl-NN-n_offset), int(n_rel_spl-n_offset)
        n_signal_1, n_signal_2 = int(n_rel_spl), int(n_rel_spl+NN)

        ## noise, signal and ratio using mean
        # noise = nanmean(tr.data[n_noise_1:n_noise_2]**2)
        # signal = nanmean(tr.data[n_signal_1:n_signal_2]**2)
        # out[h+"_snr"] = sqrt(signal/noise)

        ## find index of maximum for PFO Z
        tr_ = st_in.select(station="PFO*", location="10", channel=f"*R")[0]
        max_idx = argmax(abs(tr_.data[n_signal_1:n_signal_2]))

        ## samples offset around maximum
        n_off = int(0.1 * df)


        ## noise, signal and ratio using percentile
        try:
            noise = nanpercentile(abs(tr.data[n_noise_1:n_noise_2]), 99.9)
            signal = nanpercentile(abs(tr.data[n_signal_1:n_signal_2]), 99.9)
            out[h+"_snr"] = signal/noise
        except:
            out[h+"_snr"] = nan
            print(f" -> snr: {h} adding nan")


        if plot or plot_save:

            if ii < len(st_in):

                scaling = 1


                ax[ii].plot(abs(tr.data)*scaling, label=f"{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}")

                ax[ii].legend(loc=1)

                ## signal period
                ax[ii].axvline(n_rel_spl, color="g")
                ax[ii].axvline(n_rel_spl+NN, color="g")

                # ax[ii].axhline(signal*scaling, n_rel_spl, n_rel_spl+NN, color="g", ls="--", zorder=3)

                ## noise period
                ax[ii].axvline(n_rel_spl-n_offset, color="r")
                ax[ii].axvline(n_rel_spl-NN-n_offset, color="r")

                # ax[ii].axhline(noise*scaling, n_rel_spl-n_offset, n_rel_spl-NN-n_offset, color="r", ls="--", zorder=3)

                ## New direct picking
                # maximal_value = abs(tr.data[n_signal_1+max_idx])
                # ax[ii].scatter(n_signal_1+max_idx, maximal_value*scaling, color="tab:orange", alpha=0.7, zorder=4)


                ## OLD window picking
                maximal_idx = nanargmax(abs(tr.data[n_signal_1+max_idx-n_off:n_signal_1+max_idx+n_off]))
                maximal_value = abs(tr.data[n_signal_1+max_idx-n_off+maximal_idx])
                ax[ii].scatter(maximal_idx+n_signal_1+max_idx-n_off, maximal_value*scaling, color="tab:orange", alpha=0.7)

                ax[ii].axvline(n_signal_1+max_idx-n_off, color="orange")
                ax[ii].axvline(n_signal_1+max_idx+n_off, color="orange")
                ax[ii].axvline(n_signal_1+max_idx-n_off+maximal_idx, color="orange", ls="--")

                ax[ii].set_ylim(bottom=-0)

    if plot:
        plt.show();

    if plot_save:
        fig.savefig(config['outpath_figs']+f"SNR_{st_in[0].stats.starttime}.png", format="png", dpi=200, bbox_inches='tight')
        plt.close();

    return out


# In[ ]:


## create dataframe for output
out_df = pd.DataFrame()

out_df["Torigin"] = events.origin
out_df["Magnitude"] = events.magnitude
out_df["CoincidenceSum"] = events.cosum
out_df["Mag_type"] = events.type
out_df["BAZ"] = events.backazimuth
out_df["Edistance_km"] = events.distances_km
out_df["Hdistance_km"] = events.Hdistance_km

tmp_events = [str(ee).split(".")[0] for ee in events.origin]

data_amax, data_snr, data_adr, skipped, nan_row = [], [], [], 0, 0

## pre-define header for data frames
header = ['BSPF__E','BSPF__N','BSPF__R','BSPF__T','BSPF__Z',
          'PFO_10_E','PFO_10_N','PFO_10_R','PFO_10_T','PFO_10_Z',
          'RPFO_al_E','RPFO_al_N','RPFO_al_Z',
          'RPFO_in_E','RPFO_in_N','RPFO_in_Z',
          'RPFO_mi_E','RPFO_mi_N','RPFO_mi_Z'
         ]

## prepare dataframes
header_amax = [h+"_amax" for h in header]
header_amax.insert(0,"origin")
df_amax = pd.DataFrame(columns=header_amax)

header_snr = [h+"_snr" for h in header]
header_snr.insert(0,"origin")
df_snr = pd.DataFrame(columns=header_snr)


header_adr = []
for hh in ["BSPF_a_N_adr", "BSPF_a_E_adr", "BSPF_a_Z_adr", "BSPF_a_T_adr", "BSPF_a_R_adr",
           "BSPF_m_N_adr", "BSPF_m_E_adr", "BSPF_m_Z_adr", "BSPF_m_T_adr", "BSPF_m_R_adr",
           "BSPF_i_N_adr", "BSPF_i_E_adr", "BSPF_i_Z_adr", "BSPF_i_T_adr", "BSPF_i_R_adr",
           "RPFO_in_N_adr", "RPFO_in_E_adr", "RPFO_in_Z_adr",
           "RPFO_al_N_adr", "RPFO_al_E_adr", "RPFO_al_Z_adr",
           "RPFO_mi_N_adr", "RPFO_mi_E_adr", "RPFO_mi_Z_adr"]:
    header_adr.append(hh)

# header_adr = [h+"_adr" for h in header]
header_adr.insert(0, "origin")
df_adr = pd.DataFrame(columns=header_adr)



for event in tqdm(config['mseed_files'][200:201]):

    yy = int(event.replace(".","_").split("_")[1][:4])
    mm = int(event.replace(".","_").split("_")[1][4:6])
    dd = int(event.replace(".","_").split("_")[1][6:8])
    h = int(event.replace(".","_").split("_")[2][0:2])
    m = int(event.replace(".","_").split("_")[2][2:4])
    s = int(event.replace(".","_").split("_")[2][4:6])

    otime = f"{yy}-{mm:02d}-{dd:02d} {h:02d}:{m:02d}:{s:02d}"


    if otime not in tmp_events:
        skipped += 1
        continue
    else:
        jj = tmp_events.index(otime)

#     print(f"\n -> {jj} {events.origin[jj]} ")

    event_name = str(events.origin[jj]).replace("-","").replace(":","").replace(" ", "_").split(".")[0]


    try:
        st = obs.read(config['path_to_mseed']+event)

        st = st.resample(40)

        for tr in st:
            if "PFOIX" in tr.stats.station:
                tr.stats.station = "PFO"
                tr.stats.location = "10"


        ## add radial and transverse channels
        st = __add_radial_and_transverse_channel(st, "PFO*", events.backazimuth[jj])
        st = __add_radial_and_transverse_channel(st, "BSPF", events.backazimuth[jj])

        if len(st) != 19:
            print(f"{len(st)} is not 19")

        ## processing data stream
        st = st.detrend("linear")
        st = st.taper(0.01)
        st = st.filter("highpass", freq=0.01, corners=4, zerophase=True)

        ## compute maximal amplitude values (PGA and PGR)
        data_amax = __compute_Amax(header, st, data_amax, events.trigger_time[jj], win_length_sec=15)
        row = {**{"origin":events.origin[jj]}, **data_amax}
        df_amax.loc[len(df_amax)] = row

        ## compute signal-to-noise ratios
        data_snr = __compute_SNR(header, st, data_snr, events.trigger_time[jj], win_length_sec=15, plot=False, plot_save=False)
        row = {**{"origin":events.origin[jj]}, **data_snr}
        df_snr.loc[len(df_snr)] = row

        # ## compute signal-to-noise ratios
        # data_adr = __compute_adr_max(header, st, data_adr, events.trigger_time[jj], win_length_sec=15)
        # row = {**{"origin":events.origin[jj]}, **data_adr}
        # df_adr.loc[len(df_adr)] = row

    except Exception as e:
        print(e)
        print(f" -> failed for {event}")
        skipped += 1
        continue


