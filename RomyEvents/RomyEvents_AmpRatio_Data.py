#!/usr/bin/env python
# coding: utf-8

# # ROMY Events - Data

# In[1]:


import os
import gc
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from andbro__store_as_pickle import __store_as_pickle


# In[2]:


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename in ['lin-ffb-01', 'hochfelln', 'ambrym']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


#import matplotlib
#matplotlib.use('TkAgg')

# ### Configurations


config = {}

# path to data
config['path_to_data'] = data_path+"romy_events/data/"

config['path_to_figs'] = data_path+"romy_events/figures/"

config['path_to_mseed'] = data_path+"romy_events/data/waveforms/ACC/"

# specify event file
config['eventfile'] = "ROMYevents_2019_2024_status_select_z.pkl"

# ROMY coordinates
config['sta_lon'] = 11.275501
config['sta_lat'] = 48.162941

# config['amp_type'] = "maxima" # maxima mean perc95  -> is set as loop

# ### Load Catalog

events_z = pd.read_pickle(config['path_to_data']+config['eventfile'])



# ### Pick maximal Amplitudes in Fband


def __get_event_window(st0, deltaT1=60, deltaT2=2, plot=False):

    from obspy.signal.trigger import coincidence_trigger
    from obspy.signal.trigger import recursive_sta_lta
    from obspy.signal.trigger import plot_trigger

    st_trig = obs.Stream()
    st_trig += st0.select(station="FUR", channel="*Z").copy()
    st_trig += st0.select(station="FUR", channel="*N").copy()
    st_trig += st0.select(station="FUR", channel="*E").copy()
    st_trig += st0.select(station="ROMY", channel="*Z").copy()

    st_trig = st_trig.detrend("demean")
    st_trig = st_trig.filter("bandpass", freqmin=0.01, freqmax=0.1, corners=4, zerophase=True)

    df = st_trig[0].stats.sampling_rate

    sta = 10 # seconds
    lta = 180 # seconds

    thr_on = 4
    thr_off = 0.1

    # cft = recursive_sta_lta(st_trig[3].data, int(sta * df), int(lta * df))
    # plot_trigger(st_trig[3], cft, thr_on, thr_off)

    trig = coincidence_trigger("recstalta", thr_on, thr_off, st_trig, 4, sta=sta, lta=lta, details=True, similarity_threshold=0.)

    t1 = trig[0]['time'] - deltaT1
    t2 = trig[0]['time'] + trig[0]['duration'] * deltaT2

    if plot:

        Nrow, Ncol = len(st_trig), 1

        font = 12

        fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 8), sharex=True)

        plt.subplots_adjust(hspace=0)

        for _k, tr in enumerate(st_trig):

            if "ROMY" in tr.stats.station:
                ax[_k].plot(tr.times(), tr.data*1e9, color="k", label=f"{tr.stats.station}.{tr.stats.channel}")
            else:
                ax[_k].plot(tr.times(), tr.data*1e6, color="k", label=f"{tr.stats.station}.{tr.stats.channel}")

            ax[_k].axvline(trig[0]['time']-tr.stats.starttime - deltaT1, -10, 10)
            ax[_k].axvline(trig[0]['time']-tr.stats.starttime + trig[0]['duration'] * deltaT2, -10, 10)


        for _n in range(Nrow):
            ax[_n].grid(ls=":", zorder=0)
            ax[_n].legend(loc=1)

        plt.show();


    return t1, t2



def __get_fband_amplitude(st0, fmin, fmax, t1, t2, amp="maxima", plot=False):

    from functions.get_octave_bands import __get_octave_bands
    from scipy.signal import hilbert

    st_amp = obs.Stream()
    st_amp += st0.select(station="FUR", channel="*Z").copy()
    st_amp += st0.select(station="FUR", channel="*N").copy()
    st_amp += st0.select(station="FUR", channel="*E").copy()
    st_amp += st0.select(station="WET", channel="*Z").copy()
    st_amp += st0.select(station="WET", channel="*N").copy()
    st_amp += st0.select(station="WET", channel="*E").copy()
    st_amp += st0.select(station="ROMY",channel="*Z").copy()
    st_amp += st0.select(station="RLAS",channel="*Z").copy()

    st_amp = st_amp.trim(t1, t2)
    st_amp = st_amp.detrend("demean")


    flower, fupper, fcenter = __get_octave_bands(fmin, fmax, faction_of_octave=6, plot=False)

    out = {}
    for fl, fu, fc in zip(flower, fupper, fcenter):

        out[fc] = {}

        stx = st_amp.copy()

        df = stx[0].stats.sampling_rate

        stx = stx.detrend("linear")

        stx = stx.taper(0.05, type="cosine")

        # stx.plot(equal_scale=False);

        # zero padding to avoid filter shift effect
        Tpadding = 4*3600 # seconds
        Npadding = int(Tpadding*df)


        for tr in stx:
            # tr.data = np.pad(tr.data, (Npadding, Npadding), 'constant', constant_values=(tr.data[0], tr.data[-1]))
            tr.data = np.pad(tr.data, (Npadding, Npadding), 'constant', constant_values=(0, 0))
            tr.stats.npts = tr.stats.npts + 2*Npadding


        stx = stx.filter("bandpass", freqmin=fl, freqmax=fu, corners=4, zerophase=True)

        stx = stx.taper(0.01, type="cosine")

        for tr in stx:
            name = f"{tr.stats.station}.{tr.stats.channel}"
            if amp == "maxima":
                out[fc][name] = np.nanmax(abs(tr.data))
            elif amp == "mean":
                out[fc][name] = np.nanmean(abs(tr.data))
            elif amp == "perc95":
                out[fc][name] = np.nanpercentile(abs(tr.data), 95)
            elif amp == "envelope":
                out[fc][name] = np.nanmax(abs(hilbert(tr.data)))

        del stx

    if plot:

        plt.figure(figsize=(15, 5))
        for _i, fc in enumerate(out.keys()):

            if _i == 0:
                plt.scatter(fc, out[fc]["ROMY.BJZ"], color="tab:blue", edgecolor="k", label="ROMY", zorder=2)
                plt.scatter(fc, out[fc]["RLAS.BJZ"], color="tab:orange", edgecolor="k", label="RLAS", zorder=2)
            else:
                plt.scatter(fc, out[fc]["ROMY.BJZ"], color="tab:blue", edgecolor="k", zorder=2)
                plt.scatter(fc, out[fc]["RLAS.BJZ"], color="tab:orange", edgecolor="k", zorder=2)

        plt.xscale("log")
        plt.yscale("log")
        plt.grid(which="both", zorder=0, alpha=0.5, color="grey")
        plt.legend(loc=2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (rad/s)")
        plt.show();

    return out


def __get_ffts(st):

    from functions.get_fft import __get_fft
    from functions.get_fband_average import __get_fband_average

    stx = st.copy()

    stx = stx.detrend("linear")
    stx = stx.detrend("demean")
    stx = stx.taper(0.01)

    ffts = {}
    for tr in stx:

        ff, px, pha = __get_fft(tr.data, tr.stats.delta, window=None)

        out = __get_fband_average(ff, px, faction_of_octave=12, average="median")

        code = f"{tr.stats.station}_{tr.stats.channel}"
        ffts[code] = out['psd_means']

    ffts['freq'] = out['fcenter']

    del stx

    return ffts


def __make_control_plot(ev_num, st0, out, t1, t2, path_to_figs, plot=False):

    import gc
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    st1 = obs.Stream()
    st1 += st0.select(station="FUR", channel="*Z").copy()
    st1 += st0.select(station="FUR", channel="*N").copy()
    st1 += st0.select(station="FUR", channel="*E").copy()
    st1 += st0.select(station="WET", channel="*Z").copy()
    st1 += st0.select(station="WET", channel="*N").copy()
    st1 += st0.select(station="WET", channel="*E").copy()
    st1 += st0.select(station="ROMY", channel="*Z").copy()
    st1 += st0.select(station="RLAS", channel="*Z").copy()

    st1 = st1.detrend("demean")
    st1 = st1.filter("bandpass", freqmin=0.01, freqmax=0.1, corners=4, zerophase=True)

    Nrow, Ncol = 10, 1

    fig = plt.figure(figsize=(15, 14))

    gs = GridSpec(Nrow, Ncol, figure=fig, hspace=0)

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, :])
    ax5 = fig.add_subplot(gs[5, :])
    ax6 = fig.add_subplot(gs[6, :])
    ax7 = fig.add_subplot(gs[7, :])

    gs2 = plt.GridSpec(Nrow, Ncol, figure=fig, hspace=2, top=0.95)
    ax8 = fig.add_subplot(gs2[8:10, :])


    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    tscale = 1/60

    for ax, tr in zip(axes, st1):
        if "ROMY" in tr.stats.station or "RLAS" in tr.stats.station:
            ax.plot(tr.times()*tscale, tr.data*1e9, color="k", label=f"{tr.stats.station}.{tr.stats.channel}")
            ax.set_ylabel("Amplitude \n (rad/s)")
            ax.set_ylim(-np.amax(tr.data*1e9),np.amax(tr.data*1e9))
        else:
            ax.plot(tr.times()*tscale, tr.data*1e6, color="k", label=f"{tr.stats.station}.{tr.stats.channel}")
            ax.set_ylabel("Amplitude \n ($\mu$m/s$^2$)")
            ax.set_ylim(-np.amax(tr.data*1e6),np.amax(tr.data*1e6))
        ax.legend(loc=1)

        ax.axvline((t1-tr.stats.starttime)*tscale, -10, 10)
        ax.axvline((t2-tr.stats.starttime)*tscale, -10, 10)

        ax.fill_between([(t1-tr.stats.starttime)*tscale, (t2-tr.stats.starttime)*tscale],
                         -1e3, 1e3,
                        alpha=0.3
                        )

    ax7.set_xlabel("Time (min)")

    ax81 = ax8.twinx()
    for _i, fc in enumerate(out.keys()):

        if _i == 0:
            ax8.scatter(fc, out[fc]["ROMY.BJZ"], color="tab:blue", edgecolor="k", label="ROMY", zorder=2)
            ax8.scatter(fc, out[fc]["RLAS.BJZ"], color="tab:orange", edgecolor="k", label="RLAS", zorder=2)
        else:
            ax8.scatter(fc, out[fc]["ROMY.BJZ"], color="tab:blue", edgecolor="k", zorder=2)
            ax8.scatter(fc, out[fc]["RLAS.BJZ"], color="tab:orange", edgecolor="k", zorder=2)


        if _i == 0:
            ax81.scatter(fc, out[fc]["FUR.BHZ"], color="tab:blue", edgecolor="k", marker="d", label="FUR", zorder=2)
            ax81.scatter(fc, out[fc]["WET.BHZ"], color="tab:orange", edgecolor="k", marker="d", label="WET", zorder=2)
        else:
            ax81.scatter(fc, out[fc]["FUR.BHZ"], color="tab:blue", edgecolor="k", marker="d", zorder=2)
            ax81.scatter(fc, out[fc]["WET.BHZ"], color="tab:orange", edgecolor="k", marker="d", zorder=2)

    ax8.set_xscale("log")
    ax8.set_yscale("log")
    # ax8.grid(which="both", zorder=0, alpha=0.5, color="grey")
    ax8.legend(loc=2)
    ax8.set_xlabel("Frequency (Hz)")
    ax8.set_ylabel("Amplitude (rad/s)")

    ax81.set_ylabel("Amplitude (m/s$^2$)")
    ax81.legend(loc=1)
    ax81.set_xscale("log")
    ax81.set_yscale("log")

    # save image
    fig.savefig(path_to_figs+"auto_plots/"+f"{ev_num}.png", format="png", dpi=150, bbox_inches='tight')

    gc.collect();

    if plot:
        plt.show();
        return fig
    else:
        plt.close();


def main(config):

    fmin, fmax = 0.001, 8.0

    amp = {}
    spec = {}

    fails = []

    for atype in ["maxima", "perc95", "mean", "envelope"]:

        config['amp_type'] = atype

        for _k, (_i, ev) in enumerate(events_z.iterrows()):

            # if _k > 2:
            #     continue

            print(_k, "  ", ev.Event.replace("_filtered.png", ".mseed"))

            # specify waveform file name
            wavformfile = ev.Event.replace("_filtered.png", ".mseed")

            # load waveform data
            try:
                st0 = obs.read(config['path_to_mseed']+wavformfile)
            except:
                print(f" -> failed to load data")
                continue

            st0 = st0.detrend("demean")

            try:
                # specify event number
                ev_num = str(int(ev['# Event'])).rjust(3, "0")

                # get window of event
                # t1, t2 = __get_event_window(st0, deltaT1=60, deltaT2=2, plot=False)
                tbeg = st0[0].stats.starttime
                tend = st0[0].stats.endtime

                t1 = tbeg + ev.T1

                if ev.T2 == 0:
                    t2 = tend
                else:
                    t2 = tbeg + ev.T2

                # get maxima for fbands
                out = __get_fband_amplitude(st0, fmin, fmax, t1, t2, amp=config['amp_type'], plot=False)

                # store check up plot
                __make_control_plot(ev_num, st0, out, t1, t2, config['path_to_figs'], plot=False);

                # add maxima to dict
                amp[ev_num] = out

                # compute spectra
                ffts = __get_ffts(st0)

                # add spec to dict
                spec[ev_num] = ffts

            except Exception as e:
                print(f" -> processing failed!")
                print(e)
                fails.append(ev_num)
                continue

            gc.collect();

            del st0

        # report
        print(f" -> failed:")
        for fail in fails:
            print(f"  -> {fail}")

        # store data
        print(f" -> stored data: {config['path_to_data']}amplitudes_{config['amp_type']}.pkl")
        __store_as_pickle(amp, config['path_to_data']+f"amplitudes_{config['amp_type']}.pkl")

        print(f" -> stored data: {config['path_to_data']}spectra.pkl")
        __store_as_pickle(spec, config['path_to_data']+f"spectra.pkl")


if __name__ == "__main__":

    main(config)

# End of File



