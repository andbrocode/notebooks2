#!/bin/python3

import os
import gc
import obspy
import matplotlib.pyplot as plt
import pandas as pd

from numpy import arange, linspace, sqrt, diff, nan, gradient, nanmax, nanmean, array
from pandas import read_csv, DataFrame
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.ma import filled, isMaskedArray, masked
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from obspy import UTCDateTime
from pathlib import Path

from andbro__readYaml import __readYaml
from andbro__read_sds import __read_sds

from functions.smoothing import __smooth
from functions.reduce import __reduce
from functions.converstion_to_tilt import __conversion_to_tilt
from functions.plot_all_tilt import __plot_all_tilt

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
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


config = {}

## decide to store figures
config['save'] = True

## set time period
# config['tbeg'] = UTCDateTime("2024-02-23 12:00")
# config['tend'] = UTCDateTime("2024-03-07 12:00")

# in southern shaft
config['tbeg'] = UTCDateTime("2024-03-08 12:00")
config['tend'] = UTCDateTime("2024-05-08 00:00")

## specify paths
config['path_to_sds'] = archive_path+"romy_archive/"

config['path_to_data'] = data_path+"TiltmeterDataBackup/Tilt_downsampled/"

config['path_to_figs'] = data_path+"tiltmeter/figures/"


## tiltmeter configurations
confTilt = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/", "tiltmeter.conf")

## correction of offset (e.g. reset mass)
# offset_correction = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/", "tiltmeter_steps.yml")
offset_correction = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/", "tiltmeter_offsets.yml")

## correction for temperature trends
## based on MAT
temperature_correction = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/","tiltmeter_temperature_correction.yml")
## based on WSX
# temperature_correction = __readYaml(f"{root_path}Documents/ROMY/tiltmeter/","temperature_correction_new.yml")

def main(config):

    print(f" -> load ROMYT data ...")
    ROMYT0 = __read_sds(config['path_to_sds'], "BW.ROMYT..MA*", config['tbeg'], config['tend'])

    ROMYT0 = ROMYT0.sort()

    ROMYT0 = ROMYT0.merge(fill_value="interpolate")

    ROMYT = __conversion_to_tilt(ROMYT0, confTilt['ROMYT'])

    del ROMYT0

    print(f" -> load FUR data ...")
    fur = __read_sds(bay_path+"mseed_online/archive/", "GR.FUR..BH*", config['tbeg'], config['tend'])

    fur_inv = obspy.read_inventory(root_path+"/Documents/ROMY/stationxml_ringlaser/dataless/dataless.seed.GR_FUR")

    fur = fur.remove_response(inventory=fur_inv, output="ACC", water_level=10)

    fur = fur.merge(fill_value="interpolate")


    stt = obspy.Stream()
    stt += ROMYT.copy()
    stt += fur.copy()

    del fur, ROMYT

    for tr in stt:
        if tr.stats.station == "ROMYT" and "T" not in tr.stats.channel:
            tr.data = __reduce(tr.data, 1000)
            tr.data = tr.data*-9.81

    # stt = stt.detrend("demean");
    stt = stt.detrend("linear");
    # stt = stt.detrend("simple");

    stt = stt.taper(0.05, type="cosine");

    ff = 0.25
    stt = stt.filter("lowpass", freq=ff, corners=4, zerophase=True);
    stt = stt.resample(4*ff, no_filter=True);

    # stt = stt.filter("bandpass", freqmin=1/(100*3600), freqmax=0.25, corners=4, zerophase=True);
    # stt = stt.resample(1, no_filter=True);

    stt.plot(equal_scale=False);


    def __makeplot(st0):

        from functions.get_fband_average import __get_fband_average
        from functions.multitaper_coherence import __multitaper_coherence
        from functions.welch_coherence import __welch_coherence
        from functions.reduce import __reduce
        from functions.get_fft import __get_fft

        dat01 = st0.select(station="ROMYT", channel="*N")[0].data
        dat02 = st0.select(station="ROMYT", channel="*E")[0].data

        dat21 = st0.select(station="FUR", channel="*N")[0].data
        dat22 = st0.select(station="FUR", channel="*E")[0].data


        dt, df = st0[0].stats.delta, st0[0].stats.sampling_rate

        win = 20*86400
        # win = 6*86400

        # compute multitaper spectrum
        # out01 = __multitaper_coherence(dat01, dat21, dt, n_taper=15, time_bandwidth=3., method=0)
        out01 = __welch_coherence(dat01, dat21, dt, twin_sec=win)

        # ff11, psd11 = out01['ff1'][1:], out01['psd1'][1:]
        # ff12, psd12 = out01['ff2'][1:], out01['psd2'][1:]
        ff1_coh, coh1 = out01['fcoh'][1:], out01['coh'][1:]

        # out02 = __multitaper_coherence(dat02, dat22, dt, n_taper=15, time_bandwidth=3., method=0)
        out02 = __welch_coherence(dat02, dat22, dt, twin_sec=win)

        # ff21, psd21 = out02['ff1'][1:], out02['psd1'][1:]
        # ff22, psd22 = out02['ff2'][1:], out02['psd2'][1:]
        ff2_coh, coh2 = out02['fcoh'][1:], out02['coh'][1:]

        # compute FFT spectrum
        ff11, psd11, pha11 = __get_fft(dat01, df)
        ff12, psd12, pha12 = __get_fft(dat21, df)
        ff21, psd21, pha21 = __get_fft(dat02, df)
        ff22, psd22, pha22 = __get_fft(dat22, df)

        # compute average on octave bands
        octa11 = __get_fband_average(ff11, psd11, faction_of_octave=12, fmin=1e-6, average="mean")
        ff11, psd11 = octa11['fcenter'], octa11['psd_means']

        octa12 = __get_fband_average(ff12, psd12, faction_of_octave=12, fmin=1e-6, average="mean")
        ff12, psd12 = octa12['fcenter'], octa12['psd_means']

        octa21 = __get_fband_average(ff21, psd21, faction_of_octave=12, fmin=1e-6, average="mean")
        ff21, psd21 = octa21['fcenter'], octa21['psd_means']

        octa22 = __get_fband_average(ff22, psd22, faction_of_octave=12, fmin=1e-6, average="mean")
        ff22, psd22 = octa22['fcenter'], octa22['psd_means']

        octa_coh1 = __get_fband_average(ff1_coh, coh1, faction_of_octave=12, fmin=1e-6, average="mean")
        ff1_coh, coh1 = octa_coh1['fcenter'], octa_coh1['psd_means']

        octa_coh2 = __get_fband_average(ff2_coh, coh2, faction_of_octave=12, fmin=1e-6, average="mean")
        ff2_coh, coh2 = octa_coh2['fcenter'], octa_coh2['psd_means']

        # ________________________________________________________________________
        # plotting

        Nrow, Ncol = 4, 1

        font = 12

        ref_date = config['tbeg'].date

        fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 12), sharex=False)

        plt.subplots_adjust(hspace=0.25)

        # time_scaling = 1/86400
        time_scaling = 1
        acc_scaling = 1e6

        # ________________________________________________________________________
        #
        ax[0].plot(st0.select(station="ROMYT", channel="*N")[0].times(reftime=UTCDateTime(ref_date)),
                   st0.select(station="ROMYT", channel="*N")[0].data*acc_scaling, color="darkblue", label="ROMYT-N", lw=1)

        ax[0].plot(st0.select(station="FUR", channel="*N")[0].times(reftime=UTCDateTime(ref_date)),
                   st0.select(station="FUR", channel="*N")[0].data*acc_scaling, color="tab:blue", label="FUR-N", lw=1, ls="--")

        ax[0].plot(st0.select(station="ROMYT", channel="*N")[0].times(reftime=UTCDateTime(ref_date)),
                   __smooth(st0.select(station="ROMYT", channel="*N")[0].data, 500)*acc_scaling, color="white", lw=1)
        ax[0].plot(st0.select(station="FUR", channel="*N")[0].times(reftime=UTCDateTime(ref_date)),
                   __smooth(st0.select(station="FUR", channel="*N")[0].data, 500)*acc_scaling, color="white", lw=1, ls="--")

        ax[0].ticklabel_format(useOffset=False)
        ax[0].set_ylabel("Acc ($\mu$m/s$^2$)", fontsize=font)
        ax[0].legend(loc=1, ncol=4)

        # ________________________________________________________________________
        #
        ax[1].plot(st0.select(station="ROMYT", channel="*E")[0].times(reftime=UTCDateTime(ref_date)),
                   st0.select(station="ROMYT", channel="*E")[0].data*acc_scaling, color="darkred", label="ROMYT-E", lw=1)
        ax[1].plot(st0.select(station="FUR", channel="*E")[0].times(reftime=UTCDateTime(ref_date)),
                   st0.select(station="FUR", channel="*E")[0].data*acc_scaling, color="tab:red", label="FUR-E", lw=1, ls="--")

        ax[1].plot(st0.select(station="ROMYT", channel="*E")[0].times(reftime=UTCDateTime(ref_date)),
                   __smooth(st0.select(station="ROMYT", channel="*E")[0].data, 500)*acc_scaling, color="white", lw=1)
        ax[1].plot(st0.select(station="FUR", channel="*E")[0].times(reftime=UTCDateTime(ref_date)),
                   __smooth(st0.select(station="FUR", channel="*E")[0].data, 500)*acc_scaling, color="white", lw=1, ls="--")

        ax[1].ticklabel_format(useOffset=False)
        ax[1].set_ylabel("Acc ($\mu$m/s$^2$)", fontsize=font)
        ax[1].legend(loc=4, ncol=4)


        # ________________________________________________________________________
        #
        ax[2].plot(ff11, psd11, color="darkblue", label="ROMYT-N")
        ax[2].plot(ff12, psd12, color="tab:blue", label="FUR-N", ls="--")
        ax[2].plot(ff21, psd21, color="darkred", label="ROMYT-E")
        ax[2].plot(ff22, psd22, color="tab:red", label="FUR-E", ls="--")

        # ax[2].axvline(1/86400, min(psd11), max(psd11), color="grey", alpha=0.7, zorder=1)
        # ax[2].axvline(2/86400, min(psd11), max(psd11), color="grey", alpha=0.7, zorder=1)
        ax[2].axvline(1/86400, 1e-20, 1e10, color="grey", alpha=0.7, zorder=1)
        ax[2].axvline(2/86400, 1e-20, 1e10, color="grey", alpha=0.7, zorder=1)

        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        ax[2].legend(loc=1, ncol=4)
        ax[2].set_ylabel(f"PSD (m$^2$/s$^4$/Hz)", fontsize=font)

        # _______________________________________________________________________
        #
        ax[3].plot(ff1_coh, coh1, color="tab:blue", label="N")
        ax[3].plot(ff2_coh, coh2, color="tab:red", label="E")

        # ax[3].plot(out01['fcoh'][1:], __smooth(out01['coh'][1:], 200), color="k", lw=0.5)
        # ax[3].plot(out02['fcoh'][1:], __smooth(out02['coh'][1:], 200), color="k", lw=0.5, ls="--")

        ax[3].axvline(1/86400, 0, 1.1, color="grey", alpha=0.7, zorder=1)
        ax[3].axvline(2/86400, 0, 1.1, color="grey", alpha=0.7, zorder=1)

        ax[3].set_xscale("log")
        ax[3].set_ylabel("Coherence", fontsize=font)
        ax[3].set_xlabel("Frequency (Hz)", fontsize=font)
        ax[3].set_ylim(0, 1.1)
        ax[3].legend(loc=4, ncol=1)


        for _n in range(Nrow):
            ax[_n].grid(ls=":", zorder=0)
            # ax[_n].set_xlim(left=0)



        # add dates to x-axis
        tcks = ax[0].get_xticks()
        tcklbls = [f"{UTCDateTime(UTCDateTime(ref_date)+t).date} \n {UTCDateTime(UTCDateTime(ref_date)+t).time}" for t in tcks]
        ax[0].set_xticklabels(tcklbls, fontsize=font-3)
        tcks = ax[1].get_xticks()
        tcklbls = [f"{UTCDateTime(UTCDateTime(ref_date)+t).date} \n {UTCDateTime(UTCDateTime(ref_date)+t).time}" for t in tcks]
        ax[1].set_xticklabels(tcklbls, fontsize=font-3)


        # add labels for subplots
        for _k, ll in enumerate(['(a)', '(b)', '(c)', '(d)']):
            ax[_k].text(.005, .97, ll, ha='left', va='top', transform=ax[_k].transAxes, fontsize=font+2)

        # set axis limits
        ax[0].set_ylim(-12, 12)
        ax[1].set_ylim(-12, 12)

        gc.collect()

        plt.show();
        return fig

    fig = __makeplot(stt)

    fig.savefig(config['path_to_figs']+f"Tilt_ROMYT_FUR_spectra.png", format="png", dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main(config)

# End of File