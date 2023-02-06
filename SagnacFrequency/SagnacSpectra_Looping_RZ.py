
#!/usr/bin/env python
# coding: utf-8

# ## Compute Sagnac Frequency


import os, gc, json
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from scipy.signal import welch, periodogram
from numpy import zeros, argmax, arange, array, linspace, shape
from tqdm import tqdm
from pandas import DataFrame, date_range
from datetime import datetime, date

from andbro__querrySeismoData import __querrySeismoData
from andbro__calculate_propabilistic_distribution import __calculate_propabilistic_distribution
from andbro__cut_frequencies_array import __cut_frequencies_array

import warnings
warnings.filterwarnings('ignore')


## Configuration

config = {}

config['ring'] = "Z"

config['seed_raw'] = f"BW.DROMY..FJ{config['ring']}"
config['seed_rot'] = f"BW.ROMY.10.BJ{config['ring']}"

config['tbeg'] = "2022-08-01"
config['tend'] = "2022-08-31"

config['outpath_data'] = f"/import/kilauea-data/sagnac_spectra/R{config['ring']}/data/"
config['outpath_figs'] = f"/import/kilauea-data/sagnac_spectra/R{config['ring']}/figures/"


config['save_plots'] = True

config['repository'] = "george"

config['method'] = "welch" ## "welch" | "periodogram" | multitaper

config['rings'] = {"Z":553, "U":302, "V":448,"W":448}

config['f_expected'] = config['rings'][config['ring']]  ## expected sagnac frequency
config['f_band'] = 10 ## +- frequency band

config['segment_factor'] = 600 ## seconds

## Variables

config['threshold'] = -10

config['dn'] = 3600  ## seconds
config['buffer'] = 0  ## seconds
config['offset'] = 30 ## seconds

config['loaded_period'] = 3600  ## seconds

config['NN'] = int(config['loaded_period']/config['dn'])

config['interval'] = config['loaded_period']




## Methods

def __multitaper_estimate(data, fs, n_windows=4, one_sided=True):

    from spectrum import dpss, pmtm
    from numpy import zeros, arange, linspace

    NN = len(data)

    spectra, weights, eigenvalues = pmtm(data, NW=2.5, k=n_windows, show=False)

    ## average spectra
    estimate = zeros(len(spectra[0]))
    for m in range(n_windows):
        estimate += (abs(spectra[m])**2)
    estimate /= n_windows

    l = len(estimate)
    frequencies = linspace(-0.5*fs, 0.5*fs, l)

    if one_sided:
        f_tmp, psd_tmp = frequencies[int(l/2):], estimate[:int(l/2)]
    else:
        f_tmp, psd_tmp = frequencies, estimate


    f_max = f_tmp[argmax(psd_tmp)]
    p_max = max(psd_tmp)
    h_tmp = __half_width_half_max(psd_tmp)

    return f_tmp, f_max, p_max, h_tmp


def __multitaper_periodogram(data, fs, n_windows=4):

    from spectrum import dpss, pmtm
    from numpy import zeros, arange, linspace
    from scipy.signal import find_peaks, peak_widths

    [tapers, eigen] = dpss(len(data), 2.5, n_windows)

    f_maxima, p_maxima, hh  = zeros(n_windows), zeros(n_windows), zeros(n_windows)

    for ii in range(n_windows):

        f_tmp, psd_tmp = periodogram(data,
                                     fs=fs,
                                     window=tapers[:,ii],
                                     nfft=None,
                                     detrend='constant',
                                     return_onesided=True,
                                     scaling='density',
                                     )

        p_maxima[ii] = max(psd_tmp)
        f_maxima[ii] = f_tmp[argmax(psd_tmp)]

        ## half widths
        xx = psd_tmp[argmax(psd_tmp)-10:argmax(psd_tmp)+10]
        peaks, _ = find_peaks(xx)
        half = peak_widths(xx, peaks, rel_height=0.5)
        idx = argmax(half[1])
        hh[ii] = abs(half[3][idx] -half[2][idx])

    f_mean = sum(f_maxima) / n_windows
    p_mean = sum(p_maxima) / n_windows
    h_mean = sum(hh) / n_windows

    return f_tmp, f_mean, p_mean, h_mean


def __hilbert_frequency_estimator(config, st, fs):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()

    f_lower = config['f_expected'] - config['f_band']
    f_upper = config['f_expected'] + config['f_band']

    ## bandpass with butterworth
    st0.detrend("demean")
    st0.taper(0.1)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)

    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    ## cut first and last 5% (corrupted)

    dd = int(0.05*len(instantaneous_frequency))

    t = st0[0].times()
    t1 = st0[0].times()[1:]
    t2 = t1[dd:-dd]

    t_mid = t[int((len(t))/2)]

    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## averaging
    insta_f_cut_mean = np.mean(insta_f_cut)
#     insta_f_cut_mean = np.median(insta_f_cut)

    return t_mid, insta_f_cut_mean, np.mean(amplitude_envelope) ,np.std(insta_f_cut)


def __compute(config, st0, starttime, method="hilbert"):

    from scipy.signal import find_peaks, peak_widths, welch, periodogram
    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['dn']

    tt, ff, hh, pp = zeros(NN), zeros(NN), zeros(NN), zeros(NN)

    while n2 <= config['loaded_period']:

        try:

            ## cut stream to chuncks
            st_tmp = st0.copy().trim(starttime+n1-config['buffer']-config['offset'], starttime+n1+config['dn']+config['buffer']-config['offset'])

            ## get time series from stream
            times = st_tmp[0].times(reftime=UTCDateTime("2016-01-01T00"))

            ## get sampling rate from stream
            df = st_tmp[0].stats.sampling_rate


            if method == "hilbert":
                f_tmp, f_max, p_max, h_tmp = __hilbert_frequency_estimator(config, st_tmp, df)

            elif method == "multitaper_periodogram":

                f_tmp, f_max, p_max, h_tmp = __multitaper_perio(st_tmp[0].data, df, n_windows=config['n_windows'])

            elif method == "periodogram":

                f_tmp, f_max, p_max, h_tmp = __periodogram_estimate(st_tmp, df)

            elif method == "multitaper":

                f_tmp, f_max, p_max, h_tmp = __multitaper_estimate(st_tmp[0].data, df, n_windows=config['n_windows'], one_sided=True)


            ## append values to arrays
            tt[ii] = times[int(len(times)/2)]
            ff[ii] = f_max
            pp[ii] = p_max
            hh[ii] = h_tmp

        except:
            tt[ii], ff[ii], pp[ii], hh[ii] = nan, nan, nan, nan
#            print(" -> computing failed")

        ii += 1
        n1 += config['dn']
        n2 += config['dn']

    return tt, ff, hh, pp


def __makeplot_colorlines(config, ff, data, smooth=None):

    from numpy import log10, median
    
    def __get_median_psd(psds):

        from numpy import median, zeros, isnan

        med_psd = zeros(psds.shape[1])

        for f in range(psds.shape[1]):
            a = psds[:,f]
            med_psd[f] = median(a[~isnan(a)])

        return med_psd

    def __smooth(y, box_pts):
        from numpy import ones, convolve, hanning

        win = hanning(box_pts)
        y_smooth = convolve(y, win/sum(win), mode='same')

        return y_smooth

    
    
    cols = plt.cm.jet_r(linspace(0,1,shape(data)[0]+1))
#     cols = plt.cm.viridis(linspace(0,1,shape(data)[0]+1))
    
    ## ____________________________________________

    fig, ax = plt.subplots(1,1, figsize=(15,10))

    font = 14
    
    data_min = min([min(d) for d in data])
    data_max = max([max(d) for d in data])

    for i, psdx in enumerate(data):
        
        if smooth is not None:
            ax.plot(ff, __smooth(psdx,smooth), color=cols[i], zorder=2, label=i,  alpha=0.3)
        else:
            ax.plot(ff, psdx, color=cols[i], zorder=2, label=i,  alpha=0.3)

    ## select only psds above a median threshold for median computation
    psd_select = array([dat for dat in data if median(log10(dat)) > config['threshold']])
    try:
        psd_median = __get_median_psd(psd_select)
        ax.plot(ff, psd_median, color='k', lw=1, zorder=2)
    except:
        print(" -> median computation failed!")
    
    ax.set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax.set_yscale("log")
    ax.set_ylim(data_min-0.01*data_min, data_max+0.5*data_max)
    
    leg = ax.legend(ncol=2)

    # change the line width for the legend
    [line.set_linewidth(3.0) for line in leg.get_lines()]


    ax.grid(ls='--', zorder=1)

    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax.set_title(f"Sagnac Spetra on {date} ({config['interval']}s windows) ", fontsize=font+2)

    ax.tick_params(axis='both', labelsize=font-2)

#     plt.show();
    return fig


def __makeplot_colorlines_and_helicorder(config, ff, data, traces, peaks=None, smooth=None):

    from numpy import log10, median

    def __get_median_psd(psds):

        from numpy import median, zeros, isnan

        med_psd = zeros(psds.shape[1])

        for f in range(psds.shape[1]):
            a = psds[:,f]
            med_psd[f] = median(a[~isnan(a)])

        return med_psd

    ## extract colors from colormap
    cols = plt.cm.jet_r(linspace(0,1,shape(data)[0]+1))

    ## ____________________________________________

    fig, ax = plt.subplots(1,2, figsize=(18,8))

    plt.subplots_adjust(wspace=0.15)

    font = 14

    data_min = min([min(d) for d in data])
    data_max = max([max(d) for d in data])

    for i, psdx in enumerate(data):

        if smooth is not None:
            ax[0].plot(ff, __smooth(psdx,smooth), color=cols[i], zorder=2, label=i,  alpha=0.3)
        else:
            ax[0].plot(ff, psdx, color=cols[i], zorder=2, label=i,  alpha=0.3)

    ## select only psds above a median threshold for median computation
    psd_select = array([dat for dat in data if median(log10(dat)) > config['threshold']])
    try:
        psd_median = __get_median_psd(psd_select)
        ax[0].plot(ff, psd_median, color='k', lw=1, zorder=2)
    except:
        print(" -> median computation failed!")

    ax[0].set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax[0].set_yscale("log")
    ax[0].set_ylim(data_min-0.01*data_min, data_max+0.5*data_max)

    ## insert legend
    leg = ax[0].legend(ncol=2)

    # change the line width for the legend
    [line.set_linewidth(3.0) for line in leg.get_lines()]


    ax[0].grid(ls='--', zorder=1)

    ax[0].set_xlabel("Frequency (Hz)", fontsize=font)
    ax[0].set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax[0].set_title(f"Sagnac Spetra on {config['tbeg']} ({config['interval']}s windows) ", fontsize=font+2)

    ax[0].tick_params(axis='both', labelsize=font-2)

    ## ___________________________________
    ## PLOT 2

#     norm_st_max = np.max(traces)
    timeaxis = linspace(0, 60, len(traces[0]))

    for m, tr in enumerate(traces):

        norm_tr_max = max(tr)

        ax[1].plot(timeaxis, tr/norm_tr_max + m, color=cols[m], alpha=0.3)

    ax[1].set_yticks(linspace(0,23,24))
    ax[1].set_yticklabels([str(int(tt)).rjust(2,"0")+":00" for tt in linspace(0,23,24)])

    ax[1].set_ylim(-1, 24)

    ax[1].tick_params(axis='both', labelsize=font-2)

#    plt.show();
    return fig


def __makeplot_distribution(config, xx, yy, dist, overlay=False):

    from numpy import nanmax, nanmin
    from matplotlib import colors

    def __smooth(y, box_pts):
        from numpy import ones, convolve, hanning

        win = hanning(box_pts)
        y_smooth = convolve(y, win/sum(win), mode='same')

        return y_smooth



    cmap = plt.cm.get_cmap("YlOrRd")
#     cmap = plt.cm.get_cmap("viridis")
    cmap.set_bad("white")
    cmap.set_under("white")

    max_psds = nanmax(dist)
    min_psds = nanmin(dist)


    ## ____________________________________________

    fig, ax = plt.subplots(1,1, figsize=(15,10))

    font = 14

    im = ax.pcolormesh( xx, yy, dist.T,
                        cmap=cmap,
                        vmax=max_psds,
                        vmin=min_psds+0.01*min_psds,
                        norm=colors.LogNorm(),
                        )

    if overlay is not None:
        ax.plot(xx, __smooth(10**overlay, 50), color='k', alpha=0.6, lw=1, zorder=2, label="maxima")

    ax.set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax.set_yscale("log")

    ax.legend(ncol=2)

    ax.grid(ls='--', zorder=1)

    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax.set_title(f"Sagnac Spetra on {config['xdate']} ({config['interval']}s windows) ", fontsize=font+2)

    ax.tick_params(axis='both', labelsize=font-2)

    cb = plt.colorbar(im, ax=ax, anchor=(0.0, -0.5))
    cb.set_label("Propability Density", fontsize=font, labelpad=-60)

#     plt.show();
    return fig


def __get_welch_psd(config, arr, df):

    segments = df*config['segment_factor']

    f0, psd0 = welch(
                    arr,
                    fs=df,
                    window='hanning',
                    nperseg=segments,
                    noverlap=int(segments/2),
                    nfft=None,
                    detrend='constant',
                    return_onesided=True,
                    scaling='density',
                    )

    return f0, psd0


def __save_to_pickle(obj, filename):

    import pickle

    if not filename.split("/")[-1].split(".")[-1] == "pkl":
        filename = filename+".pkl"

    with open(filename, 'wb') as ofile:
        pickle.dump(obj, ofile)

    if os.path.isfile(filename):
        print(f" -> created: {filename}")


def __smooth(y, box_pts):
    from numpy import ones, convolve, hanning

    win = hanning(box_pts)
    y_smooth = convolve(y, win/sum(win), mode='same')

    return y_smooth


def __check_path(path):
    created=False
    if not os.path.exists(path):
        os.mkdir(path)
        created=True
    if created and os.path.exists(path):
        print(f" -> created: {path}")




###################################
### MAIN ##########################
###################################

def main():


    tbeg = date.fromisoformat(config['tbeg'])
    tend = date.fromisoformat(config['tend'])

    print(json.dumps(config, indent=4, sort_keys=True))


    ### ---------------------------------------------
    ## looping days
    for xdate in date_range(tbeg, tend):

        print(xdate)
        config['xdate'] = xdate

        idx_count=0
        NNN = int(86400/config['dn'])

        psds, traces = [], []

        ### ---------------------------------------------
        ## looping hours
        for hh in tqdm(range(NNN)):
#        for hh in tqdm(range(2)):

            ## define current time window
            dh = hh*config['loaded_period']

            t1, t2 = UTCDateTime(xdate)+dh, UTCDateTime(xdate)+config['loaded_period']+dh

            try:
                ## load data for current time window
#                print(" -> loading data ...")
                st_raw, inv_raw = __querrySeismoData(
                                                     seed_id=config['seed_raw'],
                                                     starttime=t1-2*config['offset'],
                                                     endtime=t2+2*config['offset'],
                                                     repository=config['repository'],
                                                     path=None,
                                                     restitute=None,
                                                     detail=None,
                                                    )
            except:
                print(" -> failed to load raw data!")
                continue

            try:
                ## load data for current time window
#                print(" -> loading data ...")
                st_rot, inv_rot = __querrySeismoData(
                                                     seed_id=config['seed_rot'],
                                                     starttime=t1-2*config['offset'],
                                                     endtime=t2+2*config['offset'],
                                                     repository=config['repository'],
                                                     path=None,
                                                     restitute=True,
                                                     detail=None,
                                                    )
            except:
                print(" -> failed to load rot data!")
                continue


            st_rot[0].trim(t1, t2)
            st_raw[0].trim(t1, t2)

            ## convert from counts to volts
            st_raw[0].data = st_raw[0].data * 0.59604645e-6


#            print(" -> computing welch ...")
            try:
                ff, psd = __get_welch_psd(config, st_raw[0].data, st_raw[0].stats.sampling_rate)
            except Exception as e:
                print(e)

            psds.append(psd)
            traces.append(st_rot[0].data)

            del st_raw, st_rot
            gc.collect()

        if len(psds) == 0:
            continue

        ## generate output object
        output = {}
        output['frequencies'] = ff
        output['psds'] = array(psds)

        ## store output
        date_str = str(xdate)[:10].replace("-","")
        __save_to_pickle(output, f"{config['outpath_data']}R{config['ring']}_{date_str}.pkl")


        ## limit frequency range for plotting
        try:
            f_min , f_max = config['f_expected']-config['f_band'], config['f_expected']+config['f_band']
            psds, ff = __cut_frequencies_array(array(psds), ff, f_min, f_max)
        except:
            print(f" -> failed to cut frequeny range!")

        ## Plotting
        try:
            colorlines = __makeplot_colorlines(config, ff, array(psds), smooth=None);
        except Exception as e:
            print(" -> failed to plot colorlines!")
            print(e)

        try:
            colorlines_smooth = __makeplot_colorlines(config, ff, array(psds), smooth=20);
        except Exception as e:
            print(" -> failed to plot colorlines smooth!")
            print(e)

        try:
            colorlines_heli = __makeplot_colorlines_and_helicorder(config, ff, array(psds), traces, peaks=None, smooth=None);
        except Exception as e:
            print(" -> failed to plot colorlines heli!")
            print(e)

#         out = __calculate_propabilistic_distribution(psds, bins=50, density=True, y_log_scale=True, axis=0)
#         distribution = __makeplot_distribution(config, ff, out['bin_mids'], out['dist'], overlay=out['bins_maximas']);


        if config['save_plots']:

            try:
                ### PLOT 1 -----------------
                outname = f"plot_sagnacspectra_{date_str}_{config['loaded_period']}_colorlines.png"
                subdir = f"normal/"

                __check_path(f"{config['outpath_figs']}{subdir}")

                colorlines.savefig(
                                    f"{config['outpath_figs']}{subdir}{outname}",
                                    dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                                    format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
                                   )
                print(f" -> saving: {config['outpath_figs']}{subdir}{outname}...")
            except Exception as e:
                print(e)
                pass

            try:
                ### PLOT 2 -----------------
                outname = f"plot_sagnacspectra_{date_str}_{config['loaded_period']}_colorlines_smooth.png"
                subdir = f"smooth/"

                __check_path(f"{config['outpath_figs']}{subdir}")

                colorlines_smooth.savefig(
                                          f"{config['outpath_figs']}{subdir}{outname}",
                                          dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                                          format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
                                         )
                print(f" -> saving: {config['outpath_figs']}{subdir}{outname}...")
            except Exception as e:
                print(e)
                pass

            try:
                ### PLOT 3 -----------------
                outname = f"plot_sagnacspectra_{date_str}_{config['loaded_period']}_colorlines_helicorder.png"
                subdir = f"with_helicorder/"

                __check_path(f"{config['outpath_figs']}{subdir}")

                colorlines_heli.savefig(
                                    f"{config['outpath_figs']}{subdir}{outname}",
                                    dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                                    format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
                                    )
                print(f" -> saving: {config['outpath_figs']}{subdir}{outname}...")
            except Exception as e:
                print(e)
                pass


if __name__ == "__main__":
    main()


## END OF FILE
