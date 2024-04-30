#!/usr/bin/env python
# coding: utf-8

# # Spectral Density

# ## Load Libraries

# In[1]:


import os
import gc
import pickle
import matplotlib.pyplot as plt

from andbro__querrySeismoData import __querrySeismoData
from andbro__savefig import __savefig

from obspy import UTCDateTime
from scipy.signal import welch
from numpy import log10, zeros, pi, append, linspace, mean, median, array, where, transpose, shape, histogram, arange
from numpy import logspace, linspace, log, log10, isinf, ones, nan, count_nonzero, sqrt, isnan
from pandas import DataFrame, concat, Series, date_range, read_csv, read_pickle
from tqdm import tqdm_notebook
from pathlib import Path

from functions.get_hist_loglog import __get_hist_loglog
from functions.replace_noise_psd_with_nan import __replace_noisy_psds_with_nan
from functions.cut_frequencies_array import __cut_frequencies_array
from functions.get_median_psd import __get_median_psd
from functions.get_percentiles import __get_percentiles

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')


# In[3]:


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
elif os.uname().nodename in ['ambrym', 'lin-ffb-01', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


# ## Configurations

# In[4]:


config = {}

config['seeds'] = ["BW.ROMY..BJZ", "BW.ROMY..BJU", "BW.ROMY..BJV"]
# config['sta'] = "ROMY"
# config['cha'] = ["Z", "U", "V"]

config['d1'], config['d2'] = "2024-01-01", "2024-03-31"

config['path_to_data'] = data_path+f"LNM2/PSDS/"

config['path_to_outdata'] = data_path+f"LNM2/data/"

config['outpath_figures'] = data_path+f"LNM2/figures/PPSD/"

config['frequency_limits'] = 1e-3, 1e1


# ## Methods


# In[6]:


def __makeplot_colorlines(config, ff, psds, rejected, day):

    from tqdm.notebook import tqdm
    from numpy import isnan, median, mean, std, array, zeros
    from scipy.stats import median_abs_deviation as mad

#     psds_median = __get_median_psd(array(psds))
#     psds_minimal = __get_minimal_psd(array(psds))
#     psds_minimum = __get_minimum_psd(array(psds), ff)



    ##____________________________


    fig, axes = plt.subplots(1, 1, figsize=(15,7), sharey=False, sharex=True)

    plt.subplots_adjust(hspace=0.1)

    font = 14

    N = 24
    colors = plt.cm.rainbow(linspace(0, 1, N))
    cmap = plt.get_cmap('rainbow', 24)

    for n, psd in enumerate(psds):
        axes.loglog(ff, psd, color=colors[n], alpha=0.7)
        p2 = axes.scatter(ff[0], psd[0], s=0.1, c=int(n/N), cmap=cmap, vmin=0, vmax=N)

    for reject in rejected:
         axes.loglog(ff, reject, color='grey', alpha=0.6, zorder=1)

    axes.loglog(ff, __get_median_psd(psds), 'black', zorder=3, alpha=0.6, label="Median")

    axes.grid(True, which="both", ls="-", alpha=0.5)
    axes.legend(loc='lower left')
    axes.tick_params(labelsize=font-2)

    axes.set_xlim(1e-3, 2e1)
#     axes.set_ylim(1e-23, 1e-16)

    axes.set_xlabel("  Frequency (Hz)", fontsize=font, labelpad=-1)


    # axes.set_ylabel(r"PSD$_{absolute}$ ($hPa$/$Hz)$", fontsize=font)
    axes.set_ylabel(r"PSD$_{infrasound}$ ($hPa$/$Hz)$", fontsize=font)

    ## set colorbar at bottom
    cbar = fig.colorbar(p2, orientation='vertical', ax=axes, aspect=50, pad=-1e-5,
                       ticks=arange(1,N,2))

    axes.set_title(f"{config['station']} | {day}")

    # plt.show();
    return fig



# ## RUN for all files

# In[9]:


def __get_hist_loglog(psd_array, ff, bins=20, density=False, axis=1, plot=False):

    import matplotlib.pyplot as plt
    from numpy import argmax, std, median, isnan, array, histogram, nan, zeros, count_nonzero, isinf, log10, nanmax, nanmin, nonzero
    from scipy.stats import median_abs_deviation as mad

    def __convert_to_log(in_psds):

        out_psds = zeros(in_psds.shape)
        rows_with_zeros = 0

        for i, psd in enumerate(in_psds):
            if count_nonzero(psd) != len(psd):
                rows_with_zeros += 1
                psd = [nan for val in psd if val == 0]
            out_psds[i, :] = log10(psd)
            if isinf(out_psds[i,:]).any():
                out_psds[i, :] = nan * ones(len(out_psds[i, :]))

        print(f" -> rows with zeros: {rows_with_zeros}")

        return out_psds

    ## converting to log10
    psd_array = __convert_to_log(psd_array)

    ## exclude psds with only NaN values
    psds = array([psd for psd in psd_array if not isnan(psd).all()])
    print(f" -> total spectra used: {psd_array.shape[0]}")

    ## find overall minimum and maxium values
    # max_value = max([max(sublist) for sublist in psd_array])
    # min_value = min([min(sublist) for sublist in psd_array])
    max_value = nanmax(psd_array.reshape(psd_array.size))
    min_value = nanmin(psd_array.reshape(psd_array.size))
    # print(min_value, max_value)

    ## define empty lists
    dist, dist_maximas, bins_maximas, bins_medians, stds, mads = [], [], [], [], [], []

    count = 0
    for h in range(len(psd_array[axis])):

        psd = psd_array[:, h]

        ## compute histograms
        hist, bin_edges = histogram(psd, bins=bins, range=(min_value, max_value), density=density);

        ## center bins
        bin_mids = 0.5*(bin_edges[1:] + bin_edges[:-1])

        ## normalization
#         if True:
#             hist = [val / len(psd_array[:,h]) for val in hist]
#             config['set_density'] = True

        ## check if density works
        # DX = abs(max_value-min_value)/bins
        # SUM = sum(hist)
        # if str(SUM*DX) != "1.0":
        #     count += 1

        ## modify histogram with range increment
        # hist = hist*DX
        hist = [h / sum(hist) for h in hist]


        ## append values to list
        dist.append(hist)
        stds.append(std(hist))
        dist_maximas.append(max(hist))
        bins_maximas.append(bin_mids[argmax(hist)])
        mads.append(mad(hist))

        ## compute median
        psd = psd[~(isnan(psd))]
        bins_medians.append(median(psd[psd != 0]))

    ## undo log conversion
    output = {}
    output['dist'] = array(dist)
    output['bin_mids'] = 10**array(bin_mids)
    output['bins_maximas'] = 10**array(bins_maximas)
    output['stds'] = 10**array(stds)
    output['mads'] = 10**array(mads)
    output['bins_medians'] = 10**array(bins_medians)
    output['set_density'] = density
    output['total'] = psd_array.shape[0]
    output['frequencies'] = ff


    ## check plot
    if plot:

        fig = plt.figure(figsize=(15, 5))
        cmap = plt.colormaps.get_cmap('viridis')
        cmap.set_under(color='white')

        _tmp = output['dist'].reshape(output['dist'].size)
        cb = plt.pcolormesh(ff, output['bin_mids'], output['dist'].T, cmap=cmap, shading="auto",
                            rasterized=True, antialiased=True, vmin=min(_tmp[nonzero(_tmp)]), norm="log")

        plt.yscale("log")
        plt.xscale("log")

        plt.colorbar(cb)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")

        plt.xlim(ff[1], ff[-1])

        plt.show();

    if count > 0:
        print(f" -> {count}/{len(psd_array[axis])} errors found for density computation!!!")

    return output


# ### Load PSD Files

def __read_files(seed, tbeg, tend):

    net, sta, loc, cha = seed.split('.')

    psds_medians_out, times_out = [], []

    dat, dates = [], []
    for jj, day in enumerate(date_range(tbeg, tend)):

        # if jj > 2:
        #     continue

        day = str(day).split(" ")[0].replace("-", "")

        filename = f"{sta}/{day[:4]}_{sta}_{cha}_3600_{day}_hourly.pkl"

        ## skip if file does not exist
        if not os.path.isfile(config['path_to_data']+filename):
            print(f" -> skipping {filename} ...")
            continue

        try:
            out = read_pickle(config['path_to_data']+filename)
            ff1, dat1 = out['frequencies'], out['psd']

        except Exception as e:
            print(e)
            print(f" -> {day}: no data found")
            continue

        for _k, _psd in enumerate(dat1):
            print(_k, jj)
            if jj == 0 and _k == 0:
                NN = len(_psd)

            if len(_psd) == NN:
                dat.append(_psd)
                dates.append(f"{day}_{str(_k).rjust(2, '0')}")
            else:
                print(day, len(_psd), NN)
                break

    dat = array(dat)

    return dat, ff1



def main(config):


    romy_z, ff_z = __read_files(config['seeds'][0], config['d1'], config['d2'])
    romy_n, ff_n = __read_files(config['seeds'][1], config['d1'], config['d2'])
    romy_e, ff_e = __read_files(config['seeds'][2], config['d1'], config['d2'])

    romy_z, _ = __replace_noisy_psds_with_nan(romy_z, ff_z, threshold_mean=1e-19, threshold_min=1e-23,
                                              threshold_max=1e-15, flim=[0.5, 0.9],)
    romy_n, _ = __replace_noisy_psds_with_nan(romy_n, ff_n, threshold_mean=1e-19, threshold_min=1e-22,
                                              threshold_max=1e-15, flim=[0.5, 0.9],)
    romy_e, _ = __replace_noisy_psds_with_nan(romy_e, ff_e, threshold_mean=1e-19, threshold_min=1e-22,
                                              threshold_max=1e-15, flim=[0.5, 0.9],)

#     out_romy_z = __get_hist_loglog(romy_z, ff_z, bins=100, density=False, axis=1, plot=False)
#     out_romy_n = __get_hist_loglog(romy_n, ff_n, bins=100, density=False, axis=1, plot=False)
#     out_romy_e = __get_hist_loglog(romy_e, ff_e, bins=100, density=False, axis=1, plot=False)



    out_df = DataFrame()

    out_df['frequencies'] = ff_z
    out_df['psds_median_z'] = __get_median_psd(romy_z)
    out_df['perc_low_z'], out_df['perc_high_z'] = __get_percentiles(romy_z, p_low=2.5, p_high=97.5)

    out_df['psds_median_n'] = __get_median_psd(romy_n)
    out_df['perc_low_n'], out_df['perc_high_n'] = __get_percentiles(romy_n, p_low=2.5, p_high=97.5)

    out_df['psds_median_e'] = __get_median_psd(romy_e)
    out_df['perc_low_e'], out_df['perc_high_e'] = __get_percentiles(romy_e, p_low=2.5, p_high=97.5)

    chs = [c[-1] for c in config['seeds']]
    outname = f"PSD_statistics_{chs[0]}{chs[1]}{chs[2]}_{config['d1']}_{config['d2']}.pkl"
    print(f" -> store: {outname}")
    out_df.to_pickle(config['path_to_outdata']+outname)




if __name__ == "__main__":
    main(config)

## End of File
