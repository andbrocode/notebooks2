#!/usr/bin/env python
# coding: utf-8

# # Analyse Rotation Spectra - Image Overview Hours


## __________________________________
## Load Libraries

from andbro__querrySeismoData import __querrySeismoData
from andbro__savefig import __savefig

from obspy import UTCDateTime
from numpy import log10, zeros, pi, append, linspace, mean, median, array, where, transpose, shape, histogram, arange
from numpy import logspace, linspace, log, log10, isinf, ones, nan, count_nonzero, sqrt, isnan
from pandas import DataFrame, concat, Series, date_range, read_csv, read_pickle
from tqdm import tqdm_notebook
from pathlib import Path
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import welch

import os
import pickle
import matplotlib.pyplot as plt


if os.uname().nodename == "lighthouse":
    root_path = "/home/andbro/kilauea-data/"
elif os.uname().nodename == "kilauea":
    root_path = "/import/kilauea-data/"


## Methods

def __get_minimal_psd(psds):

    from numpy import nanmin, array, nonzero

    min_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        min_psd[f] = nanmin(a[nonzero(a)])

    return min_psd


# In[5]:


def __get_median_psd(psds):

    from numpy import median, zeros, isnan

    med_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        med_psd[f] = median(a[~isnan(a)])

    return med_psd


# In[6]:


def __get_mean_psd(psds):

    from numpy import mean, zeros, isnan

    mean_psd = zeros(psds.shape[1])

    for f in range(psds.shape[1]):
        a = psds[:,f]
        mean_psd[f] = mean(a[~isnan(a)])

    return mean_psd


# In[7]:


def __get_minimum_psd(psds):

    for i, psd in enumerate(psds):
        if i == 0:
            lowest_value = psd.sum()
            idx = 0

        value = psd.sum()

        if value < lowest_value and value != 0:
            lowest_value = value
            idx = i

    return psds[idx]


# In[8]:


def __load_psds(file, config):

    ## get data to dataframe and transpose and reindex
    df = read_csv(file, index_col=False)
    df = df.transpose()
    df.reset_index(inplace=True)
    # df.dtypes
    # df.mean()

    ## set column names
    try:
        columns = pickle.load(open(f"{config['inpath']}{config['inname']}_columns.pick", 'rb'))
        df.columns = [column.replace("-","") for column in columns]
    except:
        columns = arange(0,df.shape[1]).astype(str)
        df.columns = columns
        print(" -> Failed to assign column names! Assigned numbers instead!")

    ## check for column dublicates
    if len(df.columns.unique()) != len(df.columns):
        print(f" -> removing {len(df.columns)-len(df.columns.unique())} column dublicate(s)!")
        df = df.loc[:,~df.columns.duplicated()]

    count=0
    dates_expected = date_range(config['date1'].date, config['date2'].date, periods=int((config['date2']-config['date1'])/86400)+1)
    for dex in dates_expected:
        dex=str(dex.isoformat()[:10]).replace("-","")
        if not dex in df.columns:
            count+=1
    print(f" -> missing {count} days")

    print(f" -> total of {df.shape[0]} psds")

#     ## convert to list
#     psds = []
#     for col in array(df.columns):

#         ## turn non-float series to float objects
#         df[col] = pd.to_numeric(df[col], errors = 'coerce')

#         ## add to psds list
#         psds.append(array(df[col]))

    return df


# In[9]:


def __get_array_from_dataframe(df):

    from pandas import to_numeric

    ## convert to list
    psds = []
    for col in array(df.columns):

        ## turn non-float series to float objects
        df[col] = to_numeric(df[col], errors = 'coerce')

        ## add to psds list
        psds.append(array(df[col]))

    return array(psds)


# In[10]:


# def __remove_noisy_psds(df, threshold_mean=1e-13):

#     from numpy import delete

#     l1 = len(df.columns)
#     for col in df.columns:
# #         print(col, type(col))
#         if df[col].astype(float).mean() > threshold_mean:
#             df = df.drop(columns=col)
#     l2 = len(df.columns)
#     print(f" -> removed {l1-l2} columns due to mean thresholds!")
#     print(f" -> {l2} psds remain")

#     return df


# In[11]:


def __get_percentiles(arr):

    from numpy import zeros, nanpercentile

    percentiles_lower = zeros(shape(arr)[1])
    percentiles_upper = zeros(shape(arr)[1])

    for kk in range(shape(arr)[1]):
        out = nanpercentile(arr[:, kk],  [2.5 ,97.5])
        percentiles_upper[kk] = out[1]
        percentiles_lower[kk] = out[0]

    return percentiles_lower, percentiles_upper


## _________________________________________________________
## Configurations

name = "BSPF"
inname = "2022_BSPF_Z_3600"
subdir = "BSPF_2022_Z/"
threshold = 8e-14
period_limits = 1/10, 100

#name = "PFO"
#inname = "2022_PFO_Z_3600"
#subdir = "PFO_2022_Z/"
#threshold = 8e-14
#period_limits = 1/10, 100


## _________________________________________________________


path = f"{root_path}BSPF/data/"

config = pickle.load(open(path+subdir+inname+"_config.pkl", 'rb'))

config['inname'] = inname
config['inpath'] = path+subdir
config['period_limits'] = period_limits
config['thres'] = threshold

config['outpath_figures'] = f"{root_path}BSPF/figures/"

config['outname_figures'] = f"{name}_psdimage2"

config['rlnm_model_path'] = f"{root_path}LNM/data/MODELS/"

config['frequency_limits'] = [1/config['period_limits'][1], 1/config['period_limits'][0]]

## load frequencies
# ff_Z = pickle.load(open(f"{config['inpath'].replace('Z','Z')}{config['inname'].replace('Z','Z')}_frequency_axis.pkl", 'rb'))
# ff_N = pickle.load(open(f"{config['inpath'].replace('Z','N')}{config['inname'].replace('Z','N')}_frequency_axis.pkl", 'rb'))
# ff_E = pickle.load(open(f"{config['inpath'].replace('Z','E')}{config['inname'].replace('Z','E')}_frequency_axis.pkl", 'rb'))


# ## Load as Arrays
def __load_data_files(config, path):

    from tqdm.notebook import tqdm
    from numpy import array

    print(path)

    config['files'] = [file for file in os.listdir(path) if "hourly" in file]
    config['files'].sort()

    psds_all, times_nom, times = [], arange(0, 24*365, 1), []
    count, missing  = 0, 0

    for file in tqdm(config['files']):
        date = file.split("_")[-2]
        psds_hourly = read_pickle(path+file)

        for h in range(24):
            try:
                psds_all.append(psds_hourly[h])
                times.append(times_nom[count])
            except:
                missing += 1
            count += 1

    print(f" -> missing: {missing}")
    return array(psds_all), times



# ### Cut Frequency Range
def __cut_frequencies_array(arr, freqs, fmin, fmax):

    ind = []
    for i, f in enumerate(freqs):
        if f >= fmin and f <= fmax:
            ind.append(i)

    ff = freqs[ind[0]:ind[-1]]
    pp = arr[:,ind[0]:ind[-1]]

    return pp, ff



# ## Remove Noisy PSDs
def __remove_noisy_psds(arr, times, threshold_mean=1e-16):

    from numpy import delete, shape, sort, array, nan

    l1 = shape(arr)[0]

    idx_to_remove = []
    for ii in range(shape(arr)[0]):

        ## appy upper threshold
        if arr[ii,:].mean() > threshold_mean:
            idx_to_remove.append(ii)

        ## apply default lowe threshold
        if arr[ii,:].mean() < 1e-26:
            idx_to_remove.append(ii)

    for jj in sort(array(idx_to_remove))[::-1]:

        ## option 1: delte rows
#         arr = delete(arr, jj, axis=0)
#         times = delete(times, jj, axis=0)

        ## option 2: replace with nan values
        arr[jj,:] = ones(len(arr[jj]))*nan

    l2 = shape(arr)[0]

    print(f" -> removed {l1-l2} rows due to mean thresholds!")
    print(f" -> {l2} psds remain")

    return arr, times



def __check():
    if len(times_N) != shape(ADR_N)[0]:
        print("-> times N:", len(times_N), shape(ADR_N)[0])
    if len(ff_N) != shape(ADR_N)[1]:
        print("-> freqs N:", len(ff_N), shape(ADR_N)[0])

    if len(times_E) != shape(ADR_E)[0]:
        print("-> times E:", len(times_E), shape(ADR_E)[0])
    if len(ff_E) != shape(ADR_E)[1]:
        print("-> freqs E:", len(ff_E), shape(ADR_E)[0])

    if len(times_Z) != shape(ADR_Z)[0]:
        print("-> time Z:", len(times_Z), shape(ADR_Z)[0])
    if len(ff_Z) != shape(ADR_Z)[1]:
        print("-> freqs Z:", len(ff_Z), shape(ADR_Z)[0])



# ## Plotting
def __makeplot_image_overview2(ff, psds, times, dates=None):

    from tqdm import tqdm
    from numpy import isnan, median, mean, std, array, zeros, nanmax, nanmin
    from scipy.stats import median_abs_deviation as mad
    from matplotlib import colors


    ## theoretical rlnm
    rlnm = read_csv(config['rlnm_model_path']+"rlnm_theory.csv")

    ## convert frequencies to periods
    pp=[]
    for mm in range(len(ff)):
        ppp = zeros(len(ff[mm]))
        ppp = 1/ff[mm]
        pp.append(ppp)

    ## define colormap
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad(color='white')
#     cmap.set_under(color='black')
#     cmap.set_over(color='white')

    ## compute overall maxima and minima
#    max_psds = nanmax(array([nanmax(psds[0]), nanmax(psds[1]), nanmax(psds[2])]))
#    min_psds = nanmin(array([nanmin(psds[0]), nanmin(psds[1]), nanmin(psds[2])]))
#    print(max_psds, min_psds)
    min_psds_arr, max_psds_arr = __get_percentiles((psds[0]))
    min_psds, max_psds = min(min_psds_arr), max(max_psds_arr)

    ##____________________________

    N = int(24*365)

    font = 14

    fig = plt.figure(constrained_layout=False, figsize=(15,10))
    widths = [6, 1]
    heights = [1, 1, 1]
    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

    plt.subplots_adjust(hspace=0.1, wspace=0.02)

    ax1_1 = fig.add_subplot(spec[0, 0])
    ax1_2 = fig.add_subplot(spec[0, 1], sharey=ax1_1)
    ax2_1 = fig.add_subplot(spec[1, 0], sharex=ax1_1)
    ax2_2 = fig.add_subplot(spec[1, 1], sharey=ax2_1)
    ax3_1 = fig.add_subplot(spec[2, 0], sharex=ax1_1)
    ax3_2 = fig.add_subplot(spec[2, 1], sharey=ax3_1)

    im1 = ax1_1.pcolormesh( times[0], 1/ff[0], psds[0].T,
                            cmap=cmap,
                            vmax=max_psds,
                            vmin=min_psds,
                            norm=colors.LogNorm(),
                            )
    im2 = ax2_1.pcolormesh( times[1], 1/ff[1], psds[1].T,
                            cmap=cmap,
                            vmax=max_psds,
                            vmin=min_psds,
                            norm=colors.LogNorm(),
                            )
    im3 = ax3_1.pcolormesh( times[2], 1/ff[2], psds[2].T,
                            cmap=cmap,
                            vmax=max_psds,
                            vmin=min_psds,
                            norm=colors.LogNorm(),
                            )

    set_color = "darkblue"

    perc_lower, perc_upper = __get_percentiles((psds[0]))
    ax1_2.fill_betweenx(pp[0], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax1_2.plot(__get_median_psd(psds[0]), pp[0], color=set_color, zorder=3, alpha=0.9, label="Median")
#     ax1_2.plot(rlnm['rlnm_psd_median'], rlnm['period'], color="black", zorder=2, ls="--", lw=2)

    perc_lower, perc_upper = __get_percentiles((psds[1]))
    ax2_2.fill_betweenx(pp[1], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax2_2.plot(__get_median_psd(psds[1]), pp[1], color=set_color, zorder=3, alpha=0.9, label="Median")
#     ax2_2.plot(rlnm['rlnm_psd_median'], rlnm['period'], color="black", zorder=2, ls="--", lw=2)

    perc_lower, perc_upper = __get_percentiles((psds[2]))
    ax3_2.fill_betweenx(pp[2], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax3_2.plot(__get_median_psd(psds[2]), pp[2], color=set_color, zorder=3, alpha=0.9, label="Median")
#     ax3_2.plot(rlnm['rlnm_psd_median'], rlnm['period'], color="black", zorder=2, ls="--", lw=2)


    plt.setp(ax1_1.get_xticklabels(), visible=False)
    plt.setp(ax2_1.get_xticklabels(), visible=False)

    plt.setp(ax1_2.get_xticklabels(), visible=False)
    plt.setp(ax2_2.get_xticklabels(), visible=False)

    plt.setp(ax1_2.get_yticklabels(), visible=False)
    plt.setp(ax2_2.get_yticklabels(), visible=False)
    plt.setp(ax3_2.get_yticklabels(), visible=False)

    for ax in [ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2]:
        ax.tick_params(labelsize=font-2)
        ax.set_ylim(config['period_limits'][0], config['period_limits'][1])
        ax.set_yscale("log")

    for ax in [ax1_2, ax2_2, ax3_2]:
        ax.set_xscale("logit")

    ax3_1.set_xlabel("Days of 2019", fontsize=font, labelpad=1)
    ax3_2.set_xlabel(r"PSD (rad$^2$/s$^2$/$Hz$)", fontsize=font, labelpad=-1)

    new_ticks = [int(round(t/24,0)) for t in ax3_1.get_xticks()]
    ax3_1.set_xticklabels(new_ticks)

#     ## panel labels
    ax1_1.text(.01, .99, 'a)', ha='left', va='top', color='white', transform=ax1_1.transAxes, fontsize=font+2)
    ax2_1.text(.01, .99, 'b)', ha='left', va='top', color='white', transform=ax2_1.transAxes, fontsize=font+2)
    ax3_1.text(.01, .99, 'c)', ha='left', va='top', color='white', transform=ax3_1.transAxes, fontsize=font+2)

#     ## data labels
    array = name.split("_")[0]
    ax1_1.text(.99, .99, f'{array} $vertical$', ha='right', va='top', color='white', transform=ax1_1.transAxes, fontsize=font)
    ax2_1.text(.99, .99, f'{array} $north$', ha='right', va='top', color='white', transform=ax2_1.transAxes, fontsize=font)
    ax3_1.text(.99, .99, f'{array} $east$', ha='right', va='top', color='white', transform=ax3_1.transAxes, fontsize=font)

    ax1_1.set_ylabel(r"Periods (s)", fontsize=font)
    ax2_1.set_ylabel(r"Periods (s)", fontsize=font)
    ax3_1.set_ylabel(r"Periods (s)", fontsize=font)

#     ## set colorbar at bottom
    cbar = fig.colorbar(im1, orientation='vertical', ax=[ax1_2, ax2_2, ax3_2], aspect=35, pad=0.05)
    cbar.set_label(r"PSD (rad$^2$/s$^2$/$Hz$)", fontsize=font-2, labelpad=-50)

    __savefig(fig, outpath=config['outpath_figures'], outname=config['outname_figures'], mode="png", dpi=300)

#    plt.show();
#    return fig




def main(config):


	ADR_Z, times_Z = __load_data_files(config, config['inpath'].replace("Z","Z"))

	ff_Z = pickle.load(open(f"{config['inpath'].replace('Z','Z')}{config['inname'].replace('Z','Z')}_frequency_axis.pkl", 'rb'))
	times_Z = pickle.load(open(f"{config['inpath'].replace('Z','Z')}{config['inname'].replace('Z','Z')}_times_axis.pkl", 'rb'))

	ADR_N, times_N = __load_data_files(config, config['inpath'].replace("Z","N"))

	ff_N = pickle.load(open(f"{config['inpath'].replace('Z','N')}{config['inname'].replace('Z','N')}_frequency_axis.pkl", 'rb'))
	times_N = pickle.load(open(f"{config['inpath'].replace('Z','N')}{config['inname'].replace('Z','N')}_times_axis.pkl", 'rb'))


	ADR_E, times_E = __load_data_files(config, config['inpath'].replace("Z","E"))

	ff_E = pickle.load(open(f"{config['inpath'].replace('Z','E')}{config['inname'].replace('Z','E')}_frequency_axis.pkl", 'rb'))
	times_E = pickle.load(open(f"{config['inpath'].replace('Z','E')}{config['inname'].replace('Z','E')}_times_axis.pkl", 'rb'))



	ADR_N, ff_N = __cut_frequencies_array(ADR_N, ff_N, config['frequency_limits'][0], config['frequency_limits'][1])
	ADR_E, ff_E = __cut_frequencies_array(ADR_E, ff_E, config['frequency_limits'][0], config['frequency_limits'][1])
	ADR_Z, ff_Z = __cut_frequencies_array(ADR_Z, ff_Z, config['frequency_limits'][0], config['frequency_limits'][1])


	ADR_N, times_N = __remove_noisy_psds(ADR_N, times_N, threshold_mean=config['thres'])
	ADR_E, times_E = __remove_noisy_psds(ADR_E, times_E, threshold_mean=config['thres'])
	ADR_Z, times_Z = __remove_noisy_psds(ADR_Z, times_Z, threshold_mean=config['thres'])

#	__check()

	__makeplot_image_overview2(
		                  [ff_Z, ff_N, ff_E],
		                  [ADR_Z, ADR_N, ADR_E],
		                  [times_Z, times_N, times_E],
		                  )


if __name__ == "__main__":

	main(config)

## END OF FILE
