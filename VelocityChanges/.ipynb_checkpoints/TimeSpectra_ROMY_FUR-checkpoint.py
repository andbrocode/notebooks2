#!/usr/bin/env python
# coding: utf-8

# # Analyse Velocity Changes for ROMY & FUR
# 

# In[ ]:





# ## Load Libraries

# In[6]:


import gc
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from pandas import date_range

import warnings
warnings.filterwarnings('ignore')


# In[7]:


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
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


# In[8]:


# from functions.get_mean_psd import __get_mean_psd
# from functions.get_minimum_psd import __get_minimal_psd
# from functions.get_minimal_psd import __get_minimal_psd
from functions.get_median_psd import __get_median_psd
from functions.get_percentiles import __get_percentiles
from functions.replace_noise_psd_with_nan import __replace_noisy_psds_with_nan
from functions.get_percentiles import __get_percentiles
from functions.cut_frequencies_array import __cut_frequencies_array
from functions.get_fband_averages import __get_fband_averages


# In[9]:


def __load_data_files(path, name, d1, d2):

    from numpy import array, ones, nan
    from pandas import read_pickle, date_range

    sta, cha = name.split("_")

    psds_all = []
    for _i, day in enumerate(date_range(config['d1'], config['d2'])):

        day = str(day).split(" ")[0].replace("-", "")

        year = day[:4]

        # filename = f"{name}_3600_{day}_hourly.pkl"
        filename = f"{sta}/{year}_{sta}_{cha}_3600_{day}_hourly.pkl"

        if not os.path.isfile(path+filename):
            print(f" -> no such file: {filename}")
            continue

        out = read_pickle(path+filename)
        ff = out['frequencies']

        psds_hourly = out['psd']
        for psd in psds_hourly:
            # if psd.size == 36002:
            psds_all.append(psd)
            # else:
            #     psds_all.append(ones(36002)*nan)
            #     print(psd.size)

    # psds_all_array = sum([_s for _s in psds_all], [])
    psds_all_array = array(psds_all)

    return ff, psds_all_array


# ## Configurations

# In[10]:


config = {}

config['path_to_figures'] = f"{data_path}VelocityChanges/figures/"

config['rlnm_model_path'] = f"{root_path}LNM/data/MODELS/"

config['d1'], config['d2'] = "2024-01-01", "2024-02-28"

# config['path_to_data'] = data_path+f"VelocityChanges/data/PSDS/"
config['path_to_data'] = data_path+f"LNM2/PSDS/"


# ## Load as Arrays

# ## ROMY

# In[11]:


names = ["ROMY_BJZ", "ROMY_BJN", "ROMY_BJE"]


# In[12]:


## Data1 --------------------------
name = names[0]

ff_1, psd_1 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_1 = np.arange(0, psd_1.shape[0], 1)

## cut to specified frequency range
psd_1, ff_1 = __cut_frequencies_array(psd_1, ff_1, 1e-3, 5e0)

## median for octave bands
# ff_1, psd_1 = __get_fband_averages(ff_1, psd_1)

## filter corrupt psds
psd_1, rejected_1 = __replace_noisy_psds_with_nan(psd_1, ff=ff_1,
                                                  threshold_mean=5e-19,
                                                  threshold_min=1e-23,
                                                  threshold_max=1e-16,
                                                  flim=[0.5, 0.9],
                                                  )
gc.collect()


# In[13]:


## Data2 --------------------------
name = names[1]

ff_2, psd_2 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_2 = np.arange(0, psd_2.shape[0], 1)

## cut to specified frequency range
psd_2, ff_2 = __cut_frequencies_array(psd_2, ff_2, 1e-3, 5e0)

## median for octave bands
# ff_2, psd_2 = __get_fband_averages(ff_2, psd_2)

## filter corrupt psds
psd_2, rejected_2 = __replace_noisy_psds_with_nan(psd_2, ff=ff_2,
                                                  threshold_mean=5e-19,
                                                  threshold_min=1e-23,
                                                  threshold_max=1e-16,
                                                  flim=[0.5, 0.9],
                                                  )
gc.collect()


# In[14]:


## Data3 --------------------------
name = names[2]

ff_3, psd_3 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_3 = np.arange(0, psd_3.shape[0], 1)

## cut to specified frequency range
psd_3, ff_3 = __cut_frequencies_array(psd_3, ff_3, 1e-3, 5e0)

## median for octave bands
# ff_3, psd_3 = __get_fband_averages(ff_3, psd_3)

## filter corrupt psds
psd_3, rejected_3 = __replace_noisy_psds_with_nan(psd_3, ff=ff_3,
                                                  threshold_mean=5e-19,
                                                  threshold_min=1e-23,
                                                  threshold_max=1e-16,
                                                  flim=[0.5, 0.9],
                                                  )
gc.collect();


# ## Plot PSD Comparison

# In[1]:


def __makeplot_image_overview(ff, psds, times, names):

    import gc

    from numpy import isnan, median, mean, std, array, zeros, nanmax, nanmin, shape, nanpercentile
    from scipy.stats import median_abs_deviation as mad
    from matplotlib import colors


    ## convert frequencies to periods
    # pp=[]
    # for mm in range(len(ff)):
    #     ppp = zeros(len(ff[mm]))
    #     ppp = 1/ff[mm]
    #     pp.append(ppp)
    # pp[0] = 0

    ## define colormap
    cmap = plt.colormaps.get_cmap('viridis')
    cmap.set_bad(color='lightgrey')
#     cmap.set_under(color='black')
#     cmap.set_over(color='white')

    min0 = nanpercentile(psds[0].reshape(1, psds[0].size), 5)
    max0 = nanpercentile(psds[0].reshape(1, psds[0].size), 95)


    ##____________________________

#     NN = 3
    N = int(24*365)

    font = 14

    fig = plt.figure(constrained_layout=False, figsize=(15, 10))
    widths = [6, 1]
    heights = [1, 1, 1]

    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

    plt.subplots_adjust(hspace=0.15, wspace=0.02)

    ax1_1 = fig.add_subplot(spec[0, 0])
    ax1_2 = fig.add_subplot(spec[0, 1], sharey=ax1_1)
    ax2_1 = fig.add_subplot(spec[1, 0], sharex=ax1_1)
    ax2_2 = fig.add_subplot(spec[1, 1])
    ax3_1 = fig.add_subplot(spec[2, 0], sharex=ax1_1)
    ax3_2 = fig.add_subplot(spec[2, 1])

    im1 = ax1_1.pcolormesh( times[0]/24, ff[0], psds[0].T,
                            cmap=cmap,
                            norm=colors.LogNorm(5e-23, 5e-18),
                            rasterized=True,
                            )
    im2 = ax2_1.pcolormesh( times[1]/24, ff[1], psds[1].T,
                            cmap=cmap,
                            norm=colors.LogNorm(5e-23, 5e-18),
                            rasterized=True,
                            )
    im3 = ax3_1.pcolormesh( times[2]/24, ff[2], psds[2].T,
                            cmap=cmap,
                            norm=colors.LogNorm(5e-23, 5e-18),
                            rasterized=True,
                            )

    set_color = "seagreen"

    perc_lower, perc_upper = __get_percentiles(psds[0], p_low=2.5, p_high=97.5)
    ax1_2.fill_betweenx(ff[0], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax1_2.plot(__get_median_psd(psds[0]), ff[0], color=set_color, zorder=3, alpha=0.9, label="Median")

    perc_lower, perc_upper = __get_percentiles(psds[1], p_low=2.5, p_high=97.5)
    ax2_2.fill_betweenx(ff[1], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax2_2.plot(__get_median_psd(psds[1]), ff[1], color=set_color, zorder=3, alpha=0.9, label="Median")

    perc_lower, perc_upper = __get_percentiles(psds[2], p_low=2.5, p_high=97.5)
    ax3_2.fill_betweenx(ff[2], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax3_2.plot(__get_median_psd(psds[2]), ff[2], color=set_color, zorder=3, alpha=0.9, label="Median")


    ax1_2.set_xlim(5e-23, 5e-18)
    ax2_2.set_xlim(5e-23, 5e-18)
    ax3_2.set_xlim(5e-23, 5e-18)

    # ax3_2.set_xscale("log")

    plt.setp(ax1_1.get_xticklabels(), visible=False)
    plt.setp(ax2_1.get_xticklabels(), visible=False)

#     plt.setp(ax1_2.get_xticklabels(), visible=False)
#     plt.setp(ax2_2.get_xticklabels(), visible=False)

    plt.setp(ax1_2.get_yticklabels(), visible=False)
    plt.setp(ax2_2.get_yticklabels(), visible=False)
    plt.setp(ax3_2.get_yticklabels(), visible=False)




    for ax in [ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2]:
        ax.tick_params(labelsize=font-2)
        ax.set_ylim(1e-2, 3e0)
        ax.set_yscale("log")

    ax1_2.set_xscale("logit")
    ax2_2.set_xscale("logit")
    ax3_2.set_xscale("logit")

    ax3_1.set_xlabel(f"Time (days) from {config['d1']}", fontsize=font, labelpad=1)
    ax3_2.set_xlabel("PSD (rad$^2$/s$^2$/Hz)", fontsize=font, labelpad=1)
    # ax3_2.set_xlabel(r"", fontsize=font, labelpad=-1)

    # new_ticks = [int(round(t/24, 0)) for t in ax3_1.get_xticks()]
    # ax3_1.set_xticklabels(new_ticks)

#     ## panel labels
    ax1_1.text(-.08, .99, '(a)', ha='left', va='top', transform=ax1_1.transAxes, fontsize=font+2)
    ax2_1.text(-.08, .99, '(b)', ha='left', va='top', transform=ax2_1.transAxes, fontsize=font+2)
    ax3_1.text(-.08, .99, '(c)', ha='left', va='top', transform=ax3_1.transAxes, fontsize=font+2)

#     ## data labels
    ax1_1.text(.99, .97, f'{names[0]}', ha='right', va='top', transform=ax1_1.transAxes, fontsize=font)
    ax2_1.text(.99, .97, f'{names[1]}', ha='right', va='top', transform=ax2_1.transAxes, fontsize=font)
    ax3_1.text(.99, .97, f'{names[2]}', ha='right', va='top', transform=ax3_1.transAxes, fontsize=font)

    ax1_1.set_ylabel(r"Frequency (Hz)", fontsize=font)
    ax2_1.set_ylabel(r"Frequency (Hz)", fontsize=font)
    ax3_1.set_ylabel(r"Frequency (Hz)", fontsize=font)

#     ## set colorbar at bottom
    cbar = fig.colorbar(im1, orientation='vertical', ax=ax1_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (rad$^2$/s$^2$/Hz)", fontsize=font-2, labelpad=1)

    cbar = fig.colorbar(im2, orientation='vertical', ax=ax2_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (rad$^2$/s$^2$/Hz)", fontsize=font-2, labelpad=1)

    cbar = fig.colorbar(im3, orientation='vertical', ax=ax3_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (rad$^2$/s$^2$/Hz)", fontsize=font-2, labelpad=1)

    gc.collect()

    # plt.show();
    return fig


# In[ ]:


labels = [f"{n.split('_')[1]}" for n in names]

fig = __makeplot_image_overview(
                                [ff_1, ff_2, ff_3],
                                [psd_1, psd_2, psd_3],
                                [tt_1, tt_2, tt_3],
                                labels,
                                )
print(f" -> save: {config['path_to_figures']}TimeSpectra_ROMY_PSD_{config['d1']}_{config['d2']}.png")
fig.savefig(config['path_to_figures']+f"TimeSpectra_ROMY_PSD_{config['d1']}_{config['d2']}.png", format="png", dpi=150, bbox_inches='tight')


# ## FUR

# In[ ]:


names = ["FUR_BHZ", "FUR_BHN", "FUR_BHE"]


# In[ ]:


## Data1 --------------------------
name = names[0]

ff_1, psd_1 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_1 = np.arange(0, psd_1.shape[0], 1)

ff_1, psd_1 = __get_fband_averages(ff_1, psd_1)

psd_1, rejected_1 = __replace_noisy_psds_with_nan(psd_1, f=ff_1,
                                                  threshold_mean=1e-13,
                                                  threshold_min=None,
                                                  threshold_max=None,
                                                  flim=[0.002, 0.05],
                                                  )
gc.collect();

# In[ ]:


## Data2 --------------------------
name = names[1]

ff_2, psd_2 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_2 = np.arange(0, psd_2.shape[0], 1)

ff_2, psd_2 = __get_fband_averages(ff_2, psd_2)

psd_2, rejected_2 = __replace_noisy_psds_with_nan(psd_2, f=ff_2,
                                                  threshold_mean=1e-13,
                                                  threshold_min=None,
                                                  threshold_max=None,
                                                  flim=[0.002, 0.05],
                                                  )
gc.collect();

# In[ ]:


## Data3 --------------------------
name = names[2]

ff_3, psd_3 = __load_data_files(config['path_to_data'], name, config['d1'], config['d2'])
tt_3 = np.arange(0, psd_3.shape[0], 1)

ff_3, psd_3 = __get_fband_averages(ff_3, psd_3)

psd_3, rejected_3 = __replace_noisy_psds_with_nan(psd_3, ff=ff_3,
                                                  threshold_mean=1e-13,
                                                  threshold_min=None,
                                                  threshold_max=None,
                                                  flim=[0.002, 0.05],
                                                  )
gc.collect();

# In[ ]:


def __makeplot_image_overview(ff, psds, times, names):

    import gc
    from numpy import isnan, median, mean, std, array, zeros, nanmax, nanmin, shape, nanpercentile
    from scipy.stats import median_abs_deviation as mad
    from matplotlib import colors


    ## convert frequencies to periods
    # pp=[]
    # for mm in range(len(ff)):
    #     ppp = zeros(len(ff[mm]))
    #     ppp = 1/ff[mm]
    #     pp.append(ppp)
    # pp[0] = 0

    ## define colormap
    cmap = plt.colormaps.get_cmap('viridis')
    cmap.set_bad(color='lightgrey')
#     cmap.set_under(color='black')
#     cmap.set_over(color='white')

    min0 = nanpercentile(psds[0].reshape(1, psds[0].size), 5)
    max0 = nanpercentile(psds[0].reshape(1, psds[0].size), 95)


    ##____________________________

#     NN = 3
    N = int(24*365)

    font = 14

    fig = plt.figure(constrained_layout=False, figsize=(15, 10))
    widths = [6, 1]
    heights = [1, 1, 1]

    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

    plt.subplots_adjust(hspace=0.15, wspace=0.02)

    ax1_1 = fig.add_subplot(spec[0, 0])
    ax1_2 = fig.add_subplot(spec[0, 1], sharey=ax1_1)
    ax2_1 = fig.add_subplot(spec[1, 0], sharex=ax1_1)
    ax2_2 = fig.add_subplot(spec[1, 1])
    ax3_1 = fig.add_subplot(spec[2, 0], sharex=ax1_1)
    ax3_2 = fig.add_subplot(spec[2, 1])

    im1 = ax1_1.pcolormesh( times[0]/24, ff[0], psds[0].T,
                            cmap=cmap,
                            norm=colors.LogNorm(2e-20, 2e-10),
                            rasterized=True,
                            )
    im2 = ax2_1.pcolormesh( times[1]/24, ff[1], psds[1].T,
                            cmap=cmap,
                            norm=colors.LogNorm(2e-20, 2e-10),
                            rasterized=True,
                            )
    im3 = ax3_1.pcolormesh( times[2]/24, ff[2], psds[2].T,
                            cmap=cmap,
                            norm=colors.LogNorm(2e-20, 2e-10),
                            rasterized=True,
                            )

    set_color = "seagreen"

    perc_lower, perc_upper = __get_percentiles(psds[0], p_low=2.5, p_high=97.5)
    ax1_2.fill_betweenx(ff[0], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax1_2.plot(__get_median_psd(psds[0]), ff[0], color=set_color, zorder=3, alpha=0.9, label="Median")

    perc_lower, perc_upper = __get_percentiles(psds[1], p_low=2.5, p_high=97.5)
    ax2_2.fill_betweenx(ff[1], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax2_2.plot(__get_median_psd(psds[1]), ff[1], color=set_color, zorder=3, alpha=0.9, label="Median")

    perc_lower, perc_upper = __get_percentiles(psds[2], p_low=2.5, p_high=97.5)
    ax3_2.fill_betweenx(ff[2], perc_lower, perc_upper, color=set_color, zorder=3, alpha=0.4, label="")
    ax3_2.plot(__get_median_psd(psds[2]), ff[2], color=set_color, zorder=3, alpha=0.9, label="Median")


    ax1_2.set_xlim(2e-20, 2e-10)
    ax2_2.set_xlim(2e-20, 2e-10)
    ax3_2.set_xlim(2e-20, 2e-10)

    # ax3_2.set_xscale("log")

    plt.setp(ax1_1.get_xticklabels(), visible=False)
    plt.setp(ax2_1.get_xticklabels(), visible=False)

#     plt.setp(ax1_2.get_xticklabels(), visible=False)
#     plt.setp(ax2_2.get_xticklabels(), visible=False)

    plt.setp(ax1_2.get_yticklabels(), visible=False)
    plt.setp(ax2_2.get_yticklabels(), visible=False)
    plt.setp(ax3_2.get_yticklabels(), visible=False)




    for ax in [ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2]:
        ax.tick_params(labelsize=font-2)
        ax.set_ylim(1e-2, 5e0)
        ax.set_yscale("log")

    ax1_2.set_xscale("logit")
    ax2_2.set_xscale("logit")
    ax3_2.set_xscale("logit")

    ax3_1.set_xlabel("Time (days)", fontsize=font, labelpad=1)
    ax3_2.set_xlabel("PSD (m$^2$/s$^4$/Hz)", fontsize=font, labelpad=1)
    # ax3_2.set_xlabel(r"", fontsize=font, labelpad=-1)

    # new_ticks = [int(round(t/24, 0)) for t in ax3_1.get_xticks()]
    # ax3_1.set_xticklabels(new_ticks)

#     ## panel labels
    ax1_1.text(-.08, .99, '(a)', ha='left', va='top', transform=ax1_1.transAxes, fontsize=font+2)
    ax2_1.text(-.08, .99, '(b)', ha='left', va='top', transform=ax2_1.transAxes, fontsize=font+2)
    ax3_1.text(-.08, .99, '(c)', ha='left', va='top', transform=ax3_1.transAxes, fontsize=font+2)

#     ## data labels
    ax1_1.text(.99, .97, f'{names[0]}', ha='right', va='top', transform=ax1_1.transAxes, fontsize=font)
    ax2_1.text(.99, .97, f'{names[1]}', ha='right', va='top', transform=ax2_1.transAxes, fontsize=font)
    ax3_1.text(.99, .97, f'{names[2]}', ha='right', va='top', transform=ax3_1.transAxes, fontsize=font)

    ax1_1.set_ylabel(r"Frequency (Hz)", fontsize=font)
    ax2_1.set_ylabel(r"Frequency (Hz)", fontsize=font)
    ax3_1.set_ylabel(r"Frequency (Hz)", fontsize=font)

#     ## set colorbar at bottom
    cbar = fig.colorbar(im1, orientation='vertical', ax=ax1_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (m$^2$/$s^4$/Hz)", fontsize=font-2, labelpad=1)

    cbar = fig.colorbar(im2, orientation='vertical', ax=ax2_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (m$^2$/$s^4$/Hz)", fontsize=font-2, labelpad=1)

    cbar = fig.colorbar(im3, orientation='vertical', ax=ax3_2, pad=0.05, extend="both")
    cbar.set_label(r"PSD (m$^2$/$s^4$/Hz)", fontsize=font-2, labelpad=1)

    gc.collect()

    # plt.show();
    return fig


# In[ ]:


labels = [f"{n.split('_')[1]}.{n.split('_')[2]}" for n in names]

fig = __makeplot_image_overview(
                                [ff_1, ff_2, ff_3],
                                [psd_1, psd_2, psd_3],
                                [tt_1, tt_2, tt_3],
                                labels,
                                )

print(f" -> save: {config['path_to_figures']}TimeSpectra_FUR_PSD_{config['d1']}_{config['d2']}.png")
fig.savefig(config['path_to_figures']+f"TimeSpectra_FUR_PSD_{config['d1']}_{config['d2']}.png", format="png", dpi=150, bbox_inches='tight')

## End of File