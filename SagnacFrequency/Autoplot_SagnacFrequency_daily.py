#!/bin/python3
#
#
#
#
#
## _______________________________________

from pandas import read_pickle
from obspy import UTCDateTime

import os
import matplotlib.pyplot as plt


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
    
config = {}

config['ring'] = "Z"

config['path_to_data'] = data_path+"sagnac_frequency/autodata/"
config['path_to_figs'] = data_path+"sagnac_frequency/autodata/"

## _______________________________________

def __makeplot_overview(df):

    from matplotlib.gridspec import GridSpec
    from numpy import nanmean, nanmedian
    
    def __smooth(y, box_pts):
        from numpy import ones, convolve, hanning

#         win = ones(box_pts)/box_pts
        win = hanning(box_pts)
        y_smooth = convolve(y, win/sum(win), mode='same')

        return y_smooth
    
    
    time_scaling, time_unit = 1, "MJD"
    
    ## ___________________
    ##
    
    NN = 9
    font = 14
    smooting = 5
    cut_off = int(smooting/2)  
    
    
#     fig, ax = plt.subplots(NN, 1, figsize=(15,10), sharex=True)
#     plt.subplots_adjust(hspace=0.05)
    
    fig = plt.figure(figsize=(14,12))
    
    gs = GridSpec(NN, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0:2, :])
    
    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, :])
    
    ax5 = fig.add_subplot(gs[5:7, :])
    
    ax6 = fig.add_subplot(gs[7, :])
    ax7 = fig.add_subplot(gs[8, :])
        
    plt.subplots_adjust(hspace=0.1)
    
    ## Panel 1 -------------------------
    ax1.scatter(df['times_mjd']/time_scaling, df['fz'], c="tab:blue", s=7, alpha=0.4, zorder=2)
    ax1.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['fz'],smooting)[cut_off:-cut_off], "tab:blue", lw=0.5, label="FJZ", zorder=2)
    
    ax1.scatter(df['times_mjd']/time_scaling, df['f1'], c="tab:orange", s=7, alpha=0.4, zorder=2)
    ax1.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['f1'],smooting)[cut_off:-cut_off], "tab:orange", lw=0.5, label="CCW", zorder=2)
    
    ax1.scatter(df['times_mjd']/time_scaling, df['f2'], c="tab:red", s=7, alpha=0.4, zorder=2)
    ax1.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['f2'],smooting)[cut_off:-cut_off], "tab:red", lw=0.5, label="CW", zorder=2)


    ## Panel 2-4 -------------------------
    ax2.scatter(df['times_mjd']/time_scaling, df['fz'], c="tab:blue", s=7, alpha=0.4, zorder=2)
#     ax2.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['fz'],smooting)[cut_off:-cut_off], "tab:blue", lw=0.5, label="FJZ", zorder=2)
    
    ax3.scatter(df['times_mjd']/time_scaling, df['f1'], c="tab:orange", s=7, alpha=0.4, zorder=2)
#     ax3.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['f1'],smooting)[cut_off:-cut_off], "tab:orange", lw=0.5, label="CCW", zorder=2)
    
    ax4.scatter(df['times_mjd']/time_scaling, df['f2'], c="tab:red", s=7, alpha=0.4, zorder=2)
#     ax4.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['f2'],smooting)[cut_off:-cut_off], "tab:red", lw=0.5, label="CW", zorder=2)
   
        
    ## Panel 5 -------------------------
    ax5.scatter(df['times_mjd']/time_scaling, df['pz'], c="tab:blue", s=7, alpha=0.4, zorder=2)
    ax5.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['pz'],smooting)[cut_off:-cut_off], "tab:blue", lw=0.5, label="FJZ", zorder=2)
                   
    ax5.scatter(df['times_mjd']/time_scaling, df['p1'], c="tab:orange", s=7, alpha=0.4, zorder=2)
    ax5.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['p1'],smooting)[cut_off:-cut_off], "tab:orange", lw=0.5, label="CCW", zorder=2)
    
    ax5.scatter(df['times_mjd']/time_scaling, df['p2'], c="tab:red", s=7, alpha=0.4, zorder=2)
    ax5.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['p2'],smooting)[cut_off:-cut_off], "tab:red", lw=0.5, label="CW", zorder=2)
                  
        
    ## Panel 6 -------------------------
    ax6.scatter(df['times_mjd']/time_scaling, df['ac_z'], c="tab:green", s=7, alpha=0.4, zorder=2)
    ax6.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['ac_z'],smooting)[cut_off:-cut_off], "tab:green", lw=0.5, label="AC", zorder=2)
       
    ax6.scatter(df['times_mjd']/time_scaling, df['dc_z'], c="tab:pink", s=7, alpha=0.4, zorder=2)
    ax6.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['dc_z'],smooting)[cut_off:-cut_off], "tab:pink", lw=0.5, label="DC", zorder=2)
       
        
        
    ## Panel 7 -------------------------
    ax7.scatter(df['times_mjd']/time_scaling, df['contrast_z'], c="tab:purple", s=7, alpha=0.4, zorder=2)
    ax7.plot(df['times_mjd'][cut_off:-cut_off]/time_scaling, __smooth(df['contrast_z'],smooting)[cut_off:-cut_off], "tab:purple", lw=0.5, label="FJZ", zorder=2)
         
        
        
    ax5.set_yscale("log")
    
    ax2.set_ylim(0.99999*nanmedian(df.fz), 1.00001*nanmedian(df.fz))
    ax3.set_ylim(0.99998*nanmedian(df.fz), 1.00002*nanmedian(df.fz))
    ax4.set_ylim(0.99998*nanmedian(df.fz), 1.00002*nanmedian(df.fz))
    ax7.set_ylim(0, 3*nanmedian(df.contrast_z))
    
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax3.ticklabel_format(useOffset=False, style='plain')
    ax4.ticklabel_format(useOffset=False, style='plain')
    
    
    ax1.set(xticklabels=[]) 
    ax2.set(xticklabels=[]) 
    ax3.set(xticklabels=[]) 
    ax4.set(xticklabels=[])    
    
    ax1.set_ylabel(r"Sagnac Beat (Hz)", fontsize=font)
    ax2.set_ylabel(r"FZ (Hz)", fontsize=font)
    ax3.set_ylabel(r"CW (Hz)", fontsize=font)
    ax4.set_ylabel(r"CCW (Hz)", fontsize=font)
    ax5.set_ylabel(r"max. Power (V$^2$/Hz)", fontsize=font)
    ax6.set_ylabel(r"signal(V)", fontsize=font)
    ax7.set_ylabel(r"contrast", fontsize=font)

    ax7.ticklabel_format(axis="x", useOffset=False, style='plain')
    ax7.set_xlabel("Days (MJD)", fontsize=font)
    
    ax1.set_title("ROMY-Z Sagnac-Frequency ", fontsize=font+1, pad=10)
    
    ax1.legend(ncol=3)
    ax5.legend(ncol=3)
    ax6.legend(ncol=2)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.grid(alpha=0.8, ls=":", zorder=0)

#     plt.show();
    return fig

## _______________________________________
def main(config):

    ## current date
    date = (UTCDateTime.now()-86400).date
    
    ## read data
    filename = f"FJ{config['ring']}_{date.replace('-','')}"
    df = read_pickle(config['path_to_data']+filename+".pkl")

    ## create figure
    fig = __makeplot_overview(df)
    
    ## save figure
    fig.savefig(f"{config['path_to_figs']}{filename}.png", dpi=200, facecolor='w', bbox_inches='tight',
                edgecolor='w', orientation='portrait', format='png', transparent=False, pad_inches=0.1)
    
## _______________________________________
if __name__ == "__main__":
    main(config)

## End of File