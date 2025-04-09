#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import obspy as obs
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functions.read_sds import __read_sds

def load_ffbi_data(tbeg, tend, archive_path):
    """Load FFBI pressure data"""
    
    try:
        # Initialize empty stream
        ffbi = obs.Stream()
        
        # Read absolute and differential pressure
        ffbi += __read_sds(archive_path+"temp_archive/", "BW.FFBI.30.BDO", tbeg, tend)
        ffbi += __read_sds(archive_path+"temp_archive/", "BW.FFBI.30.BDF", tbeg, tend)
        
        # Merge data
        ffbi = ffbi.merge(fill_value=np.nan)
        
        return ffbi
        
    except Exception as e:
        print(f"Error loading FFBI data: {e}")
        return None

def plot_helicorder(stream, tbeg, tend, hours_per_row=1, num_rows=24, 
                   channel="BDO", ylim=None, title=None):
    """Create helicorder plot for pressure data"""
    
    # Calculate time parameters
    row_duration = hours_per_row * 3600  # seconds per row
    trace = stream.select(channel=channel)[0]
    
    # Create figure
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(num_rows, 1)
    
    # Plot each row
    for i in range(num_rows):
        row_start = tbeg + i * row_duration
        row_end = row_start + row_duration
        
        # Create subplot
        ax = fig.add_subplot(gs[i])
        
        # Get data for this time window
        row_trace = trace.slice(row_start, row_end)
        times = row_trace.times()
        
        # Plot data
        ax.plot(times, row_trace.data, 'tab:blue', linewidth=1)
   
        # Format axis
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlim(0, row_duration)

        # Add timestamp at start of row with fixed position
        # Use figure coordinates instead of axis coordinates to ensure consistent positioning
        # ax.text(-0.01, 0.5, row_start.strftime('%H:%M'),
        #         transform=ax.get_yaxis_transform(),
        #         ha='right', va='center')
        # Add timestamp at start of row with fixed position
        # Use fixed transform to ensure consistent positioning regardless of ylim
        fig_coords = ax.transAxes + fig.transFigure.inverted()
        fig.text(fig_coords.transform((0, 0.5))[0] - 0.01, 
                 fig_coords.transform((0, 0.5))[1],
                 row_start.strftime('%H:%M'),
                 ha='right', va='center')
        
        # Remove unnecessary ticks and labels
        if i < num_rows-1:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i != num_rows-1:
            ax.spines['bottom'].set_visible(False)
        if i == num_rows-1:
            ax.set_xlabel('Time (s)')
            ax.minorticks_on()
    
    # Add title
    if title:
        fig.suptitle(title, y=0.93)
    
    # Add amplitude scale information
    if ylim:
        amplitude_info = f"Amplitude range: [{ylim[0]:.2g}, {ylim[1]:.2g}] Pa"
        fig.text(0.1, 0.92, amplitude_info, ha='left', va='top', 
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=3))
    
    # Ensure consistent spacing between subplots
    # plt.tight_layout(rect=[0.05, 0.03, 1, 0.93])
    
    return fig

def main():
    # Setup paths based on hostname
    if os.uname().nodename == 'lighthouse':
        archive_path = '/home/andbro/freenas/'
    elif os.uname().nodename in ['kilauea', 'lin-ffb-01', 'ambrym', 'hochfelln']:
        archive_path = '/import/freenas-ffb-01-data/'
    
    # Set time window
    if len(sys.argv) > 1:
        tbeg = UTCDateTime(sys.argv[1])
        tend = tbeg + 24*3600  # 24 hours
    else:
        tbeg = UTCDateTime("2024-03-01")
        tend = tbeg + 24*3600  # 24 hours
    
    # Load data
    ffbi = load_ffbi_data(tbeg-1800, tend+1800, archive_path)
    
    if ffbi is None:
        print("Failed to load FFBI data")
        return

    # Apply bandpass filter
    ffbi_filt = ffbi.select(channel="BDO").merge().copy()
    ffbi_filt = ffbi_filt.detrend('linear')
    ffbi_filt = ffbi_filt.detrend('demean')
    ffbi_filt = ffbi_filt.taper(max_percentage=0.05)
    ffbi_filt = ffbi_filt.filter('bandpass', freqmin=0.0005, freqmax=0.03, corners=4, zerophase=True)
    
    ffbi_filt = ffbi_filt.trim(tbeg, tend, nearest_sample=False)

    # Calculate ymax and ensure it's at least 0.1 to avoid very small limits
    ymax = np.nanmax(np.abs(ffbi_filt[0].data))
    ymax = max(ymax, 0.1)  # Ensure minimum scale
    # Round to 2 significant digits for cleaner display
    ymax = float(f"{ymax:.2g}")

    # Create helicorder plots
    fig1 = plot_helicorder(ffbi_filt, tbeg, tend, 
                          channel="BDO", 
                          ylim=(-ymax, ymax),
                          title=f"FFBI Absolute Pressure (BDO) - {tbeg.date}\nBandpass: 0.5-30 mHz")
    
    # Apply bandpass filter
    ffbi_filt2 = ffbi.select(channel="BDF").merge().copy()
    ffbi_filt2 = ffbi_filt2.detrend('linear')
    ffbi_filt2 = ffbi_filt2.detrend('demean')
    ffbi_filt2 = ffbi_filt2.taper(max_percentage=0.05)
    ffbi_filt2 = ffbi_filt2.filter('bandpass', freqmin=0.03, freqmax=0.1, corners=4, zerophase=True)

    ffbi_filt2 = ffbi_filt2.trim(tbeg, tend, nearest_sample=False)

    fig2 = plot_helicorder(ffbi_filt2, tbeg, tend, 
                          channel="BDF",
                          ylim=(-1, 1), 
                          title=f"FFBI Differential Pressure (BDF) - {tbeg.date}\nBandpass: 30-100 mHz")
    
    # Save figures
    path = "/import/kilauea-data/romy_baro/"
    fig1.savefig(f'{path}/figures/helicorders/ffbi_helicorder_BDO_{tbeg.date}.png', dpi=150, bbox_inches='tight')
    fig2.savefig(f'{path}/figures/helicorders/ffbi_helicorder_BDF_{tbeg.date}.png', dpi=150, bbox_inches='tight')
    
    # plt.show()

if __name__ == "__main__":
    main()
