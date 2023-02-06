#! /usr/bin/env python

import sys, os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from matplotlib import rc, font_manager
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from MyPPSD import PPSD
from obspy import read_inventory
from obspy.imaging.cm import pqlx


def plot_diagram(data, cmap, cb_label, chas):
###########################################################################
# PLOTS
    params = {'text.usetex': True, 
            'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
    plt.rcParams['figure.figsize'] = 6, 6
    plt.rcParams.update(params)
    sizeOfFont = 15
    fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}
    rc('font',**fontProperties)

    fig = plt.figure()
    ax00 = plt.subplot2grid((1, 1), (0, 0))
    axes = [ax00]


    c = (0,0,0)
    props = dict(boxstyle='round', facecolor=(1,1,1), alpha=0.8)
    for (X, Y, d, median), cha, ax in zip(data, chas, axes):
        line_median = ax.loglog(X[0], median, color=(1,1,1))
        ppsd = ax.contourf(X, Y, d.T, cmap=cmap, levels=100)
        ax.grid(b=True, which='major', color=c, linestyle='-', linewidth=0.5, alpha=0.6)
        ax.tick_params(axis='x', which='major', pad=8)
        ax.yaxis.set_ticks_position('both')
        ax.set_xlim(1.2e-3, 2e0)
        ax.set_ylim(1e-13, 3e-10)

        ax.text(1.9e-3, 1.6e-10, cha, bbox=props)   

        ax.tick_params(axis='x', which='both', direction='in', top=True, color=c)
        ax.tick_params(axis='y', which='both', direction='in', top=True, color=c)

        axins1 = inset_axes(ax,
                    width="50%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='lower right',
                    bbox_to_anchor=(-0.1, 0.12, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
        cb = plt.colorbar(ppsd, cax=axins1, orientation="horizontal", ticks=np.round(np.linspace(0, np.amax(d), 4, endpoint=True),0))
        cb.set_label(cb_label, color=c, fontsize=12)
        cb.ax.xaxis.set_tick_params(color=c)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=c, fontsize=12)
        cb.outline.set_edgecolor(c)


    ax00.set_xlabel('frequency [Hz]')
    ax00.set_ylabel('rotation rate spectral density [rads$^{-1}$Hz$^{-1/2}$]')
    plt.subplots_adjust(bottom=0.1,
                        right=0.980,
                        top=0.985,
                        left=0.15,
                        wspace=0.0,
                        hspace=0.0)


    
    plt.savefig('plots/TW_array_rot_ppsd_'+cha+'.png', dpi=300)
    plt.show()

def dB2rmsPSD(y):
    return 10**(y/20.0)

def get_ppsd(fname):
    ppsd = PPSD.load_npz(fname)

    data = (ppsd.current_histogram * 100 / (ppsd.current_histogram_count or 1))
    xedges = 1.0 / ppsd.period_xedges
    X, Y = np.meshgrid(xedges[:-1], dB2rmsPSD(ppsd.db_bin_edges)[:-1])

##########################################################
# this gets you the data for the white line:
    (p, _median) = ppsd.get_percentile(percentile=50)
    median = dB2rmsPSD(_median)
    return X, Y, data, median
     


def main():
# PPSD for the array:
    fnameZ = 'TW_array_Z_2021_01ppsd.npz'
    fnameN = 'TW_array_N_2021_01ppsd.npz'
    fnameE = 'TW_array_E_2021_01ppsd.npz'

#########################################
# enter the component here:

##### EAST ###################
#    fnames = [fnameE]

##### NORTH ##################
#    fnames = [fnameN]

##### VERTICAL ###############
    fnames = [fnameZ]

#########################################
    data = []
    chas = []
    for fname in fnames:
        X, Y, d, median = get_ppsd(fname)
        data.append((X, Y, d, median))
        cha = fname.split('_')[2]
        chas.append(cha)
    cmap = 'magma_r'
    cb_label = r'\%'


    plot_diagram(data, cmap, cb_label, chas)


if __name__ == "__main__":
    main()


