#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt


def __makeplot_compare_schemes(sig1, time1, sig2, time2, number_of_samples=100, shift_of_window=0):
    
    if time1.size != time2.size:
        print("array size does not match!")
    
    dt = time1[2]-time1[1]
    
    ## __________________________________________________________
    ##
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 10))

    plt.subplots_adjust(hspace=0.4)
    
    font = 13 
    num0 = shift_of_window
    num = number_of_samples+ num0
    
    ax1.plot(time1[num0:num], sig1[num0:num], color='darkblue')
    ax1.scatter(time1[num0:num], sig1[num0:num], s=10)
    
    ax1.plot(time1[num0:num], sig2[num0:num], color='darkorange')
    ax1.scatter(time1[num0:num], sig2[num0:num], s=10)

    
    ax2.plot(time1[num0:num], sig1[num0:num]-sig2[num0:num],'grey')
    ax2.scatter(time1[num0:num], sig1[num0:num]-sig2[num0:num], s=10, c='k')

    ax3.plot(time1, sig1, color='darkblue')
    
    ax4.plot(time1, sig2, color='darkorange')
    
    
    ax5.plot(time1, sig1-sig2,'grey')
    
    if shift_of_window != 0:
        ax3.axvline(shift_of_window*dt, color='r')
        ax4.axvline(shift_of_window*dt, color='r')
        
    
    ax2.set_xlabel("Time (s)", fontsize=font)
    
#     ax2.set_ylim(-1,1)
    
    plt.show();
    
    return fig
