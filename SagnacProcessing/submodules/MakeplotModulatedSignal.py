#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

def __makeplot_modulated_signal(sig, tt):
    
    font = 13

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15,5))

    ax1.plot(tt,sig, color="grey")

    ax2.plot(tt,sig, color="r")
    ax2.scatter(tt, sig, color="black", s=10)

    ax3.plot(tt,sig, color="b")
    ax3.scatter(tt, sig, color="black", s=10)
    
    ax4.plot(tt,sig, color="g")
    ax4.scatter(tt, sig, color="black", s=10)    
    
    
    
    ax1.tick_params(axis='both', labelsize=font)
    ax2.tick_params(axis='both', labelsize=font)
    
    ax1.set_xlabel('Time (s)', fontsize=font)
    ax1.set_ylabel('Amplitude', fontsize=font)

    ax2.set_xlabel('Time (s)', fontsize=font)
    ax2.set_ylabel('Amplitude', fontsize=font)
    
    deltaT = 200*(tt[1]-tt[0])
    deltaT = 0.08
    
    t1 = 0
    t2 = 0.5*sig.size*(tt[1]-tt[0])
    t3 = sig.size*(tt[1]-tt[0])-deltaT
    
    ax2.set_xlim(t1, t1+deltaT)
    ax3.set_xlim(t2, t2+deltaT)
    ax4.set_xlim(t3, t3+deltaT)
    
    ax1.axvline(t1, color='r', zorder=3)
    ax1.axvline(t2, color='b', zorder=3)
    ax1.axvline(t3, color='g', zorder=3)
    
    plt.subplots_adjust(hspace=0.5)
    
    plt.show();

    return fig