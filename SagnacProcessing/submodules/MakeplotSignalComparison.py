#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt


def __makeplot_signal_comparison(time_in1, sig_in, time_in2, event_in, option, sps, sgnc):

    x1 = 5
    x2 = 10

    if option == 'option1' or option == 'option2':
        signal = sig_in - sgnc
    
    elif option == 'option3':
        signal = sig_in

    ## _____________________________________________    
    ## plotting 
    
    fig, ax = plt.subplots(1, 1, figsize=(15,5))

    ax.plot(time_in2, event_in, color='black')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')


    ax.set_xlim(x1,x2)
    ax.set_ylim(-max(abs(event_in[x1*sps:x2*sps])),max(abs(event_in[x1*sps:x2*sps])))

    ax2 = ax.twinx()
    ax2.plot(time_in1, signal,color='red')

    ax2.set_ylabel('$\Delta$ f (Hz)')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    
    ## adjust y axis in scale depending on the demodulation option
    if option == 'option1':
        ax2.set_ylim(-max(abs(signal[int(x1*sps/2):int(x2*sps/2)])),max(abs(signal[int(x1*sps/2):int(x2*sps/2)])))
    elif option == 'option2':
        ax2.set_ylim(-max(abs(signal[int(x1*sps):int(x2*sps)])),max(abs(signal[int(x1*sps):int(x2*sps)])))
    elif option == 'option3':
        ax2.set_ylim(-max(abs(signal[int(x1*sps):int(x2*sps)])),max(abs(signal[int(x1*sps):int(x2*sps)])))


    plt.show();

    return fig
