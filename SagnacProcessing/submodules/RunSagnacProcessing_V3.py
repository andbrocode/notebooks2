#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import matplotlib.pyplot as plt 
import time

from scipy.signal import resample, hilbert, correlate, decimate, butter, spectrogram, sosfilt, filtfilt, iirnotch
from tqdm import tqdm
from obspy import UTCDateTime, read, read_inventory, Trace
import obspy

import sys
sys.path.insert(0, 'submodules')

from EchoPerformance import __echo_performance
from CreateSyntheticEventV2 import __create_synthetic_event_v2
from MinimizeResidual import __minimize_residual
from CreateLinearChirp import __create_linear_chirp
from Tapering import __tapering
from InstaFreqHilbert import __insta_freq_hilbert
from Normalize import __normalize
from WriteToMseed import __write_to_mseed
from LoadMseed import __load_mseed
from Modulation import __modulation
from QuerrySeismoData import __querry_seismo_data
from RingLaser import RingLaser
from Interpolation import __interpolation_fft
from UpsampleFIR import __upsample_FIR
from DemodulateWithWindows import __demodulate_with_windows
from HilbertFilter import __hibert_filter
from andbro__fft import __fft
from Downsample import __downsample


from MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum
from MakeplotTraceAndPSD import __makeplot_trace_and_psd
from MakeplotModulatedSignal import __makeplot_modulated_signal
from MakeplotDemodulationQuality import __makeplot_demodulation_quality


def __run_sagnac_processing(
                        sgnc, 
                        T, 
                        sps, 
                        upsampling_factor, 
                        f_lower_bp, 
                        f_upper_bp, 
                        signal, 
                        Twindow,
                        win_type, 
                        perc,
                        f_upper_lp,
                        modeltrace,
                        time_modeltrace,
                        timeline,
                        synthetic_signal,
                        bools,
                        show=True,
                       ):
    
    taper = bools['taper']
    upsampling = bools['upsampling']
    bandpass_pre = bools['bandpass_pre']
    lowpass_post = bools['lowpass_post']
    normalize = bools['normalize']
    remove_median = bools['remove_median']


    ## define G-Ring
    G = RingLaser(side=4., form="square", wl=632.8e-9, lat=49.16)

    ## define ROMY Z-Ring
    ROMY_Z = RingLaser(side=11., form="triangle", wl=632.8e-9, lat=49.16)

    
    ## ___________________________________________________
    ## Remove Median
    
    if remove_median:
        med = np.median(synthetic_signal)
        synthetic_signal= synthetic_signal - np.median(synthetic_signal)

        
    ## ___________________________________________________
    ## Taper
    
    if taper:
        synthetic_signal = __tapering(synthetic_signal, win_type, perc)
    
    
    ## ___________________________________________________
    ## Bandpass

    if bandpass_pre:

        ## create butterworth bandpass
        b, a = butter(4, [f_lower_bp, f_upper_bp], 'bp', fs=sps)

        ## apply butterworth bandpass forward and backwards
        synthetic_signal = filtfilt(b, a, synthetic_signal, method="pad")

    
    ## ___________________________________________________
    ## Upsampling

    if upsampling:


    #     synthetic_signal, timeline = __upsample_FIR(synthetic_signal, 
    #                                                 sps, 
    #                                                 T, 
    #                                                 sampling_factor=upsampling_factor,
    #                                                )

        synthetic_signal, timeline = __interpolation_fft(synthetic_signal, 
                                                         timeline, 
                                                         T, 
                                                         sps*upsampling_factor,
                                                        )


        ## adjust sampling rate
        sps = sps*upsampling_factor    

    
    
    ## ___________________________________________________
    ## Demodulation      
    
    demod_signal, time_demod_signal = __demodulate_with_windows(synthetic_signal, timeline, Twindow, sps)
    
    
    
    demod_signal = demod_signal[1:]
    demod_signal= np.append(demod_signal, 0)
    
    ## ___________________________________________________
    ## Remove Offset  
    
    demod_signal = demod_signal - np.median(demod_signal)    
    
    
    ## ___________________________________________________
    ## Lowpass
    
    if lowpass_post:
  
        ## create digital lowpass filter, which is applied forwards and backwards
        b, a = butter(8, f_upper_lp, 'lp', fs=sps)
        demod_signal = filtfilt(b, a, demod_signal, method="pad")
    
    
    ## ___________________________________________________
    ## Downsampling
    
    if upsampling:


        downsampling_factor = upsampling_factor

        sps0 = sps ## prevent overwriting
        demod_signal, time_demod_signal, sps = __downsample(demod_signal, sps0, ds_factor=downsampling_factor)
    #     modeltrace, time_modeltrace, sps = __downsample(modeltrace, sps0, ds_factor=downsampling_factor)

    
    ## ___________________________________________________
    ## Conversion and Normalization
    
    demod_signal *= 0.59604645e-6
    
    if normalize:
        demod_signal = __normalize(demod_signal)
        modeltrace   = __normalize(modeltrace)

    
    
    ## ___________________________________________________
    ## Plot
    
    ## samples from left and right to be ignored due to dynamic demodulation
    index1 = int(time_demod_signal[0]*sps)

    if upsampling:
        index2 = int(int(T-time_demod_signal[-1])*sps)
    else:
        index2 = int(int(T-time_demod_signal[-1])*sps)

    ## padding demodulated signal at the edges
    demod_signal = np.pad(demod_signal, (index1, index2), constant_values=(0, 0))

    ## adjust time axis
    time_demod_signal = np.arange(len(demod_signal))/sps

    if lowpass_post:
        cut1 = int(index1)
        cut2 = int(len(demod_signal)-index2)
    else:
        cut1 = int(index1)
        cut2 = int(len(demod_signal)-index2)   
    

    cut1, cut2
    
    ## calulcate cross-correlation
    cross_corr = correlate(demod_signal[cut1:cut2], modeltrace[cut1:cut2], mode='same')

    cross_corr_lags = np.arange(-cross_corr.size//2+1,cross_corr.size//2+1,1)   
    
    ## final plot
    if show:
        fig = __makeplot_demodulation_quality(time_modeltrace, 
                                              modeltrace, 
                                              time_demod_signal,
                                              demod_signal,
                                              cross_corr,
                                              cross_corr_lags,
                                              sps,
                                              cut1,
                                              cut2,
                                              fmax=10
                                             );

    
    ## caluclate spectral magnitude and phase
    asd_demod, ff_demod, phase_demod = __fft(demod_signal, 1/sps, window=None, normalize=None)
    asd_model, ff_model, phase_model = __fft(modeltrace, 1/sps, window=None, normalize=None)

    
    ## ___________________________________________________
    ## Output    

    rmse_trace = np.sqrt(np.mean((demod_signal[cut1:cut2]-modeltrace[cut1:cut2])**2))
    rmse_asd   = np.sqrt(np.mean(asd_demod - asd_model)**2)
    rmse_phase = np.sqrt(np.mean(np.unwrap(phase_demod) - np.unwrap(phase_model))**2)
    
    cc_lag_max = cross_corr_lags[abs(cross_corr).argmax()]/sps
    
    print("DONE")
    print("_______________________________")
    
    return cc_lag_max, rmse_trace, rmse_asd, rmse_phase