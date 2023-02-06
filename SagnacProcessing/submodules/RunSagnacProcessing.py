#!/usr/bin/env python
# coding: utf-8

from numpy import arange, median, array, mean, sqrt, random
from scipy.signal import resample, hilbert, correlate

from EchoPerformance import __echo_performance
from CreateSyntheticEventV2 import __create_synthetic_event_v2
from MinimizeResidual import __minimize_residual
from CreateLinearChirp import __create_linear_chirp
from Tapering import __tapering
from InstaFreqHilbert import __insta_freq_hilbert
from LoadMseed import __load_mseed
from Modulation import __modulation
from Normalize import __normalize
from MakeplotDemodulationQuality import __makeplot_demodulation_quality
from MakeplotTraceAndSpectrum import __makeplot_trace_and_spectrum

def __run_sagnac_processing(sgnc, T, sps, oversampling, mod_index, f_lower, f_upper, syn_signal, taper_type, taper_percent=0.2, noise_level=None):
    
    sps *= oversampling
    
    ## ___________________________________________________
    ## Synthetic Signal
    
    modeltrace, time_modeltrace = __create_linear_chirp(T, sps, f_lower, f_upper)

    
    ## __________________________________
    if syn_signal == 'synthetic_trace':
        
        ## load trace or create and write one if not yet existing
        iname=f"data/SYN_T{int(T)}_fs{int(sps)}_f{f_lower}-{f_upper}.mseed"
        
        
        modeltrace, time_modeltrace = __load_mseed(iname, T, sps, f_lower, f_upper, noise_level=noise_level)

        modeltrace = __tapering(modeltrace, taper_type, taper_percent)

         ## normalize trace to avoid issues with demodulation
        modeltrace = __normalize(modeltrace)

    ## __________________________________
    elif syn_signal == 'chirp':
        
        ## generate normalized chrip signal 
        modeltrace, time_modeltrace = __create_linear_chirp(T, sps, f_lower, f_upper)

        if noise_level is not None:

            print("whitening factor:", noise_level)
            noises = array(random.rand(modeltrace.size))
            modeltrace += noise_level * noises
        
    ## __________________________________    
    elif syn_signal == 'real_trace':

        ## set starttime and endtime
        tbeg = obspy.UTCDateTime(2020, 10, 30, 12, 5)
        tend = tbeg + T


        RLAS, RLAS_inv = __querry_seismo_data("BW.RLAS..BJZ", 
                                              tbeg, 
                                              tend, 
                                              restitute=True,
                                              )

        RLAS[0].resample(sampling_rate=sps)

        RLAS[0].filter('bandpass', freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

        time_modeltrace = arange(0, T+1/sps, 1/sps)
        modeltrace = RLAS[0].data[0:time_modeltrace.size]

        modeltrace = __normalize(modeltrace)

       
    ## __________________________________
    else:
        print("Wrong choise!")

    __tapering(modeltrace, taper_type, taper_percent)

    
    __makeplot_trace_and_spectrum(modeltrace, time_modeltrace, fmax=0.5*sps);
    
    ## ___________________________________________________
    ## Modulation
    
#     timeline, synthetic_signal = __modulation(time_modeltrace, modeltrace, T, sps, mod_index)
    synthetic_signal, timeline = __modulation(modeltrace,
                                              time_modeltrace, 
                                              sgnc, 
                                              T, 
                                              sps, 
                                              mod_index, 
                                              case = 3,
                                             )    
#     synthetic_signal = __tapering(synthetic_signal, taper_type, taper_percent)

#     __makeplot_trace_and_psd(synthetic_signal, timeline, fmax=0.5*sps, t1=0, t2=2);


    ## ___________________________________________________
    ## Downsample
    
    modeltrace = modeltrace[::oversampling]
    time_modeltrace = time_modeltrace[::oversampling]

    synthetic_signal = synthetic_signal[::oversampling]
    timeline = timeline[::oversampling]

    sps = 1/(timeline[1]-timeline[0])    
    
#     __makeplot_trace_and_psd(synthetic_signal, timeline, fmax=0.5*sps);

#     __makeplot_modulated_signal(synthetic_signal,timeline);

    ## ___________________________________________________
    ## Demodulation
    
    time_demod_signal, demod_signal = __insta_freq_hilbert(synthetic_signal, timeline, sps, sgnc)
    
    
#     demod_signal = demod_signal - sgnc
    demod_signal = demod_signal - median(demod_signal)

    
    
    cutoff = int(0.01*demod_signal.size)
    
    ## store cutoffs for displaying
    cut1 = array([i for i in demod_signal[:cutoff]])
    cut2 = array([i for i in demod_signal[-cutoff:]])
    
    ## zero out 1% at both ends
    demod_signal[:cutoff] = 0.0
    demod_signal[-cutoff:] = 0.0

    
    ## normalize original and demodulated signal
    demod_signal_norm = __normalize(demod_signal)
    modeltrace_norm = __normalize(modeltrace)

    
    ## ___________________________________________________
    ## CrossCorrelation  

    cross_corr = correlate(demod_signal, modeltrace, mode='same')
    cross_corr_lags = arange(-cross_corr.size//2+1,cross_corr.size//2+1,1)
    
    
    idx = abs(cross_corr).argmax()
    
    cc = cross_corr[idx]
    cclag = cross_corr_lags[idx]
    
    print(f"\n max lag CC: {cross_corr_lags.max()} \n min lag CC: {cross_corr_lags.min()}")

    print(f"\n maximal CC: {round(cross_corr[idx],3)} at lag: {cross_corr_lags[idx]}")
    
    ## ___________________________________________________
    ## Residuals 
    
    residual  = modeltrace - demod_signal
    
    residual_pre_opt  = modeltrace_norm - demod_signal_norm

    residual_post_opt, demod_signal_opt  = __minimize_residual(demod_signal_norm, modeltrace_norm)
    
    
    rms_pre  = sqrt(mean(residual_pre_opt**2))
    rms_post = sqrt(mean(residual_post_opt**2))
    rms      = sqrt(mean(residual**2))
    
    print(f'cutoff: {cutoff} --> residual median: {median(residual_post_opt)}')

    
    ## ___________________________________________________
    ## Plot
    
    __makeplot_demodulation_quality(timeline, 
                                    modeltrace_norm, 
                                    demod_signal_norm,
                                    cross_corr,
                                    cross_corr_lags,
                                    cut1,
                                    cut2,
                                    );
    
    print("DONE")
    print("_______________________________")
    
    return cc, cclag, rms_pre, rms_post, rms
