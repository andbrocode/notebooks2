#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from numpy import sqrt, mean, unwrap
from andbro__fft import __fft

def __makeplot_demodulation_quality(time_modeltrace, modeltrace, time_demod_signal, demod_signal, cross_corr, cross_corr_lags, sps, c1, c2):
    
    def __minimize_residual(model, original):

        from scipy.optimize import leastsq

        ## define cost function
        def __cost_function(params, x, y):
            a, b = params[0], params[1]
            residual = y-(a*x+b)
            return residual

        ## initials 
        params = [1,0]

        result = leastsq(__cost_function, params, (model, original))

        model_new = model * result[0][0] + result[0][1]

        print(f'\noptimized: original -  {round(result[0][0],3)} * model + {round(result[0][1],3)}\n')

        residual = (model_new - original)
        return residual, model_new

    
    ## caluclate spectral magnitude and phase
    asd_demod, ff_demod, phase_demod = __fft(demod_signal, 1/sps, window=None, normalize=None)
    asd_model, ff_model, phase_model = __fft(modeltrace, 1/sps, window=None, normalize=None)


    
    ## ______________________________________

    residual_pre_opt  = modeltrace[c1:c2]-demod_signal[c1:c2]
    
    residual_post_opt, demod_signal_opt  = __minimize_residual(demod_signal[c1:c2], modeltrace[c1:c2])
        
    max_amp = max(abs(demod_signal))
    
    rmse_pre_opt  = sqrt(mean(residual_pre_opt**2))
    rmse_post_opt = sqrt(mean(residual_post_opt**2))
    
    ## ________________________________________________
    ## plotting
    
    fig, ax = plt.subplots(5, 1, figsize=(15,12))

    font = 13
    
    plt.subplots_adjust(hspace = 0.4)
    
    
    ## ____________________________________________________________________________
    ##
    
    ax[0].plot(time_modeltrace, modeltrace, color='darkorange', linewidth=3)
    ax[0].plot(time_demod_signal, demod_signal)
    ax[0].plot(time_demod_signal[c1:c2], demod_signal_opt, color='darkblue', linestyle='--')

    ax[0].set_xlabel("time (s)", fontsize=font)
    ax[0].tick_params(axis='both', labelsize=font-2)
        
    ax[0].set_ylim(-max_amp-0.1*max_amp, max_amp+0.1*max_amp)
    ax[0].set_ylabel('norm. amplitude', fontsize=font)
    
    
    ## ____________________________________________________________________________
    ##
        
    ax[1].plot(time_demod_signal[c1:c2], abs(residual_pre_opt), color='k')
    ax[1].plot(time_demod_signal[c1:c2], abs(residual_post_opt), color='r')

    
    ax[1].set_xlabel("time (s)", fontsize=font)
    ax[1].tick_params(axis='both', labelsize=font-2)
    ax[1].set_ylabel('norm. residual', fontsize=font)
    
    ax[1].set_xlim(time_demod_signal[0], time_demod_signal[-1])
    
#     x1 = 0.1*time_demod_signal.size*(time_demod_signal[1]-time_demod_signal[0])
#     y1 = 0.75*max(residual_pre_opt)    
#     print(max(residual_post_opt))
#     ax[1].text(x1, y1, f'rmse pre: {round(rmse_pre_opt,5)} \nrmse post: {round(rmse_post_opt,5)}', fontsize=font)    
    
    ax[0].set_title(f'rmse pre: {round(rmse_pre_opt,5)};  rmse post: {round(rmse_post_opt,5)}', fontsize=font)
    
    
    ## ____________________________________________________________________________
    ##
    
    ax[2].axvline(0.0, color="grey", ls=":")
    ax[2].plot(cross_corr_lags*(time_demod_signal[1]), cross_corr,'k')
    
    ax[2].set_xlabel('lag (s)', fontsize=font)
    ax[2].tick_params(axis='both', labelsize=font-2)
    ax[2].set_ylabel('cross-correlation', fontsize=font)
    
    x0 = 0.60*max(cross_corr_lags)*(time_demod_signal[1]-time_demod_signal[0])
    y0 = 0.75*max(cross_corr)
    
    ax[2].text(x0, y0, f'max. CC at lag {cross_corr_lags[abs(cross_corr).argmax()]/sps} s', fontsize=font)
#     ax[2].text(x0, y0, f'max. CC at lag {cross_corr_lags[abs(cross_corr).argmax()]}', fontsize=font)

    
    ## ____________________________________________________________________________
    ##
    
    ax[3].plot(ff_model, asd_model, color='k')
    ax[3].plot(ff_demod, asd_demod, color='green')

    ax[3].set_xlabel("Frequency (Hz)", fontsize=font)
    ax[3].set_ylabel(f'spect. amplitude \n (rad/s $\sqrt{Hz}$)', fontsize=font)
    
    
    ## ____________________________________________________________________________
    ##
    
    ax[4].plot(ff_model, unwrap(phase_model), color='k')
    ax[4].plot(ff_demod, unwrap(phase_demod), color='green')

    ax[4].set_xlabel("Frequency (Hz)", fontsize=font)
    ax[4].set_ylabel('unwraped phase', fontsize=font)    
    
        
    ## ____________________________________________________________________________
    ##
    
    ax[0].legend(['synthetic', 'demodulated', 'demod. optimized'], loc="upper right", fontsize=font-2)
    ax[1].legend(['residual', 'optimized res'], loc="upper right", fontsize=font-2)

    ax[0].set_xlim(time_demod_signal[0], time_demod_signal[-1])
    ax[1].set_xlim(time_demod_signal[0], time_demod_signal[-1])
    ax[2].set_xlim(-time_demod_signal[-1]/2, time_demod_signal[-1]/2)
    ax[3].set_xlim(0, 0.2)

    plt.show();
    
    return fig