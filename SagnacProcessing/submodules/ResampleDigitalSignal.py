#!/usr/bin/env python
# coding: utf-8


def __resample_digital_signal(signal_in, fs, sampling_factor, taper=False):

    ''' 
    up-samping of the simulated digital signal 
    
    sig_in   = input signal
    fs       = sampling frequency of digital signal
    fac      = sampling factor (2 = double)
    taper    = True/False , to either taper or not taper output
    
    '''
    

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    def resampling(sig_in, factor, fs, case):

        if case == 1:
            sig_out = resample(sig_in,len(sig_in)*factor)

        elif case == 2:
            sig = list(sig_in); i=0

            while i < len(sig):
                sig.insert(i, 0.0)
                i += 2
            sig_out = np.array(sig)

        t_out = np.arange(0,len(sig_in)/fs,1/(fs*factor))

        return  (sig_out, t_out)



    #sig_dig_fil_res, tt_res = resampling(sig_dig_fil,2,fs,2)
    signal_out, time_out = resampling(signal_in, sampling_factor, fs, case=1)

    if taper:
        signal_out = np.hanning(signal_out.size) * sig_nal_out
        #signal_out = butter_bandpass_filter(sig_dig_res_tap,0.1,100,fs,4)
        

    ## _______________________________________________________________________________

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    tt     = time_out
    signal = signal_out
    
    
    ax.plot(tt,signal,color='grey')

    ax.scatter(tt,signal,s=15,color='black')

    ax.scatter(tt[::2],signal[::2],s=15,color='red')
    #ax.plot(tt[1::2],signal[1::2],color='darkred')


    ax.set_xlim(0,0.25)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Counts')

    plt.show();
    
    ## _______________________________________________________________________________
   
    
    return signal_out, time_out
    