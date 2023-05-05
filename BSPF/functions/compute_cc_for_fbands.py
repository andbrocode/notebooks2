#!/bin/python3

def __compute_cc_for_fbands(tr1, tr2, fmin=1, fmax=20, fband_type='octave'):
    
    
    def __get_frequency_bands(f_min=fmin, f_max=fmax, fband_type=fband_type):

        from numpy import sqrt, array

        f_lower, f_upper, f_centers = [], [], []
        fcenter = f_max

        while fcenter > f_min:
            f_lower.append(fcenter/(sqrt(sqrt(2.))))
            f_upper.append(fcenter*(sqrt(sqrt(2.))))
            f_centers.append(fcenter)

            fcenter = fcenter/(sqrt(2.))

        return array(f_lower), array(f_upper), array(f_centers)
    
    
    from numpy import array, corrcoef, correlate
    from obspy.signal.cross_correlation import correlate
    
  
    
    f_lower, f_upper, f_centers = __get_frequency_bands(f_min=fmin, f_max=fmax, fband_type=fband_type)


    ccorrs, ccorrs_max = [], []
    for fl, fu in zip(f_lower, f_upper):

        tr01 = tr1.copy();
        tr02 = tr2.copy();

        tr01.detrend("linear")
        tr02.detrend("linear")

        tr01.normalize();
        tr02.normalize();
        
        tr01.taper(0.1);
        tr02.taper(0.1);
        
        tr01 = tr01.filter("bandpass", freqmin=fl, freqmax=fu, corners=8, zerophase=True);
        tr02 = tr02.filter("bandpass", freqmin=fl, freqmax=fu, corners=8, zerophase=True);

        tr01.normalize();
        tr02.normalize();        
        
        ccorrs.append(corrcoef(tr01.data, tr02.data)[0][1])
        
        cc = correlate(tr01.data, tr02.data, int(len(tr1[0].data)/2))

        ccorrs_max.append(max(cc))
        
        
#         plt.figure()
#         plt.plot(tr01.data)
#         plt.plot(tr02.data) 
    
    return array(f_centers), array(ccorrs), array(ccorrs_max)


## End of File