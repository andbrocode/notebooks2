#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from submodules.EchoPerformance import __echo_performance


def __modulation(data, timeline, f_sgnc, T, sps, mod_index, case=3):

    '''
    Modulation of input data signal. Three cases are distinguished. Case 3 should be preferred.
    
    VARIABLES:
        data
        timeline
        f_sgnc
        T
        sps
        mod_index
        case

    DEPENDENCIES:
        import numpy as np
        import time
        from submodules.EchoPerformance import __echo_performance


    OUTPUT:
        synthetic signal
        time axis

    EXAMPLE:
        >>> s, t  = __modulation(data, timeline, f_sgnc, T, sps, mod_index, case)
    '''
    
    
    ## check if arrays have the same size
    if data.size != timeline.size:
        print(f"Difference in lengths: event = {data.size} and timeline ={timeline.size}")

    ## _______________________________________________
    ##
    if case == 1:
        print(f"\nModulation option {case} is executed ...")
        
        ## modulation 
        synthetic_signal = np.sin(2*np.pi*(f_sgnc + mod_index * data) * timeline)
    
    ## _______________________________________________
    ##    
    if case == 2:
        print(f"\nModulation option {case} is executed ...")

        ## amplitude of carrier frequency
        A_c = 1.0
        
        factor = 0.1 # carrier deviation / 
        
        w_c = 2*np.pi*f_sgnc
        
        synthetic_signal = A_c * np.sin(w_c * timeline - mod_index * np.cos(data*timeline))
        
        ## alternative exchanging sine and cosine
#         synthetic_signal = A_c * np.cos(w_c * timeline - factor * np.sin(data*timeline))

    ## _______________________________________________
    ##    
    if case == 3:
        print(f"\nModulation option {case} is executed ...")

        A_c = 1.0
        

        Npts = int(T*sps)
        
        tt = np.arange(0, T+1/sps, 1/sps)
        

        fm =  mod_index * data
#         fm = sgnc + mod_index * modeltrace

        ## _______________________
        ## start clock
        t1 = time.time()
    
        print('\n --> integrating ... '); time.sleep(1)       
        
        ifm = np.zeros(Npts+1)
        summe = 0

        for i in range(0, Npts):
            summe += fm[i]
            ifm[i] = 2*np.pi*summe
        
        ## _______________________
        ## start clock
#         t1 = time.time()
        
#         print('\nintegrating ... '); time.sleep(1)

#         ifm = np.array([fm[:i].dot(fm[:i]) for i in tqdm(range(0,modeltrace[::fraction].size))])

#         ## resample to original length
#         if fraction > 1:
#             print("resampling ...")
#             ifm = resample(ifm, timeline.size)

            
            
        ifm = ifm / sps # to finish integral f(t)*dt

        synthetic_signal = A_c * np.sin(2*np.pi*f_sgnc*timeline + ifm)
#         synthetic_signal = A_c * np.sin(2*np.pi*ifm)
        
        ## end clock
        t2 = time.time()    
        
        ## show performance
        __echo_performance(t1, t2)
    
        
        
    return synthetic_signal, timeline
