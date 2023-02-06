#!/usr/bin/env python
# coding: utf-8

# from numpy import kaiser, hanning, linspace
from obspy import Stream, Trace

def __tapering(idata, taper_type='hann', percent=0.2):

    ''' 
        taper_type:   type of taper window (default: 'hann')
        percent:      percent to taper (default: 0.2)
    '''
    
    st = Stream(Trace(data=idata))

    st.taper(percent, taper_type)    
    
    return st[0].data


#     if taper_type == "flanks":
#         flank_size = int(percent/100 * idata.size) # percent        
        
#         idata[:flank_size] = idata[:flank_size] * linspace(0,1,flank_size)
#         idata[-flank_size:] = idata[-flank_size:] * linspace(1,0,flank_size)

#     elif taper_type == "window":
# #         idata = idata * hanning(idata.size) 
#         idata = idata * kaiser(idata.size, 8.6)
        
#     else:
#         print("type not found!")
        
#     return idata