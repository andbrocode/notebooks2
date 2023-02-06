#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import matplotlib
#matplotlib.use('AGG')

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.patches import Rectangle
from obspy import UTCDateTime, read
from obspy.clients.filesystem.sds import Client
from calibrationH import calib_stream, plot_calib_matrix
from obspy.io.xseed import Parser
from obspy.signal import PPSD

# general settings
path = '/home/andbro/Desktop/Huddle_Test/Huddle_Test_1/plots'
path2plot = '/home/andbro/Desktop/Huddle_Test/Huddle_Test_1/plots'

# time window to analyze data
s0 = UTCDateTime('2021-02-11T00:00:00')
s1 = UTCDateTime('2021-02-12T13:00:00')

# plot parameters
cmap = cm.seismic
# change title string
title_str = 'Huddle Test 1'

# data client
client = Client(sds_root=path)
# chns = ["HHZ", "HHN", "HHE"]
chns = ["BHZ", "BHN", "BHE"]


# Parameter setting
######################################################
# remove response

typ = ['TH', 'TC','LE','PH'] #instrument types (must be in station name)
gains = {'TH': 1200. * 1 * 4E+05, 'TC': 754. *2* 1 * 4E+05,'LE': 1 * 4E+02 * 6.291290E+05, 'PH': 753. * 4E+05 *2* 1} #gains from instrument and digitizer
save = 1 # bool ## Save figure if True, else display it


### LOOP over 1hr data windows
for chn in chns:
    print('Channel: ' + chn)
    cnt = 0 # count number of spectral ratios in sum
    timestr = "%s - %s" % (s0.strftime("%Y.%m.%dT%H:%M"), (s1-1).strftime("%Y.%m.%dT%H:%M"))
    st1 = client.get_waveforms('*','*','*',chn,s0,s1)
    s = st1.copy()
    Z = s
    Z.merge()
    Z.sort()
    print(Z.__str__(extended=True))
    fn = path2plot + '/Huddle' + chn
    Z.plot(outfile=fn)
    t0 = s0
    t1 = s1
    while t0 <= t1 - 3600:
        print("processing t0: %s" % str(t0))
        sens = calib_stream(Z.slice(t0,t0+3600),1)
        
        if cnt == 0:
            senssum = sens
        else:
            senssum = np.dstack((senssum, sens))
        cnt += 1
        t0 += 3600
        
    #### Postprocessing
    lbls = [tr.stats.station for tr in Z]
    print(lbls)
    m = np.median(senssum, axis = 2 ) # FINAL sensitivity matrix is median of all calculated matrices
    pct = (m - 1) * 100 # deviation in percent

    d = m.copy()
    for i in range(len(lbls)):
        if lbls[i][0:2]=='TH':
            g =  gains['TH']
        elif lbls[i][0:2]=='TC':
            g =  gains['TC']
        elif lbls[i][0:2]=='LE':
            g = gains['LE']
	else:
	    g = gains['PH']

        
        for j in range(len(lbls)):
            #print('j gleich' + str(j))
            if lbls[j][0:2]=='TH':
                gg = gains['TH']
            elif lbls[j][0:2]=='TC':
                gg = gains['TC']
            elif lbls[j][0:2]=='LE':
                gg = gains['LE']
            else:
                gg = gains['PH']
            
            d[i,j] = (m[i,j]*g - gg) * 100 / gg # relative deviation in % from nominal value
    # Figures
            
    fname = path2plot + '/huddle_result_%s_%s_%s_0.3-3Hz.png' % ("-".join(typ), chn,'RR') if save else None
    plot_calib_matrix(d, title_str + ' %s - %s' % (timestr, chn), lbls, fname=fname, cmap=cmap, 
                      clabel='deviation from nominal sensitivity in %', vmax=3)
    if fname is not None:
        np.savez(fname.rsplit(".", 1)[0], senssum=senssum, lbls=lbls)
