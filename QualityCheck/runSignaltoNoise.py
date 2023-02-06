#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import obspy 
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pandas import read_csv
from obspy.clients.fdsn import RoutingClient
from time import sleep, time
import calendar
from datetime import date


# In[ ]:



def __getQuality2(code, day, tsteps, overlap, limit):

    net, sta, loc, cha = code.split(".")

    tdelta = overlap * tsteps

        
    ylimits  = np.zeros(int(1440/tdelta));
    timeline = np.zeros(int(1440/tdelta));
    
    for k in range(0,int(1440/tdelta)):

        time = obspy.UTCDateTime(day) + k*tdelta*60
        tbeg = time - tdelta*60
        tend = time + tdelta*60 

        inv = client.get_stations(network=net, station=sta, level='response');

        st = client.get_waveforms(network=net,
                                  station=sta,
                                  location=loc,
                                  channel=cha,
                                  starttime=tbeg,
                                  endtime=tend
                                 );

        st.remove_response(inventory=inv);

        st.detrend('simple');

        st.filter('bandpass', freqmin=0.001, freqmax=5., corners=4, zerophase=True);

        adata = abs(st[0].data);

        five_percent = np.int(np.ceil(len(adata)*limit));

        adata_sort = np.sort(adata);

        ylimits[k] = adata_sort[five_percent];
        timeline[k] = k*tdelta*60;

    
    return ylimits
    


# In[ ]:


opath = "/home/brotzer/Desktop/"

code = "BW.ALFT..BHZ"

client = RoutingClient('eida-routing')

tsteps = 30  # minutes
overlap = 0.5

limit = 0.95

year = "2019"


# In[ ]:


## _________________________________________

clockstart = time()

net, sta, loc, cha = code.split(".")

skipped = 0



ofile = f"{sta}-{cha}-{year}";
with open(opath+ofile, 'a') as out:
    out.write(f"#COMMENTS: Tsteps:{tsteps} Overlap:{overlap} Limit:{limit}\n")
    out.write(f"datetime medians maxima minima\n")


def __dateIter(year, month):
    for i in range(1, calendar.monthlen(year, month) + 1):
        yield i



for month in tqdm(range(1,13)):

    for day in __dateIter(int(year), month):
        
#     for day in range(1,32):

        try:
            date = obspy.UTCDateTime(f"{year}-{month}-{day}");
            
            ylimits = __getQuality2(code, date, tsteps, overlap, limit);

            with open(opath+ofile, 'a') as out:
                out.write(f"{str(date.date)} {np.median(ylimits)} {np.max(ylimits)} {np.min(ylimits)}\n");
            
        except:
            with open(opath+ofile, 'a') as out:
                out.write(f"{str(date.date)} {np.nan} {np.nan} {np.nan}\n");
            skipped += 1
            
            continue
            
print(f"{skipped} days skipped!")

clockend = time()
print(f"run lasted: {(clockend - clockstart)/60} minutes")


# In[ ]:




plt.show();


# In[ ]:




