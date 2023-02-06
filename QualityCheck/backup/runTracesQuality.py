#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
import obspy 
import numpy as np
import matplotlib.pyplot as plt
import calendar

from tqdm import tqdm
from pandas import read_csv
from obspy.clients.fdsn import RoutingClient
from time import sleep, time
from datetime import date
from configparser import ConfigParser


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

        if sta == 'ROMY':
            inv = client.get_stations(network=net, station=sta, level='response');

            st = __archive_request(code, tbeg, (tend-tbeg), raw=None)
#             st = client.get_waveforms(base_url="http://george",
#                                       network=net,
#                                       station=sta,
#                                       location=loc,
#                                       channel=cha,
#                                       starttime=tbeg,
#                                       endtime=tend
#                                      );            
            
        else:
        
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



def __archive_request(seed_id, tstart, tdelta, raw=None):
    
    ''' get data of ROMY for one component from archive 

    VARIABLES:
        seed_id:    code of seismic stations (e.g. "BW.ROMY..BJU")
        tstart:	    begin of time period
        tdelta:     temporal length of period

    DEPENDENCIES:
        import obspy

    EXAMPLE:
        >>> __get_stream_data_archive('BW.ROMY.10.BJZ', '2020-07-17 02:50', 3600, raw=False)

    '''
    from obspy.clients.fdsn import Client, RoutingClient
    from obspy import UTCDateTime, read
    
    def __getDOY(time):
        doy  = time.julday
        if doy < 10: 
            doy = f"00{doy}"
        elif doy < 100:
            doy = f"0{doy}" 
        return doy

#     print(" requesting data from archive...")

    net, sta, loc, cha = seed_id.split(".")
    
    tstart = UTCDateTime(tstart)
    
    ## defining parameters
    doy  = __getDOY(tstart)

    year = tstart.year
    tend = tstart + tdelta
    
    if tstart.date != tend.date:
        doy_1 = __getDOY(tstart)
        doy_2 = __getDOY(tend)
            
        tbeg_1 = tstart
        tend_1 = UTCDateTime(tend.date)
        tbeg_2 = UTCDateTime(tend.date)
        tend_2 = tend
    

    
    ## define local data path
    pathroot = "/import/freenas-ffb-01-data/romy_archive/"

    ## __________________________________________________________________________
    
    try:
        route = RoutingClient("eida-routing")
        inv   = client.get_stations(network=net, station=sta, level="response")
        obtained_inventory=True
#         print("  --> inventory could be obtained...")
        
    except:
        if raw is not True:
            print("  --> inventory could not be obtained..."); 
            obtained_inventory=False
    
    
    
    ## -------------------------- ##
    if raw is None:
        if tstart.date != tend.date:
            st = read().clear();
            for t1, t2, d in zip((tbeg_1, tbeg_2), (tend_1, tend_2), (doy_1, doy_2)):
                ## recreate file structure of archive
                path = f"{pathroot}{year}/{net}/{sta}/{cha}.D/"
                name = f"{net}.{sta}.{loc}.{cha}.D.{year}.{d}"
                ## get stream data
                st += read(path+name, starttime=t1, endtime=t2)
            st.merge()
        else:
            ## recreate file structure of archive
            path = f"{pathroot}{year}/{net}/{sta}/{cha}.D/"
            name = f"{net}.{sta}.{loc}.{cha}.D.{year}.{doy}"
            ## get and return stream data
            st = read(path+name, starttime=tstart, endtime=tend)
        
#         if obtained_inventory:
#             print("  --> trend and response is being removed...")
#             try: 
#                 st.detrend('simple')
#                 st.remove_response(inventory=inv) 
#             except:
#                 st.remove_response(inventory=inv) 

#             return st
#         else:
#             return st   
        return st
        
    ## -------------------------- ##
    elif raw is True: 
        if sta == "ROMY":
            if tstart.date != tend.date:
                st = read().clear();
                for t1, t2, d in zip((tbeg_1, tbeg_2), (tend_1, tend_2), (doy_1, doy_2)):
                    ## recreate file structure of archive
                    path = f"{pathroot}{year}/{net}/DROMY/FJ{cha[2]}.D/"
                    name = f"{net}.D{sta}..F{cha[1:3]}.D.{year}.{d}"

                    ## get and return stream data
                    st += read(path+name, starttime=t1, endtime=t2);
                return st.merge()
            
            else:
                ## recreate file structure of archive
                path = f"{pathroot}{year}/{net}/DROMY/FJ{cha[2]}.D/"
                name = f"{net}.D{sta}..F{cha[1:3]}.D.{year}.{doy}"
                ## get and return stream data
                st = read(path+name, starttime=tstart, endtime=tend)
                return st

        
        else:
            print("  --> something went wrong! perhaps with seed_id?")


# In[ ]:



config = ConfigParser();
config.read("config.ini");


## Variables
opath = config["VAR"]["opath"]

code = config["VAR"]["code"]

year = config["VAR"]["year"]

## Settings

tsteps = int(config["SETTINGS"]["tsteps"])

overlap = float(config["SETTINGS"]["overlap"])

limit = float(config["SETTINGS"]["limit"])


client = RoutingClient('eida-routing')


# In[ ]:


# opath = "/home/brotzer/Desktop/"

# code = "BW.ROMY.10.BJZ"

# client = RoutingClient('eida-routing')

# tsteps = 30  # minutes
# overlap = 0.5 # 0.5 = 50%

# limit = 0.95

# year = "2019"



# In[ ]:


# print(type(opath), opath)
# print(type(code), code)
# print(type(year), year)
# print(type(client), client)
# print(type(tsteps), tsteps)
# print(type(overlap), overlap)
# print(type(limit), limit)


# In[ ]:


## _________________________________________

clockstart = time()

net, sta, loc, cha = code.split(".")

skipped = 0


ofile = f"{sta}-{cha}-{year}";
with open(opath+ofile, 'w') as out:
    out.write(f"#COMMENTS: Tsteps:{tsteps} Overlap:{overlap} Limit:{limit}\n");
    out.write(f"datetime medians maxima minima\n");


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
#             with open(opath+ofile, 'a') as out:
#                 out.write(f"{str(date.date)} {np.nan} {np.nan} {np.nan}\n");
            skipped += 1
            
            continue
            
print(f"{skipped} days skipped!")

clockend = time()
print(f"run lasted: {round((clockend - clockstart)/60,2)} minutes")


# In[ ]:




