#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[6]:


import sys, os
import numpy as np
import obspy 
import time

from tqdm import tqdm


# ## Setting Variables

# In[5]:


path = '/home/brotzer/Documents/ROMY/ROMY_QualityCheck/'


## interactively 
if len(sys.argv) > 1:
    date_to_analyse = sys.argv[1]
    channel = sys.argv[2]
else:
    date_to_analyse = input("\n Enter the date (e.g. 2019-05-14):  "); print("\n")
    channel = input("\n Enter the date (e.g. 2019-05-14):  "); print("\n")


## manually
# date_to_analyse = '2020-09-03'
# channel = 'BJZ'


tstart = obspy.UTCDateTime(date_to_analyse)

## setting output paths and files automatically
opath = f'{path}Qfiles/{str(date_to_analyse)[:7]}/'

if not os.path.isdir(opath):
    os.mkdir(opath)


# ### Defining Methods

# In[3]:



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
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime, read
    
#     print(" requesting data from archive...")

    net, sta, loc, cha = seed_id.split(".")
    
    ## defining parameters
    year = tstart.year
    doy  = tstart.julday
    tend = tstart + tdelta
    
    if tstart.date != tend.date:
        doy_1 = doy
        doy_2 = tend.julday
        
        tbeg_1 = tstart
        tend_1 = UTCDateTime(tend.date)
        tbeg_2 = UTCDateTime(tend.date)
        tend_2 = tend
        
        
    ## define station depending if raw is set or not
#     sta = sta if raw is None else f"D{sta}"

    ## define local data path
    pathroot = "/import/freenas-ffb-01-data/romy_archive/"

    ## __________________________________________________________________________
    
    try:
        route = obspy.clients.fdsn.RoutingClient("eida-routing")
        inv   = client.get_stations(network=net, station=sta, level="response")
#         print("  --> inventory was obtained"); obtained_inventory=True

    except:
        if raw is not True:
            print("  --> inventory could not be obtained..."); obtained_inventory=False
    
    ## -------------------------- ##
    if raw is None:
        ## recreate file structure of archive
        path = f"{pathroot}{year}/{net}/{sta}/{cha}.D/"
        name = f"{net}.{sta}.{loc}.{cha}.D.{year}.{doy}"
        ## get stream data
        st = obspy.read(path+name, starttime=tstart, endtime=tend)
        
        if obtained_inventory:
            print("  --> trend and response is being removed...")
            return st.detrend("linear").remove_response(inventory=inv) 
        else:
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
                    st += obspy.read(path+name, starttime=t1, endtime=t2);
                return st.merge()
            
            else:
                ## recreate file structure of archive
                path = f"{pathroot}{year}/{net}/DROMY/FJ{cha[2]}.D/"
                name = f"{net}.D{sta}..F{cha[1:3]}.D.{year}.{doy}"
                ## get and return stream data
                st = obspy.read(path+name, starttime=tstart, endtime=tend)
                return st

        
        else:
            print("  --> something went wrong! perhaps with seed_id?")


# In[28]:



def __create_and_write_quality_data(trace, opath, twin, tsubwin, over, count):
    
#     print(" evaluating data...")
    
    samples = trace[0].stats.npts
    deltaT  = trace[0].stats.delta
    steps = int(twin / deltaT) # every minute

    ## define one minute intervalls counted in samples
    intervalls = np.arange(0,samples,steps)

    ## define locations of means
    samples_in_minutes = np.arange(steps/2,samples-steps/2,steps) 
    minute_axis_time = []
    
    
    for m in samples_in_minutes :
        minute_axis_time.append(trace[0].times()[int(m)])

        
    ## allocate memory for variables
    delta_min    = np.zeros(len(intervalls[:-1]))
    delta_max    = np.zeros(len(intervalls[:-1]))
    sigma_minute = np.zeros(len(intervalls[:-1]))
    mean_minute  = np.zeros(len(intervalls[:-1]))
    ifreq        = np.zeros(len(intervalls[:-1]))

    
    
    for k, idx in enumerate(range(len(intervalls[:-1]))):
            
            x = trace[0][intervalls[idx]:intervalls[idx+1]+1]
            
#             print(intervalls[idx]*deltaT/60,"min", "-", intervalls[idx+1]*deltaT/60, "min")

            binsize = tsubwin / deltaT 
            overlap = int(binsize * over)


            bins = np.arange(binsize/2, len(x)-binsize/2+overlap, overlap)

            
            ## allocate variable memory
            dif_tmp   = np.zeros(len(bins))
            avg_tmp   = np.zeros(len(bins))
            std_tmp   = np.zeros(len(bins))
            ifreq_tmp = np.zeros(len(bins))

            for l, xbin in enumerate(bins):
#                 print(f'{(xbin-binsize/2)*deltaT-1}--{(xbin+binsize/2)*deltaT-1}')
               
                wdata = x[int(xbin-binsize/2):int(xbin+binsize/2)]
                
                avg_tmp[l] = (np.mean(wdata))

                std_tmp[l] = (np.std(wdata))

                dif_tmp[l] = (np.abs(np.max(wdata))-np.abs(np.min(wdata)))

                ifreq_tmp[l] = (np.nonzero(np.diff(wdata-np.mean(wdata) > 0))[0].size) /2 /tsubwin
        
            
            ## assign values to vectors
            mean_minute[k] = (np.mean(avg_tmp))
            sigma_minute[k] = (np.std(std_tmp))
            delta_max[k] = (np.max(dif_tmp))
            delta_min[k] = (np.min(dif_tmp))
            ifreq[k] = np.mean(ifreq_tmp)
            
            del avg_tmp, std_tmp, dif_tmp, ifreq_tmp
    

    ## calulcate offset to add for each iteration (time and samples) 
    if count == 0:
        toffset=0; soffset=0;
    else:
        toffset = trace[0].stats.starttime.time.hour*3600+trace[0].stats.starttime.time.minute*60+trace[0].stats.starttime.time.second
        soffset = toffset/trace[0].stats.delta

    ## writing output
    out = open(opath + oname, "a+")
    
    for idx in range(0,len( samples_in_minutes)):
        out.write(f"{soffset+samples_in_minutes[idx]} {toffset+minute_axis_time[idx]} {mean_minute[idx]} {sigma_minute[idx]} {delta_max[idx]} {delta_min[idx]} {ifreq[idx]}\n")
    out.close()


# ### Requesting and Evaluating Data

# In[29]:



start_time = time.time()


## 
tdelta = 15 # minutes

twin = 60 # seconds
tsubwin = 2 # seconds
overlap = 0.5 

## create lists for elapsed times
tr, te = [], []

## define output filename
oname = f"{tstart.date}.Q{channel[-1]}"

## create output file and add header
out = open(opath + oname, "w");
out.write(f"# TimeSteps[sec]: {int(twin)} SubSteps[sec]: {tsubwin} Overlap[sec]: {overlap} \n");
out.write(f"sample_id seconds average sigma delta_max delta_min frequency\n");

out.close();


for period in tqdm(np.arange(0,1440,tdelta)):
    
#     tbeg = obspy.UTCDateTime(tstart+period*60)
    tbeg = obspy.UTCDateTime(tstart+period*60 - tsubwin/2)

    
#     print("\n", tbeg, "-", tbeg+tdelta*60)

    ## requesting data in pieces
    tr1=time.time()
    
    data = __archive_request(f"BW.ROMY.10.{channel}", tbeg, tdelta*60+tsubwin, raw=True)
    
    tr2=time.time()
    tr.append(tr2-tr1)
    
    
    ## evaluation of the data window
    te1=time.time()
    
    __create_and_write_quality_data(data, opath, twin, tsubwin, overlap, period)
    
    te2=time.time()
    te.append(te2-te1)
    
    del data

## feedback on performance
print(f"\n elapsed time overall:  {round((time.time()-start_time)/60,2)} minutes")
print(f" requesting average time: {round(np.mean(tr),2)} seconds")
print(f" evaluating average time: {round(np.mean(te),2)} seconds\n")


# ### Testing

# In[ ]:


# tstart = obspy.UTCDateTime(date_to_analyse)
# toffset = 1 # seconds
# tstart -= toffset


# tdelta = 3


# st = __archive_request(f"BW.ROMY.10.{channel}", tstart, tdelta*60+2*toffset, raw=True)
# trace = st[0]
# print(st[0].stats.starttime, st[0].stats.endtime)
# st.plot();


# In[ ]:


# tsteps = 60 # minutes

# N  = trace.stats.npts
# dt = trace.stats.delta

# nsteps = int(tsteps / dt)
# print(f'nsteps: {nsteps}')


# nbins = np.arange(0, N+nsteps, nsteps)

# mean_minute = []
# for i in range(len(nbins)-1):
# #     print(i, nbins[i]*dt-30, nbins[i+1]*dt-30)

#     data  = trace[nbins[i]:nbins[i+1]]
    
#     tlen = 4
#     binsize = int(tlen / dt)
#     step = int(binsize/2)
    
#     positions = np.arange(0,nsteps,step)

#     binidx = np.arange(0,len(data)-binsize+step, step)
    
    
#     mean = []
#     for k in range(len(binidx)-1):
#         idx = binidx[k]
#         print(k,"", positions[k]*dt,"", f"{idx*dt-1}--{(idx+binsize)*dt-1} sec","",  np.mean(data[idx:idx+binsize+1]))        
    
#         mean.append(np.mean(data[idx:idx+binsize+1]))
        
#     mean_minute.append(np.mean(mean))   
    
# # plt.plot(mean_minute, 'ro')


# In[ ]:


# data = np.arange(-10,111,1)


# binsize = 20
# step = 10

# positions = np.arange(0,len(data)+step,step)
# print(f"length positions: {len(positions)}")

# binidx = np.arange(0,len(data)-binsize, step)

# for k in range(len(binidx)-1):
#     idx = binidx[k]
#     print(positions[k],  np.mean(data[idx:idx+binsize+1]))   


# In[ ]:


# from obspy.clients.fdsn import RoutingClient

# t1 = RoutingClient("eida-routing").get_waveforms(network = "GR",
#                                                  station = "FUR",
#                                                  location = "",
#                                                  channel = "BHZ",
#                                                  starttime = obspy.UTCDateTime("2020-06-01 00:00"),
#                                                  endtime = obspy.UTCDateTime("2020-06-01 00:10"))

# t2 = RoutingClient("eida-routing").get_waveforms(network = "GR",
#                                                  station = "FUR",
#                                                  location = "",
#                                                  channel = "BHZ",
#                                                  starttime = obspy.UTCDateTime("2020-06-01 00:10"),
#                                                  endtime = obspy.UTCDateTime("2020-06-01 00:20"))


# t = t1+t2
# print(len(t1[0]),len(t2[0]),len(t[0]))

# t.merge()
# t1.plot();
# t2.plot();
# t.plot();


# In[ ]:




