#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[6]:


import sys, os
import numpy as np
import obspy 
import time
import io

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool 


# In[7]:


def __check_if_file_exists(tstart, channel, path):
    
    doy = tstart.julday
    
    if doy < 10:
        doy = f"00{doy}"
    elif doy >= 10 and doy < 100:
        doy = f"0{doy}"
    
    if os.path.exists(f"/import/freenas-ffb-01-data/romy_archive/{tstart.year}/BW/DROMY/FJ{channel[-1]}.D/BW.DROMY..FJ{channel[-1]}.D.{tstart.year}.{doy}"):
        print(f" data file: BW.DROMY..FJ{channel[-1]}.D.{tstart.year}.{doy} exists!\n")
    else:
        print(f" data file: BW.DROMY..FJ{channel[-1]}.D.{tstart.year}.{doy} is missing! --> is being skipped!\n")
        
        if tstart.day < 10:
            day = f"0{tstart.day}"
        else:
            day = tstart.day
        if tstart.month < 10:
            month = f"0{tstart.month}"
        else: 
            month = tstart.month
            
        Path(f"{path}{tstart.year}{month}{day}.missing.txt").touch(mode=755)
        sys.exit(1)


# ## Configurations

# In[8]:


path='/home/andbro/Documents/ROMY/QualityCheck/runy/Qfiles/'
path='/home/andbro/Documents/ROMY/data/'

## interactively 
if len(sys.argv) == 6:
    date_to_analyse = sys.argv[1]
    channel = sys.argv[2]
    path = sys.argv[3]
    twin1 = sys.argv[4]
    twin2 = sys.argv[5]
else:
    date_to_analyse = input("\n Enter the date (e.g. 2019-05-14):  "); print("\n")
    channel = input("\n Enter the channel:  "); print("\n")
    twin1, twin2 = [],[]


tstart = obspy.UTCDateTime(date_to_analyse)


## setting output paths and files automatically
opath = f'{path}Qfiles/{str(date_to_analyse)[:7]}/'

## is done in bash script
# if not os.path.isdir(opath):
#     os.mkdir(opath)
#     print("creating folder")
#     os.listdir(opath)


## 
tdelta = 15 # length of requested raw data chunks (in minutes)
overlap = 0.5 # overlap of sub-windows (in percent) 

if twin1 and twin2:
    twin = int(twin1)
    tsubwin = int(twin2)
else:
    twin = 60 # essential sampling interval for averages (in seconds) default value
    tsubwin = 20 # sub-windows to calculate quantites (in seconds) default value

print(f"\n processing using {twin}s and {tsubwin}s windows with {overlap} percent overlap...\n")



## check if input file exists 
# __check_if_file_exists(tstart, channel, path)


# In[9]:


# path='/home/andbro/Documents/ROMY/QualityCheck/runy/Qfiles/'


# ### Defining Methods

# In[15]:



def __archive_request(seed_id, tstart, tdelta, raw=None):
    
    ''' get data of ROMY for one component from archive 

    VARIABLES:
        seed_id:    code of seismic stations (e.g. "BW.ROMY..BJU")
        tstart:	    begin of time period
        tdelta:     temporal length of period

    DEPENDENCIES:
        import obspy

    OUTPUT:

    EXAMPLE:
        >>> __get_stream_data_archive('BW.ROMY.10.BJZ', '2020-07-17 02:50', 3600, raw=False)

    '''
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime, read, Stream
    
#     print(" requesting data from archive...")

    def __extend_digits(doy):
        if doy < 10:
            doy = f"00{doy}"
        elif doy >= 10 and doy < 100:
            doy = f"0{doy}"
        return doy 


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
        
        doy_1 = __extend_digits(doy_1)
        doy_2 = __extend_digits(doy_2)
    else:
        doy = __extend_digits(doy)  
        
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
#                 st = read().clear();
                st = Stream()
                for t1, t2, d in zip((tbeg_1, tbeg_2), (tend_1, tend_2), (doy_1, doy_2)):
                    ## recreate file structure of archive
                    path = f"{pathroot}{year}/{net}/DROMY/FJ{cha[2]}.D/"
                    name = f"{net}.D{sta}..F{cha[1:3]}.D.{year}.{d}"

                    ## get and return stream data
                    try:
                        st += obspy.read(path+name, starttime=t1, endtime=t2);
                    except:
#                         print("failed to read data")

                        import io

                        reclen = 512
                        chunksize = 100000 * reclen # Around 50 MB

                        with io.open(path+name, "rb") as fh:
                            while True:
                                with io.BytesIO() as buf:
                                    c = fh.read(chunksize)
                                    if not c:
                                        break
                                    buf.write(c)
                                    buf.seek(0, 0)
                                    st = obspy.read(buf)

                    return st.merge()
            
            else:
                ## recreate file structure of archive
                path = f"{pathroot}{year}/{net}/DROMY/FJ{cha[2]}.D/"
                name = f"{net}.D{sta}..F{cha[1:3]}.D.{year}.{doy}"
                
                ## get and return stream data
                st = obspy.read(path+name, starttime=tstart, endtime=tend)
#                 st = obspy.read('/home/andbro/Documents/ROMY/data/BW.DROMY..FJZ.D.2021.053', starttime=tstart, endtime=tend)

                return st.merge()

        
        else:
            print("  --> something went wrong! perhaps with seed_id?")


# In[16]:



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
            amax_tmp   = np.zeros(len(bins))
            amin_tmp   = np.zeros(len(bins))
            avg_tmp    = np.zeros(len(bins))
            dif_tmp    = np.zeros(len(bins))
            ifreq_tmp  = np.zeros(len(bins))

            for l, xbin in enumerate(bins):
#                 print(f'{(xbin-binsize/2)*deltaT-1}--{(xbin+binsize/2)*deltaT-1}')
               
                wdata = x[int(xbin-binsize/2):int(xbin+binsize/2)]
                
                avg_tmp[l] = (np.mean(wdata))

#                 std_tmp[l] = (np.std(wdata))

                dif_tmp[l] = (np.abs(np.max(wdata))-np.abs(np.min(wdata)))
                
                amax_tmp[l] = np.max(wdata)
                amin_tmp[l] = np.min(wdata)

                ifreq_tmp[l] = (np.nonzero(np.diff(wdata-np.mean(wdata) > 0))[0].size) /2 /tsubwin
        
            
            ## assign values to vectors
            mean_minute[k] = (np.mean(avg_tmp))
            sigma_minute[k] = (np.max(dif_tmp)-np.min(dif_tmp))
#             sigma_minute[k] = (np.mean(std_tmp))
            delta_max[k] = (np.median(amax_tmp))
            delta_min[k] = (np.median(amin_tmp))
            ifreq[k] = np.median(ifreq_tmp)
            
            del avg_tmp, dif_tmp, amax_tmp, amin_tmp, ifreq_tmp
    

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


# In[17]:



def func(period):
#     tbeg = obspy.UTCDateTime(tstart+period*60)
    tbeg = obspy.UTCDateTime(tstart+period*60 - tsubwin/2)

#     print("\n", tbeg, "-", tbeg+tdelta*60)

    ## requesting data in pieces
    tr1=time.time()
    
    if channel[2] == "Z":
        data = __archive_request(f"BW.ROMY.10.{channel}", tbeg, tdelta*60+tsubwin, raw=True)
    else:
        data = __archive_request(f"BW.ROMY..{channel}", tbeg, tdelta*60+tsubwin, raw=True)
        
    tr2=time.time()
    tr.append(tr2-tr1)
    
    
    ## evaluation of the data window
    te1=time.time()
    
    __create_and_write_quality_data(data, opath, twin, tsubwin, overlap, period)
    
    te2=time.time()
    te.append(te2-te1)
    
    del data
    
    return [tr ,te]


# ### Requesting and Evaluating Data

# In[18]:



if __name__ == '__main__':

    start_time = time.time()



    ## create lists for elapsed times
    tr, te = [], []

    ## define output filename
    oname = f"{tstart.date}.Q{channel[-1]}"

    ## create output file and add header
    out = open(opath + oname, "w");
    out.write(f"# TimeSteps[sec]: {int(twin)} SubSteps[sec]: {tsubwin} Overlap[sec]: {overlap} \n");
    out.write(f"sample_id seconds average avar delta_max delta_min frequency\n");
    out.close();

    
    with Pool(2) as p:
        times = list(tqdm(p.imap(func, np.arange(0, 1440, tdelta)), total=int(1440/tdelta)))
    print(times[0][0])
    tr = times[:][0]
    te = times[:][1]


    ## feedback on performance
    print(f"\n elapsed time overall:  {round((time.time() - start_time)/60,2)} minutes")
    print(f" average requesting time: {round(np.mean(tr),2)} seconds")
    print(f" average evaluating time: {round(np.mean(te),2)} seconds\n")


# In[ ]:


# for period in tqdm(np.arange(0,1440,tdelta)):
    
# #     tbeg = obspy.UTCDateTime(tstart+period*60)
#     tbeg = obspy.UTCDateTime(tstart+period*60 - tsubwin/2)

# #     print("\n", tbeg, "-", tbeg+tdelta*60)

#     ## requesting data in pieces
#     tr1=time.time()
    
#     if channel[2] == "Z":
#         data = __archive_request(f"BW.ROMY.10.{channel}", tbeg, tdelta*60+tsubwin, raw=True)
#     else:
#         data = __archive_request(f"BW.ROMY..{channel}", tbeg, tdelta*60+tsubwin, raw=True)
        
#     tr2=time.time()
#     tr.append(tr2-tr1)
    
    
#     ## evaluation of the data window
#     te1=time.time()
    
#     __create_and_write_quality_data(data, opath, twin, tsubwin, overlap, period)
    
#     te2=time.time()
#     te.append(te2-te1)
    
#     del data

