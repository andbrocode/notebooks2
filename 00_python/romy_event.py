#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


__author__ = 'AndreasBrotzer'


##__________________________________________________________________

import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy as obs
import matplotlib.gridspec as gridspec



from scipy import signal, fft
from obspy.clients.fdsn import Client, RoutingClient

from andbro__get_data_george import __get_data_george
# from andbro__save_to import __save_to
from andbro__savefig import __savefig
from andbro__get_data_archive import __get_data_archive

from mpl_toolkits.basemap import Basemap

# ### Defining Methods

# In[7]:



def __request_event_parameters():

    ## set of global variables
    global name, delta_T, dt, freq, source, mag

    ## event name
    name = input("*\n* Enter Name of Event:  ")

    ## origin time
    dt0 = input("*\n* Enter DateTime (e.g. 2020-06-29 09:52):  ")

    ## check input
    while len(dt0) <= 10:
        dt0   = input("*\n* --> Try Again (e.g. 2020-06-29 09:52):  ")

    ## magnitude
    mag = input("*\n* Enter Magnitude of Event:  ")

    ## frequencies for filter
    freq0 = input("*\n* Enter Frequency range (e.g. 0.1,1 or 0.1):   ")

    ## duration of the trace
    delta = input("*\n* Enter Trace Length (in minutes):   ")

    ## source for data querry
    source = input("*\n* Enter source [george (default) or archive]: ")


    delta_T = float(delta) * 60
    dt   = obs.UTCDateTime(dt0)

    freq = [float(idx) for idx in freq0.split(',')]
    if len(freq) == 1:
        freq = float(freq[0])
    if len(source) == 0:
        source = 'george'


    print("*\n* ________________________________________ \n*")
    print('*\n* {}   {}   {} \n*'.format(name,mag,dt))


# In[18]:



def __querry_event_of_catalog(dt, delta_T, mag):

    global lon_event, lat_event, event_time, timestring, event_depth

    for catalog in ["IRIS","GFZ","BGR"]:

        try:
            ''' get event from client as catalog object '''


            try:
                event_client = Client(catalog)
                cat = event_client.get_events(starttime=dt, endtime=dt+delta_T, minmagnitude=mag)
            except:
                ## test for slightly lower magnitude (magnitudes vary a little)
                mag = str(round(float(mag)-0.1,1))
                cat = event_client.get_events(starttime=dt, endtime=dt+delta_T, minmagnitude=mag)


            ## get event attributes
            lon_event=cat[0].origins[0].longitude
            lat_event=cat[0].origins[0].latitude

            event_time  = cat[0].origins[0].time
            timestring  = event_time.isoformat()

            event_depth = cat[0].origins[0].depth

            if not len(cat) == 0:
                print('\n -->  Using Event in {} !!\n'.format(catalog))
                return cat
                break

        except:
            cat = obs.core.event.Catalog()
            print('\n !! Event not found in {} !!\n'.format(catalog))

    if len(cat) == 0:
            sys.exit()


# In[9]:



def get_stream_data(path,comp,day,year,samplerate):
    ''' get stream from romy archive daily files for one component '''

    if comp == 'Z':
        name = '{}J{}.D/BW.ROMY.10.{}J{}.D.{}.'.format(str(samplerate),comp,str(samplerate),comp,year)
        st = obs.read(path+name+str(day))

    else:
        name = '{}J{}.D/BW.ROMY..{}J{}.D.{}.'.format(str(samplerate),comp,str(samplerate),comp,year)
        st = obs.read(path+name+str(day))

    return st



# In[10]:



def get_stream_data_event(path, comp, day, year, samplerate, beg=None, end=None):
    ''' get stream from romy archive daily files for one component and one event'''

    if day < 10:
        day = f"00{day}"
    elif day >= 10 and day < 100:
        day = f"0{day}"

    if comp == 'Z':
        name = '{}J{}.D/BW.ROMY.10.{}J{}.D.{}.'.format(str(samplerate),comp,str(samplerate),comp,year)
        st = obs.read(path+name+str(day),starttime=beg,endtime=end)

    else:
        name = '{}J{}.D/BW.ROMY..{}J{}.D.{}.'.format(str(samplerate),comp,str(samplerate),comp,year)
        st = obs.read(path+name+str(day),starttime=beg,endtime=end)

    return st



# In[11]:



def process_traces(st_in,f):
    traces = st_in[0]

    timeline = np.arange(0, traces.stats.npts / traces.stats.sampling_rate, traces.stats.delta)

    ## remove linear trend
    traces.detrend('linear')


    if type(f) is float:
        #print('lowpass applied')
        traces.filter('lowpass',freq=f, corners=2, zerophase=True)
    elif type(f) is list:
        #print('bandpass applied')
        traces.filter('bandpass',freqmin=f[0],freqmax=f[1], corners=2, zerophase=True)

    ## normalize traces
    #traces.data = traces.data/max(abs(traces.data))
    #traces.normalize()

    return timeline, traces


# In[12]:



def draw_map():
    ''' equidistant polar plot centered at ROMY'''

    wide, long = [24e6,20e6]
    thres=1000

    map1 = Basemap(projection='aeqd',lat_0=lat_OBS,lon_0=lon_OBS,fix_aspect=True,width=wide,height=long,                    area_thresh=thres,resolution='l')


    ### read a shapefile
    #map.readshapefile('/home/brotzer/resources/NE_countries/ne_10m_admin_0_countries', name='world', drawbounds=False, color='gray')


    ''' draw second map for grid '''
    map2 = Basemap(projection='aeqd',lat_0=90,lon_0=180,fix_aspect=True,width=wide,height=long,area_thresh=thres)
    map2.drawparallels(range(-90, 90, 30),dashes=[1,0], fontsize=8,color='grey')
    map2.drawmeridians(range(0, 360, 30),labels=[1,1,1,1],dashes=[1,0],fontsize=10,color='grey',labelstyle='+/-')

    ''' draw map items '''
    map1.drawcoastlines(linewidth=0.5,antialiased=1)
    map1.drawmapboundary(fill_color='lightblue')
    map1.fillcontinents(color='lightgrey',lake_color='lightblue')
    map1.drawcountries()

    ''' locate  ROMY ringlaser '''
    x_obs, y_obs = map1(lon_OBS,lat_OBS)
#     obs = map1.plot(x_obs,y_obs,marker='v',color='r',markersize=8,markeredgecolor='k',markeredgewidth=1,alpha=1,zorder=3)
    obs = map1.scatter(x_obs, y_obs, marker='v', color='r', s=100, edgecolor='k', linewidth=1, alpha=1, zorder=3)
    #plt.annotate('ROMY',xy=(map1(lon_OBS,lat_OBS)),xytext=(map1(15,50)),color='r',arrowprops=dict(arrowstyle="->",color='r'))

    ''' locate event '''
    x_ev, y_ev = map1(lon_event,lat_event)
    event = map1.scatter(x_ev, y_ev, s=200, marker='*', color='yellow', edgecolor='k', linewidths=0.5, alpha=1, zorder=3)

#    map1.plot(x_ev,y_ev,marker='*',color='yellow',markersize=12,markeredgecolor='k')

    legend = plt.legend([obs, event],
                        ["ROMY", "Event"],
                        ncol=1,
                        bbox_to_anchor=(0.7, 0.0),
                        loc='lower left',
                        framealpha=0.3)


# In[13]:



def get_distance_great_circle(lon1, lat1, lon2, lat2):
    from math import radians, degrees, sin, cos, asin, acos, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    deg = (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))

    return (6371 * deg , 180 / np.pi * deg)


# In[14]:



def get_baz(lon1, lat1, lon2, lat2):

    wide, long = [24e6,20e6]

    map1 = Basemap(
            projection='aeqd',
            lat_0=lat_OBS,
            lon_0=lon_OBS,
            fix_aspect=True,
            width=wide,
            height=long,
            area_thresh=1000,
            resolution='l'
            )

    x_1, y_1 = map1(lon1,lat1)
    x_2, y_2 = map1(lon2,lat2)

    if x_2 > x_1 and y_2 > y_1:
        return float(90-np.rad2deg(np.arctan(abs(y_1-y_2)/abs(x_1-x_2))))
    elif x_2 > x_1 and y_2 < y_1:
        return float(90+np.rad2deg(np.arctan(abs(y_1-y_2)/abs(x_1-x_2))))
    elif x_2 < x_1 and y_2 < y_1:
        return float(270-np.rad2deg(np.arctan(abs(y_1-y_2)/abs(x_1-x_2))))
    elif x_2 < x_1 and y_2 > y_1:
        return float(270+np.rad2deg(np.arctan(abs(y_1-y_2)/abs(x_1-x_2))))


# ### Interactively request event paramters

# In[15]:


## manually

# event = ['Alaska Peninsula', '7.8',"2020-07-22 06:12", "0.01", "60*60"]

# name, mag, dt, freq0, delta_T = event


##__________________________________________________________________
## check if arguments are passed to .py code or assigned interactively

if sys.argv[0] == "romy_event.py" and len(sys.argv) > 1:
    print("\nusing provided parameters...\n")

    name = sys.argv[1]
    mag  = float(sys.argv[3])
    dt   = obs.UTCDateTime(sys.argv[2])
    freq = float(sys.argv[4])
    delta_T = float(sys.argv[5])*60
    source = sys.argv[6]

    print(name,mag,dt)

else:

    __request_event_parameters()


# ### Setting essential varibales

# In[16]:


##__________________________________________________________________
## some variables are set


## set fixed parameters
day     = dt.julday
year    = dt.year

## output parameters
outname = name+'_'+str(dt.year)+str(dt.month)+str(dt.day)+'_'+str(mag)
outpath = "/home/brotzer/notebooks/figs"

## define path to local data
path="/import/freenas-ffb-01-data/romy_archive/{}/BW/ROMY/".format(str(year))

## coordinates of observatory, loc='upper left'
lon_OBS = 11.277
lat_OBS = 48.165


# ### Main

# In[70]:



def __querry_waveforms(dt, delta_T, source):

    global fur, rlas, u, v, w, z

    fur = obs.core.stream.Stream()
    rlas = obs.core.stream.Stream()
    z = obs.core.stream.Stream()
    u = obs.core.stream.Stream()
    v = obs.core.stream.Stream()
    w = obs.core.stream.Stream()

    def __check(st, trace_dummy, seed_id=None):

        if seed_id is not None:
            net, sta, loc, cha = seed_id.split(".")

        if len(st) == 0:
            print(f"empty stream found for {cha}")
            st.append(trace_dummy)
            st[0].data *= 0.0
            st[0].stats.network = net
            st[0].stats.station = sta
            st[0].stats.location = loc
            st[0].stats.channel = cha
        else:
            print(f"{st[0].stats.station}.{st[0].stats.channel} passed check!")

    route = RoutingClient('eida-routing')


    ## ______________________________________________________________
    ''' get waveforms of FUR '''

    net, sta, loc, cha = "GR.FUR..BHZ".split(".")

    inv = route.get_stations(network=net, station=sta, location=loc, channel=cha, level='response')

    fur = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, starttime=dt, endtime=dt + delta_T, attach_responses=True)
    fur.remove_response(inventory=inv)

    ## ______________________________________________________________
    ''' get waveforms of RLAS '''

    net, sta, loc, cha = "BW.RLAS..BJZ".split(".")

    inv = route.get_stations(network=net, station=sta, location=loc, channel=cha, level='response')

    rlas = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, starttime=dt, endtime=dt + delta_T, attach_responses=True)

    rlas.remove_response(inventory=inv)

    ## ______________________________________________________________
    ''' get waveforms of ROMY '''

    inv = route.get_stations(network="BW", station="ROMY", level='response')

    if source == 'george':

        '''get data of ring components for event from george '''
        z = __get_data_george('Z', sr, dt, delta_T)
        u = __get_data_george('U', sr, dt, delta_T)
        v = __get_data_george('V', sr, dt, delta_T)
        w = __get_data_george('W', sr, dt, delta_T)

    elif source == 'archive':
        print("get archive data ...\n")
        '''get data of ring components for event from archive '''

        z, z_inv = __get_data_archive("BW.ROMY.10.BJZ", dt, dt+delta_T, raw=None)
        u, u_inv = __get_data_archive("BW.ROMY..BJU", dt, dt+delta_T, raw=None)
        v, v_inv = __get_data_archive("BW.ROMY..BJV", dt, dt+delta_T, raw=None)
        w, w_inv = __get_data_archive("BW.ROMY..BJW", dt, dt+delta_T, raw=None)


        trace_dummy = rlas[0].copy()

        __check(fur, trace_dummy)
        __check(rlas, trace_dummy)
        __check(z, trace_dummy, "BW.ROMY..BJU")
        __check(u, trace_dummy, "BW.ROMY..BJU")
        __check(v, trace_dummy, "BW.ROMY..BJU")
        __check(w, trace_dummy, "BW.ROMY..BJU")


    z.remove_response(inventory=inv)
    u.remove_response(inventory=inv)
    v.remove_response(inventory=inv)
    w.remove_response(inventory=inv)


    if fur and rlas and u and v and w and z:
        print('\n --> retrieved waveforms for: FUR, RLAS and ROMY\n')


# In[66]:


''' querry event data from online catalog'''
cat = 0
cat = __querry_event_of_catalog(dt, delta_T, mag)


if cat == 0:
    print("aborted")
    sys.exit(1)

''' querry waveform data either online or locally '''
__querry_waveforms(dt, delta_T, source)



'''detrend and filter data of ROMY'''
#freq=0.1
#freq=[0.01,0.2]

z_timeline, z_tr = process_traces(z, freq)
u_timeline, u_tr = process_traces(u, freq)
v_timeline, v_tr = process_traces(v, freq)
w_timeline, w_tr = process_traces(w, freq)


fur_timeline, fur_tr = process_traces(fur, freq)

#rlas_timeline, rlas_tr = np.arange(0, rlas[0].stats.npts * rlas[0].stats.delta, rlas[0].stats.delta), rlas[0].data
rlas_timeline, rlas_tr = process_traces(rlas, freq)

'''calculate the distance to the event along a great circle'''
dist_km, dist_deg = get_distance_great_circle(lon_OBS,lat_OBS,lon_event,lat_event)


'''calculate BAZ ROMY and EVENT location mapped on the equidistant plot'''
baz = get_baz(lon_OBS,lat_OBS,lon_event,lat_event)




# ### Plotting

# In[69]:


def __makeplot():

    rlas , fur = True , False

    fig = plt.figure(figsize=(14,6))
#    gridspec_kw={'hspace': 2,'wspace':2}

#     fig.subplots_adjust(wspace=0.7, hspace=0.9)

    gs = gridspec.GridSpec(5, 5, wspace=0.9, hspace=.4)

    lw = 0.8


    ## -------------------------------------------------------------
    ##
    ax1 = plt.subplot(gs[:1,:2])
    ax1.axis('off')
    ax1.text(0,0.05,'Event:   {}  (Mw={})\nTime:    {}  {} UTC\nDepth:  {} km\n\n'.format(name,mag,timestring[0:10],timestring[11:19],event_depth/1000),fontsize=11)

    ax1.text(0,0,'Dist:     {:.0f} km   ({:.1f} °)\nBAZ:     {:.1f} °'.format(dist_km,dist_deg,baz),fontsize=11)


    ## -------------------------------------------------------------
    ##
    ax2 = plt.subplot(gs[1:, 0:2])
    draw_map()


    ## -------------------------------------------------------------
    ##
    ax3 = plt.subplot(gs[0,2:])
    #ax3.plot(np.arange(0,fur[0].stats.npts*fur[0].stats.delta,fur[0].stats.delta),fur[0].data/max(abs(fur[0].data)),'k',label='FUR')
    if fur:
        ax3.plot(fur_timeline/60,fur_tr,'k',label='FUR')
    elif rlas:
        ax3.plot(rlas_timeline/60,rlas_tr,'k',label='RLAS', linewidth=lw)

    ax3.plot((event_time-dt)/60,0,marker='*',color='yellow',markeredgecolor='k',markersize=15,alpha=1,markeredgewidth=1)
    ax3.legend(loc='upper right',bbox_to_anchor=(1.05, 1.15), framealpha=1, fontsize=8)
    ax3.set_xticklabels([])
#     ax3.grid(True,zorder=0,color='grey',linestyle='--')
#    ax3.set_ylim(-1,1)
    ax3.set_ylabel(r"$\Omega$ (rad/s)")

    if type(freq) is list:
        ax3.set_title('bandpass filtered: {} - {} Hz'.format(freq[0],freq[1]))
    else:
        ax3.set_title('lowpass filtered: {} Hz'.format(freq) )

    ## -------------------------------------------------------------
    ##
    ax4 = plt.subplot(gs[1,2:])
    ax4.plot(z_timeline/60,z_tr.data,'k',label='ROMY Z', linewidth=lw)
    ax4.plot((event_time-dt)/60,0,marker='*',color='yellow',markeredgecolor='k',markersize=15,alpha=1,markeredgewidth=1)
    ax4.legend(loc='upper right',bbox_to_anchor=(1.05, 1.15), framealpha=1, fontsize=8)
    ax4.set_xticklabels([])
    ax4.set_ylabel(r"$\Omega$ (rad/s)")
    ax4.set_ylim([-max(abs(z_tr.data)), max(abs(z_tr.data))])
#     ax4.grid(True,zorder=0,color='grey',linestyle='--')


    ## -------------------------------------------------------------
    ##
    ax5 = plt.subplot(gs[2,2:])
    ax5.plot(u_timeline/60,u_tr.data,'k',label='ROMY U', linewidth=lw)
    ax5.plot((event_time-dt)/60,0,marker='*',color='yellow',markeredgecolor='k',markersize=15,alpha=1,markeredgewidth=1)
    ax5.legend(loc='upper right',bbox_to_anchor=(1.05, 1.15), framealpha=1, fontsize=8)
    ax5.set_xticklabels([])
    ax5.set_ylabel(r"$\Omega$ (rad/s)")
    ax5.set_ylim([-max(abs(u_tr.data)), max(abs(u_tr.data))])
#     ax5.grid(True,zorder=0,color='grey',linestyle='--')


    ## -------------------------------------------------------------
    ##
    ax6 = plt.subplot(gs[3,2:])
    ax6.plot(v_timeline/60,v_tr.data,'k',label='ROMY V', linewidth=lw)
    ax6.plot((event_time-dt)/60,0,marker='*',color='yellow',markeredgecolor='k',markersize=15,alpha=1,markeredgewidth=1)
    ax6.legend(loc='upper right',bbox_to_anchor=(1.05, 1.15), framealpha=1, fontsize=8)
    ax6.set_xticklabels([])
    ax6.set_ylabel(r"$\Omega$ (rad/s)")
    ax6.set_ylim([-max(abs(v_tr.data)), max(abs(v_tr.data))])

#     ax6.grid(True,zorder=0,color='grey',linestyle='--')

    ## -------------------------------------------------------------
    ##
    ax7 = plt.subplot(gs[4,2:])
    ax7.plot(w_timeline/60,w_tr.data,'k',label='ROMY W', linewidth=lw)
    ax7.plot((event_time-dt)/60,0,marker='*',color='yellow',markeredgecolor='k',markersize=15,alpha=1,markeredgewidth=1)
    ax7.legend(loc='upper right',bbox_to_anchor=(1.05, 1.15), framealpha=1, fontsize=8)
    ax7.set_ylim([-max(abs(w_tr.data)), max(abs(w_tr.data))])
    ax7.set_ylabel(r"$\Omega$ (rad/s)")
    ax7.set_xlabel('Time from {} {} UTC (min)'.format(dt.isoformat()[0:10],dt.isoformat()[11:19]))
#     ax7.grid(True,zorder=0,color='grey',linestyle='--')


    plt.show();
    return fig

fig = __makeplot()


# ### Saving

# __save_to(fig)

## _______________________________________________________________
## check if figure should be saved

save = input('*\n* SAVE FIGURE (y3.5/n) ??:  ') or "n"

if save == 'y':

    default_opath = '/home/brotzer/notebooks/figs/romy_events/'
    opath = input(f'\n* Enter path (default: {default_opath}): ') or default_opath

    default_oname = 'ROMY_{}{}{}_{}'.format(str(dt.year),str(dt.month),str(dt.day),name.replace(', ','_'))
    oname = input('\n* Enter name (default: ROMY_<date>_<name>): ') or default_oname

    # outname = outname.replace(' ','_')

    form = input('\n* Enter format (default: png): ') or 'png'

    __savefig(fig, outpath=opath, outname=oname, mode='png')

else:
    print('*\n* Figure discared!\n*')



## End of File
