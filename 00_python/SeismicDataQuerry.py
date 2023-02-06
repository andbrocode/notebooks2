#!/usr/bin/env python
# coding: utf-8

# ### Querry Waveforms of SeismicStations using ObsPy

# In[ ]:


##________________________________________________________
''' importing libraries ... '''

import os, subprocess
import matplotlib.pyplot as plt 
import obspy as obs

from obspy.clients.fdsn import Client, RoutingClient
from re import split

#from andbro__save_to import __save_to
#from andbro__get_data import __get_data


# In[ ]:


##________________________________________________________
''' setting variables ... '''

#ipath = '/home/brotzer/'
#ifile = ''

#opath = '/home/brotzer/'
#ofile = ''

## Turkey Quake
# tbeg = obs.UTCDateTime(2020, 10, 30, 11, 50)
# tend = obs.UTCDateTime(2020, 10, 30, 12, 15)

## Alaska Quake
tbeg = obs.UTCDateTime(2020, 10, 19, 20, 58)
tend = obs.UTCDateTime(2020, 10, 19, 22,  0 )


# In[ ]:


def showClients():
    from obspy.clients.fdsn.header import URL_MAPPINGS
    names = []
    for key in sorted(URL_MAPPINGS.keys()):

        names.append("{0:<11} {1}".format(key,  URL_MAPPINGS[key]))
    return names

#showClients()


# In[ ]:


#client = Client("LMU")
#client = Client("IRIS")

#route  = RoutingClient("eida-routing")
#route  = RoutingClient("iris-federator")

#fur = route.get_stations(network="GR", station="FUR")
# fur = route.get_waveforms(network="GR", station="FUR", location=None, channel="BH*", starttime=tbeg, endtime=tbeg+30*60)

# fur.plot();


# In[ ]:


def getMeMyData(seed_id=None, beg=None, end=None, restitute=True):
#def getMeMyData(net=None, sta=None, loc=None, cha=None, beg=None, end=None, restitute=True):
    '''
    Dependencies: 
        from obspy.clients.fdsn import Client, RoutingClient
    '''   
    ## split seed_id string 
    net, sta, loc, cha = seed_id.split(".")
    
    ## check if input variables are as expected
    for arg in [net, sta, loc, cha, beg, end]:
        if arg is None and not 'loc':
            raise NameError(print(f"\nwell, {arg} has not been defined after all!"))
            exit()
            
    ## state which data is requested        
    if loc != None:
        print(f'\nGet data:  {net}.{sta}.{loc}.{cha} ({float(end-beg)/60}) ...')
    else: 
        print(f'\nGet data:  {net}.{sta}..{cha} (traces of {float(end-beg)/60} min duration) ...')
        
    ## attempting to get data from either EIDA or IRIS.
    try: 
        route = RoutingClient("eida-routing")
        print("\nUsing Eida-Routing...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")

            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)

    except: 
        route = RoutingClient("iris-federator")
        print("\nUsing Iris-Federator...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")
        
            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)
    
    ## remove response of instrument specified in inventory
    if restitute:
               
#        pre_filt = [0.001, 0.005, 45, 50]
        
        out="VEL"  # "DISP" "ACC"
        
        st.remove_response(
            inventory=inv, 
#             pre_filt=pre_fit,
            output=out
        )
        
        print(f"\nremoving response ...")
        print(f"\noutput {out}")
        
    print('\nFinished\n_______________\n')
    
    return st, inv

from andbro__querrySeismoData import __querrySeismoData


# In[ ]:


fur, fur_inv = getMeMyData("GR.FUR..BH*", tbeg, tend, restitute=True)

#G, G_inv = getMeMyData("BW.RLAS..BJZ", tbeg, tend, restitute=True)
G, G_inv = __querrySeismoData("BW.RLAS..BJZ", tbeg, tend, restitute=True)


#wet = getMeMyData("GR", "WET", None, "BH*", tbeg, tend)

#fur.filter("bandpass", min_freq=0.001, max_freq=0.5)

G.plot();
#fur_inv.plot()

g=G.copy()

g.filter("bandpass", 
         freqmin=0.01, 
         freqmax=1.0, 
         corners=4,
         zerophase=True,
        )

g.plot();


# In[ ]:


from andbro__plot_CompareStreams import __plotCompareStreams

__plotCompareStreams(g, G)


# In[ ]:


def __querrySeismoData(seed_id=None, beg=None, end=None, restitute=True):
#def getMeMyData(net=None, sta=None, loc=None, cha=None, beg=None, end=None, restitute=True):
    '''
    querry seismic traces and station data
    
    Dependencies: 
        from obspy.clients.fdsn import Client, RoutingClient
        
    example:
    	>>> fur, fur_inv = getMeMyData("GR.FUR..BH*", tbeg, tend, restitute=True)
    ''' 
    
    from obspy.clients.fdsn import Client, RoutingClient
      
    ## split seed_id string 
    net, sta, loc, cha = seed_id.split(".")
    
    ## check if input variables are as expected
    for arg in [net, sta, loc, cha, beg, end]:
        if arg is None and not 'loc':
            raise NameError(print(f"\nwell, {arg} has not been defined after all!"))
            exit()
            
    ## state which data is requested        
    if loc != None:
        print(f'\nGet data:  {net}.{sta}.{loc}.{cha} ({float(end-beg)/60}) ...')
    else: 
        print(f'\nGet data:  {net}.{sta}..{cha} (traces of {float(end-beg)/60} min duration) ...')
        
    ## attempting to get data from either EIDA or IRIS.
    try: 
        route = RoutingClient("eida-routing")
        print("\nUsing Eida-Routing...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")

            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)

    except: 
        route = RoutingClient("iris-federator")
        print("\nUsing Iris-Federator...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")
        
            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)
    
    ## remove response of instrument specified in inventory
    if restitute:
               
#        pre_filt = [0.001, 0.005, 45, 50]
        pre_filt = (0.01, 0.05, 20, 40)

        out="VEL"  # "DISP" "ACC"
        
        st.remove_response(
            inventory=inv, 
            pre_filt=pre_fit,
            output=out,
        )
        
        if pre_fil:
            print(f'\npre-filter is applied: {pre_filt}')

        print(f"\nremoving response ...")
        print(f"\noutput {out}")
        
    print('\nFinished\n_______________\n')
    
    return st, inv


# In[ ]:




