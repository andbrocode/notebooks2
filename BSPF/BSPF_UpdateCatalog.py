#!/usr/bin/env python
# coding: utf-8

# # Update Catalog for BSPF at PFO

# _____________________________________________________

import os, sys
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from andbro__savefig import __savefig
from obspy.clients.fdsn import Client

import warnings
warnings.filterwarnings('ignore')

# _____________________________________________________
# ## Configurations


config = {}

config['minlatitude'] = 32.444
config['maxlatitude'] = 34.8286
config['minlongitude'] = -118.652
config['maxlongitude'] = -114.941

config['BSPF_lon'] = np.array([-116.455439])
config['BSPF_lat'] = np.array([33.610643])

config['minmagnitude'] = 2.5

config['tbeg'] = obs.UTCDateTime("2022-10-01")
config['tend'] = obs.UTCDateTime().now().date

config['eventfile'] = "BSPF_event_catalog.pkl"

config['outpath'] = "./"


# _____________________________________________________

def __cat_to_df(cat):
    
    from pandas import DataFrame
    
    times = []
    lats = []
    lons = []
    deps = []
    magnitudes = []
    magnitudestype = []
    
    for event in cat:
        if len(event.origins) != 0 and len(event.magnitudes) != 0:
            times.append(event.origins[0].time.datetime)
            lats.append(event.origins[0].latitude)
            lons.append(event.origins[0].longitude)
            deps.append(event.origins[0].depth)
            magnitudes.append(event.magnitudes[0].mag)
            magnitudestype.append(event.magnitudes[0].magnitude_type )
            
    df = DataFrame({'latitude':lats,'longitude':lons,'depth':deps,
                    'magnitude':magnitudes,'type':magnitudestype}, 
                     index = times
                  )
    
    return df


def __makeplot_eventmap(config, data):
    

    import pygmt
    import pandas as pd

    
    # Set the region
    region = [config['minlongitude'], config['maxlongitude'], config['minlatitude'], config['maxlatitude']]

    fig = pygmt.Figure()
    

    # make color pallets
    pygmt.makecpt(
        cmap='etopo1',
        series='-8000/5010/1000',
        continuous=True
    )
    
    
    # define etopo data file
    topo_data = "@earth_relief_03s"
    

    # plot high res topography
    fig.grdimage(
        grid=topo_data,
        region=region,
        projection='M4i',
        shading=True,
        frame=True
    )
        
    
    ## add coastlines
    fig.coast(shorelines=True, frame=False)
#     fig.coast(rivers="1/0.5p,blue") # Rivers

    
    # colorbar colormap
    pygmt.makecpt(cmap="rainbow", series=[data.depth.min(), data.depth.max()])

    
    ## plot data coordinates
    fig.plot(
        x=data.longitude,
        y=data.latitude,
        sizes=0.07*data.magnitude,
        color=data.depth,
        cmap=True,
        style="cc",
        pen="black",
    )

    ## plot PFO
    fig.plot(
        x=config['BSPF_lon'],
        y=config['BSPF_lat'],
        sizes=np.array([0.3]),
        color="white",
        style="t",
        pen="black",
    )
    
    fig.text(
        text="PFO",
        x=config['BSPF_lon'],
        y=config['BSPF_lat'],
        offset=[0.3,0.3],
        font="9p,Helvetica-Bold,black"
    )
    
    ## add depth colorbar
    fig.colorbar(frame='af+l"Depth (km)"')
    
    
#     fig.savefig(config['outpath']+'event_map.png')

    fig.show();
    return fig


def __export_new_events(config, events_old, events):
    
    ## combine new and old catalog
    tmp = pd.concat([__cat_to_df(events_old), __cat_to_df(events)]).reset_index(drop=False)
    ## remove duplicates
    df = tmp.drop_duplicates(subset=['index'], keep=False)
    ## sort and set index
    df = df.sort_index(ascending=False)
    df.set_index('index', inplace=True, drop=True)
    
    ## export new events to pickle file
    print(f" -> Export new events: {config['outpath']}new_events.pkl")
    df.to_pickle(config['outpath']+"new_events.pkl")
    
    del tmp, df  


# _____________________________________________________
# ## MAIN

if __name__ == '__main__':
    
    
    client = Client("USGS")

    print(f" -> Checking events for USGS: {config['tbeg']} - {config['tend']} ...")
    
    try:
        events = client.get_events(minlatitude=config['minlatitude'], maxlatitude=config['maxlatitude'],
                                   minlongitude=config['minlongitude'], maxlongitude=config['maxlongitude'],
                                   starttime=config['tbeg'],
                                   endtime=config['tend'],
                                   minmagnitude=config['minmagnitude'],
                                  )
    except:
        print(" -> Failed to querry Events!")
    
    
    ## check for new events 
    events_old = obs.read_events(config['outpath']+"events.xml")

    if events_old == events:
        print(" -> No new events found!")
        sys.exit()
    else:
        __export_new_events(config, events_old, events)
        events.write(config['outpath']+"events.xml", format="QUAKEML")
    
    ## write events to dataframe
    events_df = __cat_to_df(events)

    ## store dataframe
    try:
        events_df.to_pickle(config['outpath']+config['eventfile'])
    except:
        print(" -> Failed to store catalog!")
    finally:
        print(f" -> Export catalog as: {config['outpath']}{config['eventfile']}")
    
    ## create figure plot      
    try:
        fig = __makeplot_eventmap(config, events_df)
        __savefig(fig, outpath=config['outpath'], outname="event_map", mode="png", dpi=200)   
    except:
        print(" -> Failed to create event map!")
    finally:
        print(f" -> Saved event map: {config['outpath']}event_map.png")
           
              
print(" -> Done")

## END OF FILE