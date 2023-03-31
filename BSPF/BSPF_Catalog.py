#!/usr/bin/env python
# coding: utf-8

# # Analyse BlueSeis BSPF Events

# With pressure sensor parascientific and new sensor 

# In[1]:


import os 
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from andbro__savefig import __savefig
from obspy.clients.fdsn import Client


# ## Configurations

# In[2]:


config = {}

config['minlatitude'] = 31
config['maxlatitude'] = 35
config['minlongitude'] = -119
config['maxlongitude'] = -114

config['BSPF_lon'] = np.array([-116.455439])
config['BSPF_lat'] = np.array([33.610643])

config['minmagnitude'] = 2.5

config['tbeg'] = obs.UTCDateTime("2022-10-01")
config['tend'] = obs.UTCDateTime("2023-03-20")

config['eventfile'] = "BSPF_event_catalog.pkl"

config['outpath'] = "./"


# In[3]:


def __export_new_events(config, events_old, events):
    
    ## combine new and old catalog
    tmp = pd.concat([__cat_to_df(events_old), __cat_to_df(events)]).reset_index(drop=False)
    ## remove duplicates
    df = tmp.drop_duplicates(subset=['index'], keep=False)
    ## sort and set index
    df = df.sort_index(ascending=False)
    df.set_index('index', inplace=True, drop=True)
    
    ## export new events to pickle file
    print(f" -> export new events: {config['outpath']}new_events.pkl")
    df.to_pickle(config['outpath']+"new_events.pkl")
    
    del tmp, df  


# In[4]:


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


# In[5]:


def __add_distances_and_backazimuth(config, df):

    from obspy.geodetics.base import gps2dist_azimuth

    dist = np.zeros(len(df))
    baz = np.zeros(len(df))

    
    for ii, ev in enumerate(df.index):
        try:
            dist[ii], az, baz[ii] = gps2dist_azimuth(config['BSPF_lat'], config['BSPF_lon'],
                                                     df.latitude[ii], df.longitude[ii],
                                                     a=6378137.0, f=0.0033528106647474805
                                                     )
        except:
            print(" -> failed to compute!")
            
    df['backazimuth'] = baz
    df['distances_km'] = dist/1000

    return df


# ## Get Events

# In[6]:


client = Client("USGS")

## events - all in area and time period
events_all = client.get_events(minlatitude=config['minlatitude'], maxlatitude=config['maxlatitude'],
                               minlongitude=config['minlongitude'], maxlongitude=config['maxlongitude'],
                               starttime=config['tbeg'],
                               endtime=config['tend'],
                               )

# ## events smaller than 2.0
# events_1 = client.get_events(minlatitude=config['minlatitude'], maxlatitude=config['maxlatitude'],
#                              minlongitude=config['minlongitude'], maxlongitude=config['maxlongitude'],
#                              starttime=config['tbeg'],
#                              endtime=config['tend'],
#                              maxmagnitude=2.0,
#                             )

## events between 2.0 and 3.0 within distance 0.5 degrees
events_2 = client.get_events(
                             latitude=config['BSPF_lat'], longitude=config['BSPF_lon'],
                             starttime=config['tbeg'],
                             endtime=config['tend'],
                             minmagnitude=2.0,
                             maxmagnitude=3.0,
                             maxradius=0.5,
                            )

## events between 2.0 and 3.0 within distance 2.0 degrees
events_3 = client.get_events(
                             latitude=config['BSPF_lat'], longitude=config['BSPF_lon'],
                             starttime=config['tbeg'],
                             endtime=config['tend'],
                             minmagnitude=3.0,
                             maxmagnitude=5.0,
                             maxradius=5.0,
                            )

## events larger than 5.0 
events_4 = events_all.filter("magnitude > 5.0")


## join specified event catalogs together
events = events_2 + events_3 + events_4


# events.plot(projection="local");

# events.write(config['outpath']+"events.xml", format="QUAKEML")


# In[ ]:


## convert catalog object to data frame
events_df = __cat_to_df(events)
events_all_df = __cat_to_df(events_all)

## add epicentral distances
__add_distances_and_backazimuth(config, events_df)
__add_distances_and_backazimuth(config, events_all_df)

## write data frame as pickle file
events_df.to_pickle(config['outpath']+config['eventfile'])


# ## Plot Event Timeline

# In[ ]:


events_df


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15,5))

cax = ax.scatter(events_df.index, events_df.distances_km, s=4**events_df.magnitude, c=events_df.magnitude, alpha=0.5, cmap='viridis_r')

ax.scatter(events_all_df.index, events_all_df.distances_km, s=4**events_all_df.magnitude, c='grey', alpha=0.5, zorder=-1)

plt.colorbar(cax, ax=ax, )

ax.set_ylim(bottom=0)
ax.set_ylabel("Distance (km)", fontsize=14)


# ## Plot on Map

# In[ ]:


def __makeplot_eventmap(config, data):
    

    import pygmt
    import pandas as pd

    
    # Set the region
    region = [config['minlongitude'], config['maxlongitude'], config['minlatitude'], config['maxlatitude']]



    fig = pygmt.Figure()
    
#     fig.basemap(region=region, projection="M15c", frame=True)


    # make color pallets
    pygmt.makecpt(
        cmap='etopo1',
        series='-8000/5010/1000',
        continuous=True
    )
    
    
    # define etopo data file
    topo_data = "@earth_relief_03s"
#     topo_data = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
    
    ## adjust land and sea/lakes
#     land = topo_data * pygmt.grdlandmask(region=region, 
#                                          spacing="03s", 
#                                          maskvalues=[0, 1], 
#                                          resolution="f"
#                                         )
#     wet = topo_data * pygmt.grdlandmask(region=region, 
#                                         spacing="03s", 
#                                         maskvalues=[1, "NaN"], 
#                                         resolution="f"
#                                        )

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
    
    
    fig.savefig(config['outpath']+'event_map.png')

    fig.show();
    return fig

# fig = __makeplot_eventmap(config, events_df)


# In[11]:


def __makeplot_eventmap2(config, data1, data2):
    

    import pygmt
    import pandas as pd

    
    # Set the region
    region = [config['minlongitude'], config['maxlongitude'], config['minlatitude'], config['maxlatitude']]



    fig = pygmt.Figure()
    
#     fig.basemap(region=region, projection="M15c", frame=True)


    # make color pallets
    pygmt.makecpt(
        cmap='etopo1',
        series='-8000/5010/1000',
        continuous=True
    )
    
    
    # define etopo data file
    topo_data = "@earth_relief_03s"
#     topo_data = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
    
    ## adjust land and sea/lakes
#     land = topo_data * pygmt.grdlandmask(region=region, 
#                                          spacing="03s", 
#                                          maskvalues=[0, 1], 
#                                          resolution="f"
#                                         )
#     wet = topo_data * pygmt.grdlandmask(region=region, 
#                                         spacing="03s", 
#                                         maskvalues=[1, "NaN"], 
#                                         resolution="f"
#                                        )

    # plot high res topography
    fig.grdimage(
        grid=topo_data,
        region=region,
        projection='M4i',
        shading=True,
        frame=True
    )
        
    
    ## add coastlines
    fig.coast(shorelines=True, frame=False, region=region, projection='M4i')
#     fig.coast(rivers="1/0.5p,blue") # Rivers

    
    # colorbar colormap
#     pygmt.makecpt(cmap="rainbow", series=[data1.depth.min(), data1.depth.max()])
    pygmt.makecpt(cmap="rainbow", series=[1.0, 6.0])

    
    ## plot data coordinates
    fig.plot(
        x=data2.longitude,
        y=data2.latitude,
        sizes=0.07*data2.depth,
        color='grey',
        style="cc",
        pen="black",
    )    
    fig.plot(
        x=data1.longitude,
        y=data1.latitude,
        sizes=0.07*data1.depth,
        color=data1.depth,
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
    
    
    fig.savefig(config['outpath']+'event_map.png')

    fig.show();
    return fig

fig = __makeplot_eventmap2(config, events_df, events_all_df)


# In[ ]:


fig


# In[ ]:





# ## Others 

# In[ ]:


def __makeplot_eventmap_3d(config, data):
    

    import pygmt
    import pandas as pd

    
    # Set the region
    region = [config['minlongitude'], config['maxlongitude'], config['minlatitude'], config['maxlatitude']]



    fig = pygmt.Figure()
#     fig.basemap(region=region, projection="M15c", frame=True)

    # make color pallets
    pygmt.makecpt(
        cmap='etopo1',
        series='-8000/5000/1000',
        continuous=True
    )

    # define etopo data file
    topo_data = "@earth_relief_03s"

    
    fig.grdview(
        grid=topo_data,
        # Set the azimuth to -130 (230) degrees and the elevation to 30 degrees
        perspective=[-130, 30],
        frame=["xaf", "yaf", "WSnE"],
        projection="M15c",
        zsize="1.5c",
        surftype="s",
        cmap="geo",
        plane="1000+ggrey",
        # Set the contour pen thickness to "0.1p"
        contourpen="0.1p",
    )
    fig.colorbar(perspective=True, frame=["a500", "x+lElevation", "y+lm"])
    
    
#     # plot high res topography
#     fig.grdimage(
#         grid=topo_data,
#         region=region,
#         projection='M4i',
#         shading=True,
#         frame=True
#     )
    
#     fig.colorbar(perspective=True, frame=["a500", "x+lElevation", "y+lm"])
    
#     fig.coast(shorelines=True, frame=False)

#     # colorbar colormap
#     pygmt.makecpt(cmap="magma", series=[data.depth.min(), data.depth.max()])

    fig.plot(
        x=data.longitude,
        y=data.latitude,
        sizes=0.05*data.magnitude,
        color=data.depth,
        cmap=True,
        style="cc",
        pen="black",
    )

    fig.colorbar(frame='af+l"Depth (km)"')
    
    
    fig.savefig(config['outpath']+'event_map.png')

    
    fig.show()

# fig = __makeplot_eventmap_3d(config, events_df)


# In[ ]:


fig = plt.subplot()


# In[ ]:


__savefig(fig, outpath=config['outpath'], outname="event_map", mode="png", dpi=200)


# In[ ]:




