#!/usr/bin/env python
# coding: utf-8

# # Create Dayplots for BSPF

# ## Import Libraries

# In[17]:


import os, sys
import obspy as obs
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy.signal.trigger import coincidence_trigger
from pandas import date_range

from functions.request_data import __request_data


# In[2]:


if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'


# ## Configurations

# In[24]:


config = {}

config['seed'] = "PY.BSPF..HJ*"

config['tbeg'] = obs.UTCDateTime("2022-10-07 00:00:00")
config['tend'] = obs.UTCDateTime("2023-01-01 00:00:00")


config['outpath_figs'] = data_path+"BSPF/figures/dayplots/"
#config['outpath_figs'] = "/export/dump/abrotzer/dayplots/"


config['client'] = Client("IRIS")



# ## Looping

# In[44]:


for date in date_range(config['tbeg'].date, config['tend'].date):
    
    from numpy import nanmean
    
    d1 = obs.UTCDateTime(date)
    d2 = obs.UTCDateTime(date)+86400

    print(d1.date)
    
    st_bspf, inv_bspf = __request_data(config['seed'], d1, d2)
    
    if st_bspf is None:
        print(f" -> no data for {d1.date}")
        continue
    
    st_bspf.trim(d1, d2)
    
#    st_bspf = st_bspf.detrend("demean")

#     if len(st_bspf) > 3:
#         st_bspf = st_bspf.merge(fill_value=0)

    filename = f'{str(d1.date).replace("-","")}.png'
    
    if not os.path.exists(config['outpath_figs']):
        print(f" -> no such directory: {config['outpath_figs']}")


    for tr in st_bspf:
    
        if not os.path.exists(config['outpath_figs']+tr.stats.channel+"/"):
            os.mkdir(config['outpath_figs']+tr.stats.channel+"/")

        tr.plot(
                type="dayplot", 
                interval=60, 
                right_vertical_labels=False,
#                vertical_scaling_range=1e-7, 
                one_tick_per_line=True,
                color=['k', 'r', 'b', 'g'],
                show_y_UTC_label=False,
                handle=False,
                automerge=True,
                show=False,
                outfile=config['outpath_figs']+tr.stats.channel+"/"+tr.stats.channel+"_"+filename,
                );

    del st_bspf
    
#    fig.savefig(config['outpath_figs']+filename, dpi=200, bbox_inches='tight', pad_inches=0.05)


##  End of File
