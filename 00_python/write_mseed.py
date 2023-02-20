#!/usr/bin/env python
# coding: utf-8

# # Write data to .mseed files

# ### Imports & Methods

# In[1]:


import sys
import obspy as obs

from andbro__querrySeismoData import __querrySeismoData


# In[2]:


def __user_interaction():

    conf = {}

    conf['seed'] = input("\nEnter seed name:  ") or None

    if conf['seed'] is None:
        print(" -> No seed id provided!")
        sys.exit()

    conf['repository'] = input("\nEnter repository (archive / local / [george]):  ") or "george"  
    
    if conf['repository'].lower() == 'local':
        conf['datapath'] = input("\nEnter datapath:  ")
    else:
        conf['datapath'] = None


    ## ask for time period
    conf['tbeg'], conf['tend'] = None, None
    while conf['tbeg'] is None:
        conf['tbeg']  = obs.UTCDateTime(input("\nEnter start time (e.g. 2020-06-29 09:52):  ")) or None

    while conf['tend'] is None:
        conf['tend']  = obs.UTCDateTime(input("\nEnter end time (e.g. 2020-06-29 10:00):  ")) or None

    conf['outpath'] = input("\nEnter output path:  ") or None
    
    if conf['outpath'] is None:
        print(" -> No output path id provided!")
        sys.exit()
    if conf['outpath'][-1] != "/":
        conf['outpath'] += "/"
        
    conf['outformat'] = input("\nEnter output file format ([mseed] | ascii):  ") or "mseed"

    if conf['outformat'] == "ascii":
        conf['outformat_type'] = "SLIST"
    else:
        conf['outformat_type'] = conf['outformat']
        
        
        
#     ## ask for filter parameters
#     conf['set_filter'] = input("\nSet Filter (yes/no)?  ") or None

#     if conf['set_filter'].lower() in ["yes", "y"]:
#         conf['filter_type'] = input("\nEnter filter type (bp, lp, hp): ")

#         if conf['filter_type'].lower() in ['bp', 'bandpass']:
#             conf['filter_type'] = 'bandpass'
#             conf['lower_corner_frequency'] = float(input("\nEnter lower corner frequency (in Hz): ")) or None
#             conf['upper_corner_frequency'] = float(input("Enter upper corner frequency (in Hz): ")) or None

#         elif conf['filter_type'].lower() in ['hp', 'highpass']:
#             conf['filter_type'] = 'highpass'
#             conf['lower_corner_frequency'] = float(input("\nEnter lower corner frequency (in Hz): ")) or None
#             conf['upper_corner_frequency'] = None

#         elif conf['filter_type'].lower() in ['lp', 'lowpass']:
#             conf['filter_type'] = 'lowpass'
#             conf['lower_corner_frequency'] = None
#             conf['upper_corner_frequency'] = float(input("\nEnter upper corner frequency (in Hz): ")) or None

    return conf


# ### Configurations

# In[7]:


config = __user_interaction()


# ### Load Data from Repository

# In[8]:


st0, inv = __querrySeismoData(
                            seed_id=config.get("seed"),
                            starttime=config.get("tbeg"),
                            endtime=config.get("tend"),
                            repository=config.get("repository"),
                            path=config['datapath'],
                            restitute=False,
                            detail=True,
                            fill_value=None,
                            )
st0


# ### Write Data to Output

# In[ ]:


for tr in st0:
    sta = config['seed'].split(".")[1]
    cha = config['seed'].split(".")[3]
    tbeg_date = config['tbeg'].date

    config['outname'] = f"{sta}_{cha}_{tbeg_date}.{config['outformat']}"

    tr.write(config['outpath']+config['outname'], format=config['outformat_type'])


# In[ ]:




