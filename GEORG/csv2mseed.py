#!/usr/bin/env python
# coding: utf-8

# # Convert GEORG CSV to Mseed

# In[2]:


from pandas import read_csv


# In[4]:


from functions.get_stream import __get_stream
from functions.write_stream_to_sds import __write_stream_to_sds


# ### Configurations

# In[10]:


# path_to_data = "/home/andbro/kilauea-data/sagnac_frequency/bonn/"
path_to_data = "/import/kilauea-data/GEORG/"

path_to_data_out = "/import/kilauea-data/GEORG/data/"

# filename = "4h_GEORG_Data.csv"
filename = "newab"

sps = 7000

starttime = "2024-12-26 02:00"

seed_code = "XX.GEORG..FJZ"


# ### read csv data to dataframe

# In[8]:


df = read_csv(path_to_data+filename, names=["time", "data"])


# ### create stream

# In[9]:


st = __get_stream(df['data'].values, seed_code, starttime, sps=sps)


# ### write data as mseed to SDS archive

# In[11]:


__write_stream_to_sds(st, path_to_data_out)


# In[ ]:




