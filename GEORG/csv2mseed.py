#!/usr/bin/env python
# coding: utf-8

# # Convert CSV to Mseed

# In[1]:


from pandas import read_csv


# In[2]:


def __get_stream(arr, seed, starttime, dt=None, sps=None):

    from obspy import Stream, Trace

    net, sta, loc, cha = seed.split(".")

    tr00 = Trace()
    tr00.data = arr

    if dt is not None:
        tr00.stats.delta = dt
    elif sps is not None:
        tr00.stats.sampling_rate = sps

    tr00.stats.starttime = starttime
    tr00.stats.network = net
    tr00.stats.station = sta
    tr00.stats.location = loc
    tr00.stats.channel = cha

    return Stream(tr00)


# In[3]:


def __write_stream_to_sds(st, path_to_sds):

    import os

    ## check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

        if not os.path.exists(path_to_sds+f"{yy}/"):
            os.mkdir(path_to_sds+f"{yy}/")
            print(f"creating: {path_to_sds}{yy}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

        try:
            st_tmp = st.copy()
            st_tmp.select(network=nn, station=ss, location=ll, channel=cc).write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
        except:
            print(f" -> failed to write: {cc}")
        finally:
            print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")


# ### Configurations

# In[4]:


path_to_data = "/import/kilauea-data/sagnac_frequency/bonn/"

filename = "4h_GEORG_Data.csv"

sps = 7000

starttime = "2024-10-01 00:00"

seed_code = "XX.GEORG..BJZ"


# ### read csv data to dataframe

# In[ ]:


df = read_csv(path_to_data+filename, names=["time", "data"])


# ### create stream

# In[ ]:


st = __get_stream(df['data'].values, seed_code, starttime, sps=sps)


# ### write data as mseed to SDS archive

# In[ ]:


__write_stream_to_sds(st, path_to_data)


# In[ ]:




