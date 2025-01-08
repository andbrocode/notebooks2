#!/usr/bin/env python
# coding: utf-8

# # Convert Tiltmeter CSV to MSEED SDS

# In[1]:


from pandas import read_csv, date_range
from obspy import UTCDateTime


# In[2]:


from functions.get_stream import __get_stream
from functions.write_stream_to_sds import __write_stream_to_sds


# In[3]:


def __write_stream_to_sds(st, path_to_sds):

    import os
    from obspy import UTCDateTime, Stream
    from pandas import date_range

    # check if output path exists
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

        dates = date_range(tr.stats.starttime.date, tr.stats.endtime.date)

        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel

        for d in dates:

            trx = tr.copy()

            trx = trx.trim(UTCDateTime(d), UTCDateTime(d)+86400, nearest_sample=False)

            yy, jj = trx.stats.starttime.year, str(trx.stats.starttime.julday).rjust(3,"0")

            try:
                stx = Stream(trx)
                stx.write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
            except:
                print(f" -> failed to write: {cc}")
            finally:
                print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")


# ### Configurations

# In[4]:


path_to_data = "/home/andbro/kilauea-data/GEORG/data/"

filename = "LGMBoxmessung12_16.csv"

seed_code = "XX.TILT.."


# ### read csv data to dataframe

# In[5]:


head = ["time", "north", "east", "X1", "X2", "pressure", "X3", "X4", "X5", "X6"]

df = read_csv(path_to_data+filename, delimiter=";", names=head)


# In[6]:


starttime = UTCDateTime(df.time[0])
print(starttime)

mean_dt = df.time.diff().mean()
print(mean_dt)

sps = 1/mean_dt


# ### create stream

# In[7]:


st = __get_stream(df['north'].values, seed_code+"LAN", starttime, sps=sps)
st += __get_stream(df['east'].values, seed_code+"LAE", starttime, sps=sps)


# In[8]:


print(st)


# ### write data as mseed to SDS archive

# In[9]:


__write_stream_to_sds(st, path_to_data)


# In[23]:


__write_stream_to_sds(st_rasp, path_to_data)


# In[ ]:




