#!/usr/bin/env python
# coding: utf-8

# ## Obs Station Metadata

# In[1]:


# %matplotlib nbagg

import re
from obspy import UTCDateTime, read_inventory
from obspy.clients.nrl import NRL
from obspy.io.xseed import Parser
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.clients.fdsn import Client as FDSNClient


# In[2]:


# could be replaced with a local download of NRL
nrl = NRL()


# ### Example to find out correct keys for given sensor in NRL

# In[3]:


print(nrl.sensors)

manufacturer = input("\nChoose manufacturer: ");print("\n_______________________________")


# In[4]:


print(nrl.sensors[manufacturer])

sensor = input("\nChoose sensor: ");print("\n_______________________________")


# In[5]:


print(nrl.sensors[manufacturer][sensor])

sensitivity = input("\nChoose sensitivity: ");print("\n_______________________________")


# In[6]:


print(nrl.sensors[manufacturer][sensor][sensitivity])


generation = input("\nChoose generation: ");print("\n_______________________________")


# In[7]:


nrl.sensors[manufacturer][sensor][sensitivity][generation]


# In[ ]:


print(nrl.dataloggers)

datalogger = input("\nChoose datalogger: ");print("\n_______________________________")


# In[17]:


print(nrl.dataloggers[datalogger])

model = input("\nChoose datalogger model: ");print("\n_______________________________")


# In[ ]:


print(nrl.dataloggers[datalogger][model])

gain = input("\nChoose datalogger gain: ");print("\n_______________________________")


# In[34]:


print(nrl.dataloggers[datalogger][model][gain])

sampling_rate = input("\nChoose datalogger sampling rate: ");print("\n_______________________________")


# In[ ]:


print(nrl.dataloggers[datalogger][model][gain][sampling_rate])


# In[ ]:


[datalogger, model, gain, sampling_rate]


# In[44]:


response = nrl.get_response(
    datalogger_keys=[datalogger, model, gain, sampling_rate],
    sensor_keys=[manufacturer, sensor, sensitivity, generation]
    )


# In[46]:


response.plot(0.001);


# ### Prepare Writing XML-File

# In[54]:


net = input("\nEnter network: ");print("\n_______________________________")

sta = input("\nEnter station name: ");print("\n_______________________________")

site_name = input("\nEnter site name: ");print("\n_______________________________")

serial_number = input("\nEnter serial number: ");print("\n_______________________________")

outpath = input("\nEnter path of output file: ");print("\n_______________________________")

location = input("\nSpecify location (y/n)? ")

if location == "y" or location == "yes":
    lat = input("Enter latitude: ")
    lon = input("Enter longitude: ")
    ele = input("Enter elevation: ")
    
else:
    lat, lon, ele = 0.0, 0.0, 0.0

outfile = f"{serial_number}_{sta}.xml"


# In[55]:


channel1 = Channel(code='HHZ', 
                   location_code='', 
                   latitude=lat, 
                   longitude=lon,
                   elevation=ele, 
                   depth=0,
#                    azimuth=0,
#                    dip=-90,
                   sample_rate=sampling_rate,
                   response=response,
                  )

channel2 = Channel(code='HHN', 
                   location_code='', 
                   latitude=lat, 
                   longitude=lon,
                   elevation=ele, 
                   depth=0,
#                    azimuth=0,
#                    dip=0,
                   sample_rate=sampling_rate,
                   response=response,
                  )

channel3 = Channel(code='HHE', 
                   location_code='', 
                   latitude=lat, 
                   longitude=lon,
                   elevation=ele, 
                   depth=0,
#                    azimuth=90,
#                    dip=0,
                   sample_rate=sampling_rate,
                   response=response,
                  )


# In[81]:


site = Site(name=site_name)


station = Station(code=sta, 
                  latitude=lat, 
                  longitude=lon,
                  elevation=ele,
                  channels=[channel1,channel2,channel3],
                  site=site,
                 )

network = Network(code=net,
                  stations=[station],
                 )


inv = Inventory(networks=[network], 
                source='LMU',
               )


if outpath[-1] == "/":
    outpath = outpath[:-1]

inv.write(f"{outpath}/{outfile}", 
          format='STATIONXML',
         )


# In[64]:


try:
    read_inventory(f"{outpath}/{outfile}")
    print("\n DONE")
except:
    print("\n Something went wrong! File: {outpath}/{outfile} could not be loaded!")


# In[82]:


import sys
sys.exit(1)


# In[50]:


# compare with existing info in Jane
client = FDSNClient('LMU')
inv2 = client.get_stations(station='BE1', channel='HHZ', level='response')
response2 = inv2[0][0][0].response
response2.plot(0.001)
print(response2)


# ### Station Metadata Definitions

# In[79]:


# could be stored in some ASCII file instead for convenience

# station line:
# 1. station code
# 2. latitude
# 3. longitude
# 4. elevation
# 5. site description
#  ... could be extended, see StationXML. e.g. site, vault, geology,
#      contact person, description, comments etc.

# channel line:  (if lon/lat/elevation changes: new station epoch!)
# 1. location code (e.g. '', '00')
# 2. stream label (e.g. 'HH', 'EH')
# 3. components
# 4. azimuths (e.g. '0,0,90' or 'None' for perfect ZNE orientation) 
# 5. start time
# 6. end time
# 7. depth
# 8. sampling rate
# 9. response lookup key
#  ... could be extended, also needs means to specify orientation e.g. 
data = """BW
 BE1 48.0 12.0 500.0 WbH Monatshausen, Bernried, Bavaria, Germany
  None HH ZNE None 2010-01-01 None 0.0 200.0 RT130-1-200_TRC120s
"""
response_lookup = """RT130-1-200_TRC120s NRL ['REF TEK', 'RT 130 & 130-SMA', '1', '200'] ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
"""
data


# In[80]:


data = f"{net}\n {sta} {lat} {lon} {ele} {site_name}\n {None} {sta[:2]} {'ZNE'} {None} {'2021-02-10'} {None} {0.0} {sampling_rate} {'RT130-1-200_TRC120s'}\n"
data

response_lookup = """RT130-1-200_TRC120s NRL ['REF TEK', 'RT 130 & 130-SMA', '1', '200'] ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
"""


# In[70]:



def parse_response(line):
    key, type_, data = line.split(None, 2)
    if type_ == 'NRL':
        match = re.search(r'\[([^\]]*)\] \[([^\]]*)\]', data)
        args = []
        for group in match.groups():
            keys = re.findall(r"'([^']*)'", group)
            args.append(keys)
        response = nrl.get_response(*args)
    else:
        raise NotImplementedError()
    return key, response


# In[71]:



def add_station(network, line):
    
    parts = line.split(None, 4)
    code, lat, lon, elevation, site_description = parts
    lat = float(lat)
    lon = float(lon)
    elevation = float(elevation)
    site = Site(name=site_description)
    
    sta = Station(code=code, latitude=lat, longitude=lon, elevation=elevation,
                  channels=[], site=site)
    network.stations.append(sta)

    return sta


# In[75]:



def add_channels(station, line):
    parts = line.split()
    loc, stream_label, components, azims, start, end, depth, sampling_rate, resp_key = parts
    if loc == 'None':
        loc = ''
    if azims == 'None':
        azi1 = 0
        azi2 = 0
        azi3 = 90
    start = UTCDateTime(start)
    if end == 'None':
        end = None
    else:
        end = UTCDateTime(end)
    depth = float(depth)
    sampling_rate = float(sampling_rate)
    response = responses[resp_key]
    for component, azi, dip in zip(components, (azi1, azi2, azi3), (-90, 0, 0)):
        cha = Channel(
            code=stream_label + component, location_code=loc, start_date=start, end_date=end,
            latitude=station.latitude, longitude=station.longitude, elevation=station.elevation,
            depth=depth, azimuth=azi, dip=dip, sample_rate=sampling_rate, response=response)
        station.channels.append(cha)
        # update station epoch times
        if station.start_date is None:
            station.start_date = cha.start_date
        else:
            station.start_date = min(station.start_date, cha.start_date)
        if station.end_date is None:
            station.end_date = cha.end_date
        else:
            station.end_date = max(station.end_date, cha.end_date)


# In[78]:


# set up all responses
responses = {}

for line in response_lookup.splitlines():
    key, response = parse_response(line)
    responses[key] = response


# In[ ]:


# assemble all epochs

inventory = Inventory(networks=[], source='EDB')

lines = data.splitlines()
network = None
station = None

while lines:
    line = lines.pop(0)
    if line.startswith('  '):
        add_channels(station, line)
    elif line.startswith(' '):
        station = add_station(network, line)
    else:
        network = Network(code=line, stations=[])
        inventory.networks.append(network)


# In[ ]:


inventory.write('/tmp/example_stationxml_inventory.xml', format='STATIONXML')


# In[ ]:




