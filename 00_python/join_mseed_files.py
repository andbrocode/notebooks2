#!/usr/bin/env python
# coding: utf-8

# ## Joing Two Mseed Files

# ### Importing


import os, sys
import obspy as obs


# ### Setting Variables


##________________________________________________________

#ipath1 = '/import/kilauea-data/TiltmeterDataBackup/ROMYT_backup/mseed/MAE.D/BW.ROMYT..MAE.D.2022.299'
ipath1 = input("\n Enter data file 1: ")

#ipath2 = '/home/brotzer/Documents/ROMY/tiltmeter/DataTmp/mseed/MAE.D/BW.ROMYT..MAE.D.2022.299'
ipath2 = input("\n Enter data file 2: ")

#opath = '/home/brotzer/Downloads/tmp/'
opath = input("\n Enter data output path: ")

if opath[-1] != "/":
    opath = opath+"/"
    
## load data

try:
    st1 = obs.read(ipath1)
except:
    print(" -> failed to load stream 1")
    
try:
    st2 = obs.read(ipath2)
except:
    print(" -> failed to load stream 2")


## perform basic checks
        
if len(st1) > 1:
    print(f"stream 1 has {len(st1)} traces!")
if len(st2) > 1:
    print(f"stream 2 has {len(st2)} traces!")
    
for tr1, tr2 in zip(st1, st2):
    if tr1.stats.network != tr2.stats.network:
        print(f" -> Error: networks apparently different! {tr1.stats.network} != {tr2.stats.network}")
        sys.exit()
    if tr1.stats.station != tr2.stats.station:
        print(f" -> Error: networks apparently different! {tr1.stats.station} != {tr2.stats.station}")
        sys.exit()
    if tr1.stats.channel != tr2.stats.channel:
        print(f" -> Error: networks apparently different! {tr1.stats.channel} != {tr2.stats.channel}")
        sys.exit()
    
ofile = ipath2.split("/")[-1]
    
print(f"stream 1: {st1[0].stats.npts}")
print(f"stream 2: {st2[0].stats.npts}")


st_out = st1.copy();
st_out += st2.copy();

st_out.merge();

if st_out[0].stats.npts < int(86400*st_out[0].stats.sampling_rate):
    print(f" -> masked stream!")
    print(st_out)
    
    
print(f" -> writing data to: {opath}{ofile}")
st_out.write(opath+ofile,"MSEED")
st1.write(opath+ofile+"_part1","MSEED")
st2.write(opath+ofile+"_part2","MSEED")

print("\n -> DONE")

## END OF FILE
