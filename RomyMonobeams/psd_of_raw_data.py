#!/bin/python3


from pandas import DataFrame
from obspy import Stream, UTCDateTime
from scipy.signal import welch, hann
from obspy.signal import PPSD
from andbro__querrySeismoData import __querrySeismoData
from andbro__read_sds import __read_sds


## _____________________________________
## configurations 

config = {}

config['seeds'] = ["BW.DROMY..F1V",
                   "BW.DROMY..F4V",
                   "BW.DROMY..FJZ"]

config['path_to_sds'] = "/import/freenas-ffb-01-data/romy_archive/"
# config['path_to_sds'] = "/home/andbro/freenas/romy_archive/"

config['path_to_output'] = "/import/kilauea-data/"
# config['path_to_output'] = "/home/andbro/kilauea-data/"


## all 6 recording [after prism installation]
config['tbeg'] = UTCDateTime("2023-05-09 01:00")
config['tend'] = UTCDateTime("2023-05-09 02:00")

## define window length in seconds for welch psd
config['win_time'] = 1000 ## seconds


## _____________________________________
## load data

st0 = Stream()

for seed in config['seeds']:

    print(f" -> loading {seed}...")
    
    try:
        ## alternative
        st00 = __read_sds(config['path_to_sds'], seed, config['tbeg'], config['tend'], data_format='MSEED')
        
    except:
        print(f" -> failed to load data for {seed}")
        
    st0 += st00

st0 = st0.sort()


for tr in st0:
    tr.data = tr.data*0.59604645e-6 # V / count  [0.59604645ug  from obsidian]
    

## _____________________________________
## compute PSD
    
NN = st0[0].stats.npts
df = st0[0].stats.sampling_rate

nblock = int(df*config['win_time'])
overlap = int(0.5*nblock)
win = hann(nblock)

print(nblock, overlap, NN)

tr = st0.select(channel="FJZ")[0]

ff, fjz_psd = welch(tr.data, fs=tr.stats.sampling_rate, 
                    window=win, noverlap=overlap, nperseg=nblock,
                    scaling="density",
                    return_onesided=True)

tr = st0.select(channel="F1V")[0]

ff, f1v_psd = welch(tr.data, fs=tr.stats.sampling_rate, 
                    window=win, noverlap=overlap, nperseg=nblock,
                    scaling="density",
                    return_onesided=True)

tr = st0.select(channel="F4V")[0]

ff, f2v_psd = welch(tr.data, fs=tr.stats.sampling_rate, 
                    window=win, noverlap=overlap, nperseg=nblock,
                    scaling="density",
                    return_onesided=True)


## _____________________________________
## create and store output
    
out = DataFrame()

out['frequencies'] = ff
out['fjz_psd'] = fjz_psd
out['f1v_psd'] = f1v_psd
out['f2v_psd'] = f2v_psd

out.to_pickle(config['path_to_output']+f"psd_{config['tbeg'].date}.pkl")


## End of File