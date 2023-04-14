#!/bin/python3


## Imports  _______________________________________

import os, sys
import matplotlib.pyplot as plt
import multiprocessing as mp

from tqdm import tqdm
from pandas import date_range
from obspy.clients.fdsn import Client
from obspy.signal.trigger import coincidence_trigger
from pprint import pprint
from obspy import UTCDateTime, Stream

## Configs _______________________________________

global config 

config = {}

## before 2023-04-01
config['seeds'] = {"rotation":"PY.BSPF..HJ*", "translation":"II.PFO.10.BH*"}
## after 2023-04-01
# config['seeds'] = {"rotation":"PY.BSPF..HJ*", "translation":"PY.PFOIX..HH*"}

config['date1'] = "2022-10-03"
config['date2'] = "2022-10-05"

config['output_path'] = "/home/andbro/kilauea-data/BSPF/trigger/"

config['client'] = Client("IRIS")

config['sampling_rate'] = 40 ## Hz

config['trigger_type'] = 'recstalta'
config['thr_on'] = 4  ## thr_on (float) – threshold for switching single station trigger on
config['thr_off'] = 3.5 ## thr_off (float) – threshold for switching single station trigger off
config['lta'] = int(10*config['sampling_rate'])
config['sta'] = int(0.5*config['sampling_rate'])
config['thr_coincidence_sum'] = 4
config['similarity_thresholds'] = {"BSPF": 0.8, "PFO": 0.7}

config['time_interval'] = 3600 ## in seconds
config['time_overlap'] = 600 ## seconds

## Methods _______________________________________

def __store_as_pickle(obj, filename):

    import pickle
    from os.path import isdir

    ofile = open(filename, 'wb')
    pickle.dump(obj, ofile)

    if isdir(filename):
        print(f"created: {filename}")


def __request_data(seed, client, tbeg, tend):

    net, sta, loc, cha = seed.split(".")

    try:
        inventory = client.get_stations(network=net, 
                                         station=sta,
                                         starttime=tbeg,
                                         endtime=tend,
                                         level="response",
                                         )
    except:
        print(f" -> Failed to load inventory for {seed}!")
        return

    try:
        waveform = client.get_waveforms(network=net,
                                       station=sta,
                                       location=loc,
                                       channel=cha, 
                                       starttime=tbeg-60,
                                       endtime=tend+60,
                                       )

    except:
        print(f" -> Failed to load waveforms for {seed}!")
        return
    
    return waveform, inventory


def __trigger(config, st):
    
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset, plot_trigger
    from obspy.signal.trigger import coincidence_trigger

    st_tmp = st.copy()

    df = st_tmp[0].stats.sampling_rate

#     for ii in range(len(st)):
#         tr = st_tmp[ii]

#         cft = recursive_sta_lta(tr.data, config['sta'], config['lta'])

#         on_off = trigger_onset(cft, config['thr_on'], config['thr_off'])

#         plot_trigger(tr, cft, config['thr_on'], config['thr_off'])


    trig = coincidence_trigger(trigger_type = config['trigger_type'],
                               thr_on = config['thr_on'],
                               thr_off = config['thr_off'],
                               stream = st_tmp, 
                               thr_coincidence_sum = config['thr_coincidence_sum'], 
                               sta = config['sta'], 
                               lta = config['lta'],
                              )

    return trig

def __join_pickle_files(config):
    
    import pickle

    files = os.listdir(config['output_path'])

    trigger_events = []
    for file in files:
        if ".pkl" in file:
            with open(config['output_path']+file, 'rb') as f:
                triggerfile = pickle.load(f)
            if len(triggerfile) != 0:
                for event in triggerfile:
                    trigger_events.append(event)
                    
    return trigger_events



def main(times):
    
    jj, tbeg, tend = times
    
    jj = str(jj).rjust(3,"0")
    
    trigger_list = []


    st_xpfo, inv_xpfo = __request_data(config['seeds']['translation'], config['client'], tbeg, tend)
    st_bspf, inv_bspf = __request_data(config['seeds']['rotation'], config['client'], tbeg, tend)


    ## Processing Data
    st_xpfo_proc = st_xpfo.copy()
    st_xpfo_proc.remove_response(inventory=inv_xpfo)

    st_bspf_proc = st_bspf.copy()
    st_bspf_proc.remove_sensitivity(inventory=inv_bspf)
    st_bspf_proc.resample(config['sampling_rate'])

    ## Join Data
    st = Stream()

    st += st_bspf_proc.copy()
    st += st_xpfo_proc.copy()

    st.detrend("linear")
    st.taper(0.01)
    st.filter('bandpass', freqmin=1.0, freqmax=18, corners=4, zerophase=True)  # optional prefiltering


    trig = __trigger(config, st)

    del st
    
    ## store trigger list
    print(f"-> {config['output_path']}trigger_{date}_{jj}.pkl")
    __store_as_pickle(trigger_list, config['output_path']+f"trigger_{date}_{jj}.pkl")

    
    
## MAIN ___________________________________________

if __name__ == '__main__':
    
    
    pprint(config)
    
    list_of_times, list_of_dates = [], []

    for j, date in enumerate(date_range(config['date1'], config['date2'])):
        
        date = UTCDateTime(UTCDateTime(date).date)
        
        list_of_dates.append(date)

        hh, counter = 0, 0
        while hh <= 86400: 

            tbeg = date - config['time_overlap'] + hh
            tend = date + config['time_overlap'] + hh + config['time_interval']

            list_of_times.append((counter, tbeg, tend))

            hh += config['time_interval']
            counter += 1
        
    
    ## launch parallel processes
    with mp.Pool(processes=10) as pool:
        
        list(tqdm(pool.imap_unordered(main, list_of_times), total=len(list_of_times)))
          
    pool.close()
    pool.join()

    ## join files
    print(" -> joining pickle files to one trigger file ...")
    triggers = __join_pickle_files(config)
    
    print(f" -> writing triggered events to file: \n  -> {config['output_path']}trigger_all.pkl")
    __store_as_pickle(triggers, config['output_path']+f"trigger_all.pkl")
    
    print("\n -> Done")

## End of File