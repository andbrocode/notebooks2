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
from numpy import sort

## Configs _______________________________________

global config

config = {}

## before 2023-04-01
#config['seeds'] = {"rotation":"PY.BSPF..HJ*", "translation":"II.PFO.10.BH*"}
## after 2023-04-01
config['seeds'] = {"rotation":"PY.BSPF..HJ*", "translation":"PY.PFOIX..HH*"}


## set date range limits
config['date1'] = "2023-04-01"
config['date2'] = "2023-06-15"


# config['output_path'] = "/home/andbro/kilauea-data/BSPF/trigger/"
config['output_path'] = "/import/kilauea-data/BSPF/data/catalogs/"

## specify client to use for data
config['client'] = Client("IRIS")

## specify expected sampling rate
config['sampling_rate'] = 40 ## Hz

## select trigger method
config['trigger_type'] = 'recstalta'

## thr_on (float) – threshold for switching single station trigger on
config['thr_on'] = 3.0  ## 4

## thr_off (float) – threshold for switching single station trigger off
config['thr_off'] = 2.0 ## 3.5

## set time parameters for STA-LTA
config['lta'] = int(10*config['sampling_rate'])
config['sta'] = int(0.5*config['sampling_rate'])

## specify coincidence sum
config['thr_coincidence_sum'] = 4

#config['similarity_thresholds'] = {"BSPF": 0.8, "PFO": 0.7}

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

#    try:
    inventory = client.get_stations(network=net,
                                     station=sta,
                                     starttime=tbeg,
                                     endtime=tend,
                                     level="response",
                                     )
#    except:
#        print(f" -> Failed to load inventory for {seed}!")
#        return None, None

#    try:
    waveform = client.get_waveforms(network=net,
                                   station=sta,
                                   location=loc,
                                   channel=cha,
                                   starttime=tbeg-60,
                                   endtime=tend+60,
                                   )

#    except:
#        print(f" -> Failed to load waveforms for {seed}!")
#        return None, None

    return waveform, inventory


def __trigger(config, st):

    from obspy.signal.trigger import recursive_sta_lta, trigger_onset, plot_trigger
    from obspy.signal.trigger import coincidence_trigger

    st_tmp = st.copy()

    df = st_tmp[0].stats.sampling_rate


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

    files = os.listdir(config['output_path']+"tmp/")

    trigger_events = []
    for file in sort(files):
        if ".pkl" in file:
            with open(config['output_path']+"tmp/"+file, 'rb') as f:
                triggerfile = pickle.load(f)
            if len(triggerfile) != 0:
                for event in triggerfile:
                    trigger_events.append(event)


    return trigger_events



def main(times):

    jj, tbeg, tend = times

    jj = str(jj).rjust(3,"0")


    try:
        st_xpfo, inv_xpfo = __request_data(config['seeds']['translation'], config['client'], tbeg, tend)
        st_bspf, inv_bspf = __request_data(config['seeds']['rotation'], config['client'], tbeg, tend)
    except:
        # print(f" -> failed to load data: {tbeg}-{tend}")
        errors.append(f" -> failed to load data: {tbeg}-{tend}")
        return

    ## Processing Data
    st_xpfo_proc = st_xpfo.copy()
    st_xpfo_proc = st_xpfo_proc.remove_response(inventory=inv_xpfo, output="ACC")

    st_bspf_proc = st_bspf.copy()
    st_bspf_proc = st_bspf_proc.remove_sensitivity(inventory=inv_bspf)
    st_bspf_proc = st_bspf_proc.resample(40.0)

    ## normalize to better compare rotation and velocity
    st_xpfo_proc = st_xpfo_proc.normalize(global_max=False)
    st_bspf_proc = st_bspf_proc.normalize(global_max=False)

    ## Join Data
    st = Stream()

    st += st_bspf_proc.copy()
    st += st_xpfo_proc.copy()

    st.detrend("linear")
    st.taper(0.01)
    st.filter('bandpass', freqmin=1.0, freqmax=18, corners=4, zerophase=True)  # optional prefiltering

    trig = __trigger(config, st)
    #print(trig)

    del st

    ## store trigger list
    #print(f"-> {config['output_path']}trigger_{date}_{jj}.pkl")
    if not os.path.isdir(config['output_path']+f"tmp"):
        os.mkdir(config['output_path']+f"tmp")

    __store_as_pickle(trig, config['output_path']+f"tmp/trigger_{tbeg}_{tend}_{jj}.pkl")



## MAIN ___________________________________________

if __name__ == '__main__':

    pprint(config)

    global errors
    errors = []

    ## generate arguments for final parallel loop
    list_of_times = []

    for j, date in enumerate(date_range(config['date1'], config['date2'])):

        date = UTCDateTime(UTCDateTime(date).date)

        hh, counter = 0, 0
        while hh <= 86400:

            tbeg = date - config['time_overlap'] + hh
            tend = date + config['time_overlap'] + hh + config['time_interval']

            list_of_times.append((counter, tbeg, tend))

            hh += config['time_interval']
            counter += 1



    ## launch parallel processes
    with mp.Pool(processes=5) as pool:

        list(tqdm(pool.imap_unordered(main, list_of_times), total=len(list_of_times)))

    pool.close()
    pool.join()


#    pprint(errors)
    __store_as_pickle(errors, config['output_path']+f"trigger_all_errors.pkl")

    ## join files
    print("\n -> joining pickle files to one trigger file ...")
    triggers = __join_pickle_files(config)

    print(f"\n -> writing triggered events to file: \n  -> {config['output_path']}trigger_all.pkl")
    __store_as_pickle(triggers, config['output_path']+f"trigger_all.pkl")

    print("\n -> Done")


## End of File
