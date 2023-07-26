#!/bin/python3


## Imports  _______________________________________

import os, sys
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging

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

config['seeds'] = {"rotation":"PY.BSPF..HJ*", "translation1":"II.PFO.10.BH*", "translation2":"PY.PFOIX..HH*"}

## set date range limits
config['date1'] = "2022-10-01"
config['date2'] = "2023-06-15"

## path to write output
#config['output_path'] = "/import/kilauea-data/BSPF/data/catalogs/"
config['output_path'] = "/export/dump/abrotzer/"

## name of output file for data
config['output_filename'] = f"triggered_{config['date1']}_{config['date2']}.pkl"

config['output_logfile']= f"triggered_{config['date1']}_{config['date2']}.log"

## specify client to use for data
config['client'] = Client("IRIS")

## specify expected sampling rate
# config['sampling_rate'] = 40 ## Hz

## select trigger method
config['trigger_type'] = 'recstalta'

## thr_on (float) – threshold for switching single station trigger on
config['thr_on'] = 2.0  ## 4

## thr_off (float) – threshold for switching single station trigger off
config['thr_off'] = 1.2 ## 3.5

## set time parameters for STA-LTA
config['lta'] = 10.0  ## seconds
config['sta'] = 1.5  ## seconds

## specify coincidence sum
config['thr_coincidence_sum'] = 4

#config['similarity_thresholds'] = {"BSPF": 0.8, "PFO": 0.7}

config['time_interval'] = 86400 ## 3600 ## in seconds
config['time_overlap'] = 600 ## seconds


## Methods _______________________________________

def __store_as_pickle(obj, filename):

    import pickle
    from os.path import isdir

    ofile = open(filename, 'wb')
    pickle.dump(obj, ofile)

    if isdir(filename):
#        print(f"created: {filename}")
        logging.info(f"created: {filename}")

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
#        print(f" -> Failed to load inventory for {seed}!")
        logging.error(f" -> Failed to load inventory for {seed}!")
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
#        print(f" -> Failed to load waveforms for {seed}!")
        logging.error(f" -> Failed to load waveforms for {seed}!")
        return

    try:
        inventory = client.get_stations(network=net,
                                         station=sta,
                                         starttime=tbeg,
                                         endtime=tend,
                                         level="response",
                                         )
    except:
#        print(f" -> Failed to load inventory for {seed}!")
        logging.error(f" -> Failed to load inventory for {seed}!")
        return None, None

    try:
        waveform = client.get_waveforms(network=net,
                                       station=sta,
                                       location=loc,
                                       channel=cha,
                                       starttime=tbeg-60,
                                       endtime=tend+60,
                                       )

    except:
#        print(f" -> Failed to load waveforms for {seed}!")
        logging.error(f" -> Failed to load waveforms for {seed}!")
        return None, None

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

    jj = str(jj).rjust(4,"0")


    try:
        ## load translation data
        if tbeg < UTCDateTime("2023-04-01"):
            st_xpfo, inv_xpfo = __request_data(config['seeds']['translation1'], config['client'], tbeg, tend)
        else:
            st_xpfo, inv_xpfo = __request_data(config['seeds']['translation2'], config['client'], tbeg, tend)
        
        ## load roation data
        st_bspf, inv_bspf = __request_data(config['seeds']['rotation'], config['client'], tbeg, tend)
   
    except Exception as e:
#        print(e)
#        print(f" -> failed to load data: {tbeg}-{tend}")
        logging.error(f" -> failed to load data: {(tbeg+3600).date)}")
        return

    if st_bspf is None or inv_bspf is None:
        return

    try:
        ## Processing Data
        st_xpfo = st_xpfo.remove_response(inventory=inv_xpfo, output="ACC")

        st_bspf = st_bspf.remove_sensitivity(inventory=inv_bspf)
        
        if tbeg < UTCDateTime("2023-04-01"):
            st_bspf = st_bspf.resample(40.0)

        ## normalize to better compare rotation and velocity
        st_xpfo = st_xpfo.normalize(global_max=False)
        st_bspf = st_bspf.normalize(global_max=False)

        ## Join Data
        st = Stream()

        st += st_bspf
        st += st_xpfo

        st.detrend("linear")
        st.taper(0.01)
    #     st.filter('bandpass', freqmin=1.0, freqmax=18, corners=4, zerophase=True)  # optional prefiltering
        st.filter('highpass', freq=0.01, corners=4, zerophase=True)
                  
        trig = __trigger(config, st)

        
        del st

        ## store trigger list
        #print(f"-> {config['output_path']}trigger_{date}_{jj}.pkl")
        if not os.path.isdir(config['output_path']+f"tmp"):
            os.mkdir(config['output_path']+f"tmp")

        __store_as_pickle(trig, config['output_path']+f"tmp/trigger_{tbeg}_{tend}_{jj}.pkl")

    except:
        logging.error(f" -> {tbeg} {tend}  fatal exception occurred!")


## MAIN ___________________________________________

if __name__ == '__main__':

    pprint(config)

    logging.basicConfig(filename=config['output_path']+config['output_logfile'], 
                        encoding='utf-8',
                        filemode='w',
                        level=logging.DEBUG
                        )

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
#    with mp.Pool(processes=5) as pool:

    n_cpu = mp.cpu_count()
    logging.info(f" -> using {n_cpu-1} of {n_cpu} CPU cores!")
    
    with mp.Pool(int(n_cpu-1)) as pool:
    
        list(tqdm(pool.imap_unordered(main, list_of_times), total=len(list_of_times)))

    pool.close()
    pool.join()



    ## join files
    print("\n -> joining pickle files to one trigger file ...")
    triggers = __join_pickle_files(config)

    print(f"\n -> writing triggered events to file: \n  -> {config['output_path']}{config['outout_filename']}")
    __store_as_pickle(triggers, config['output_path']+config['output_filename'])

    print("\n -> Done")


## End of File
