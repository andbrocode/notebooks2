#!/usr/bin/env python
# coding: utf-8

from obspy.core.trace import Trace
from obspy.core.stream import Stream


def __write_to_mseed(idata, itime, oname, sps):
    

    ## create trace object
    synthetic_event = Trace(idata)
    time_sythetic_event = Trace(itime)

    ## edit header of trace 
    synthetic_event.stats.sampling_rate = sps

    time_sythetic_event.sampling_rate = sps

    ##create a stream
    synthetic_event_out = Stream(traces=[synthetic_event, time_sythetic_event])

    ## write trace object to file

    synthetic_event_out.write(oname, format="MSEED")

    print(f"--> stored synthetic event to: {oname}")
