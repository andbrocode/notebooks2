def __get_time_intervals(tbeg, tend, interval_seconds, interval_overlap):

    from obspy import UTCDateTime

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    times = []
    t1, t2 = tbeg - interval_overlap, tbeg + interval_seconds + interval_overlap
    while t2 <= tend:
        times.append((t1, t2))
        t1 = t1 + interval_seconds
        t2 = t2 + interval_seconds

    return times