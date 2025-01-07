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