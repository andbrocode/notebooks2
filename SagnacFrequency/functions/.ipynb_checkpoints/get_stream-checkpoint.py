def __get_stream(arr, dt, seed, starttime):

    from obspy import Stream, Trace

    net, sta, loc, cha = seed.split(".")

    tr00 = Trace()
    tr00.data = arr
    tr00.stats.delta = dt
    tr00.stats.starttime = starttime
    tr00.stats.network = net
    tr00.stats.station = sta
    tr00.stats.location = loc
    tr00.stats.channel = cha

    return Stream(tr00)