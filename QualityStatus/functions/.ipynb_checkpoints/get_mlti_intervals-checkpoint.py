def __get_mlti_intervals(mlti_times, time_delta=60):

    from obspy import UTCDateTime
    from numpy import array

    if len(mlti_times) == 0:
        return array([]), array([])

    t1, t2 = [], []
    for k, _t in enumerate(mlti_times):

        _t = UTCDateTime(_t)

        if k == 0:
            _tlast = _t
            t1.append(_t)

        if _t -_tlast > time_delta:
            t2.append(_tlast)
            t1.append(_t)

        _tlast = _t

    # t2.append(_t)
    t2.append(mlti_times[-1])

    return array(t1), array(t2)