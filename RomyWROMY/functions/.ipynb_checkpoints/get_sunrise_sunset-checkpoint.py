def __get_sunrise_sunset(tbeg, tend):

    from obspy import UTCDateTime
    from pandas import date_range
    from suntime import Sun, SunTimeException

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    sun = Sun(48.162941, 11.275501)

    sr, ss = [], []
    for _date in date_range(tbeg.date, tend.date):
        # abd = datetime.date(2014, 10, 3)
        sr.append(sun.get_local_sunrise_time(_date))
        ss.append(sun.get_local_sunset_time(_date))

    return sr, ss