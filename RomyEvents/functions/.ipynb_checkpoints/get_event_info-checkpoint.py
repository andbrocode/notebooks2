def __get_event_info(config, min_mag=5.0):

    from obspy.geodetics.base import gps2dist_azimuth

    event = config['Client'].get_events(starttime=config['tbeg']-3600, endtime=config['tend'], minmagnitude=min_mag)

    if len(event) > 1:
        print("-> more than one event\n")

    config['event'] = event[0]

    ## Eventtime
    config['eventtime'] = event[0].origins[0].time

    print(event[0])

    dist, az, baz = gps2dist_azimuth(event[0].origins[0].latitude, event[0].origins[0].longitude,
                                     config['ROMY_lat'], config['ROMY_lon'],
                                     )
    print("Distance ", dist/1000, "m", "Azimuth ", az, "Backazimuth ", baz)

    return config, dist, baz, az