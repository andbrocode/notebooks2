def __add_distances_and_backazimuth(config, df):

    from numpy import zeros
    from obspy.geodetics.base import gps2dist_azimuth

    dist = zeros(len(df))
    baz = zeros(len(df))


    for ii, ev in enumerate(df.index):
        try:
            dist[ii], az, baz[ii] = gps2dist_azimuth(df.latitude[ii], df.longitude[ii],
                                                     config['ROMY_lat'], config['ROMY_lon'],
                                                     # a=6378137.0, f=0.0033528106647474805
                                                     )
        except:
            print(" -> failed to compute!")

    df['backazimuth'] = baz
    df['distances_km'] = dist/1000

    return df