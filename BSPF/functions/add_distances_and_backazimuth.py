def __add_distances_and_backazimuth(reference_latitude, reference_longitude, df):

    from obspy.geodetics.base import gps2dist_azimuth
    from numpy import zeros
    
    dist = zeros(len(df))
    baz = zeros(len(df))

    
    for ii, ev in enumerate(df.index):
        try:
            dist[ii], az, baz[ii] = gps2dist_azimuth(reference_latitude, reference_longitude,
                                                     df.latitude[ii], df.longitude[ii],
                                                     a=6378137.0, f=0.0033528106647474805
                                                     )
        except:
            print(" -> failed to compute!")
            
    df['backazimuth'] = baz
    df['distances_km'] = dist/1000

    return df