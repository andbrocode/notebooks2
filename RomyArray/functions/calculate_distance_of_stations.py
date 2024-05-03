def __calculate_distance_of_stations(array_stations, output="km", ref_station=None):
    '''
    from obspy.geodetics import locations2degrees

    '''

    from numpy import round, zeros
    from obspy.geodetics import locations2degrees

    N = len(array_stations)

    # modify ref_station
    if ref_station is not None:
        net, sta, loc, cha = ref_station.split(".")
        ref_station = f"{net}.{sta}"

    if ref_station is not None:
        dist_in_deg = {}
        station1 = array_stations[array_stations.codes == ref_station]
        lat1, lon1 = station1.lat, station1.lon
        for j, station2 in array_stations.iterrows():
            name = str(station2.codes)
            _dist= locations2degrees(lat1  = lat1,
                                      long1 = lon1,
                                      lat2  = station2[2],
                                      long2 = station2[1],
                                      )
            if output == "km":
                dist_in_deg[name] = round(_dist[0]*111, decimals=2)
            elif output == "deg":
                dist_in_deg[name] = round(_dist[0], decimals=2)

    else:
        dist_in_deg = zeros((N, N))
        for i, station1 in array_stations.iterrows():
            for j, station2 in array_stations.iterrows():
                 _dist = locations2degrees(lat1  = station1[2],
                                          long1 = station1[1],
                                          lat2  = station2[2],
                                          long2 = station2[1],
                                          )
            if output == "km":
                dist_in_deg[i][j] = round(_dist*111, decimals=2)
            elif output == "deg":
                dist_in_deg[i][j] = round(_dist, decimals=2)

    return dist_in_deg