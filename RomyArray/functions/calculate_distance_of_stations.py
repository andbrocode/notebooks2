def __calculate_distance_of_stations(array_stations, output="km", ref_station=None):

    '''
    compute distances between stations

    '''

    from pandas import DataFrame
    from numpy import zeros, sqrt
    from obspy.geodetics import locations2degrees
    from obspy.signal.util import util_geo_km

    def get_distance(x, y):
        return sqrt(x**2 + y**2)

    N = len(array_stations)

    # with respect to reference station
    if ref_station is not None:

        distances = {}

        net, sta, loc, cha = ref_station.split(".")
        refsta = f"{net}.{sta}"

        reflat = array_stations[array_stations.codes == refsta]["lat"].iloc[0]
        reflon = array_stations[array_stations.codes == refsta]["lon"].iloc[0]

        for j, station2 in array_stations.iterrows():
            name = str(station2.codes)

            x, y = util_geo_km(reflon, reflat, station2.lon, station2.lat)

            if output == "km":
                distances[name] = round(get_distance(x, y), 2)
            elif output == "m":
                distances[name] = round(get_distance(x, y)*1000, 2)

    # cross station distances
    if ref_station is None:

        distances = zeros((N, N))

        for i, station1 in array_stations.iterrows():

            for j, station2 in array_stations.iterrows():

                x, y = util_geo_km(station1.lon, station1.lat, station2.lon, station2.lat)

                if output == "km":
                    distances[i][j] = round(get_distance(x, y), 2)

                elif output == "m":
                    distances[i][j] = round(get_distance(x, y)*1000, 2)

    return distances