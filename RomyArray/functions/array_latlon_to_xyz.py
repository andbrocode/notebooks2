def __array_latlon_to_xyz(array_stations, ref_station="GR.FUR"):
    """Convert angluar to cartesian coordiantes

    latitude is the 90deg - zenith angle in range [-90;90]
    lonitude is the azimuthal angle in range [-180;180] 
    """

    from numpy import zeros
    import utm

    # modify ref_station
    net, sta, loc, cha = ref_station.split(".")
    ref_station = f"{net}.{sta}"

    if ref_station not in list(array_stations.codes):
        print(f"-> {ref_station} not vaild")
        return


    for _i in ["x_m", "y_m", "z_m", "utm_n", "utm_e", "utm_zone", "utm_letter"]:
        array_stations[_i] = zeros(array_stations.shape[0])

    sta_ref = array_stations[array_stations.codes == ref_station]
    utm_ref_e, utm_ref_n, utm_zone, utm_letter = utm.from_latlon(sta_ref.lat.iloc[0], sta_ref.lon.iloc[0])
    z_ref = sta_ref.elev.iloc[0]

    for i, sta in array_stations.iterrows():

        utm_e, utm_n, utm_zone, utm_letter = utm.from_latlon(sta.lat, sta.lon)

        array_stations.loc[i, "utm_n"] = utm_n
        array_stations.loc[i, "utm_e"] = utm_e
        array_stations.loc[i, "x_m"] = round(( utm_e - utm_ref_e ), 2)
        array_stations.loc[i, "y_m"] = round(( utm_n - utm_ref_n ), 2)
        array_stations.loc[i, "z_m"] = ( array_stations.loc[i, "elev"] - z_ref ) * 1e3
        array_stations.loc[i, "utm_zone"] = utm_zone
        array_stations.loc[i, "utm_letter"] = str(utm_letter)

    return array_stations