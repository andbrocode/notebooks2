def __load_mlti(tbeg, tend, ring, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    year = tbeg.year

    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    path_to_mlti = path_to_archive+f"romy_archive/{year}/BW/CROMY/{year}_romy_{rings[ring]}_mlti.log"

    mlti = read_csv(path_to_mlti, names=["time_utc","Action","ERROR"])

    mlti = mlti[(mlti.time_utc > tbeg) & (mlti.time_utc < tend)]

    return mlti