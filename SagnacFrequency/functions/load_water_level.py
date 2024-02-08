def __load_water_level(tbeg, tend):

    from datetime import date
    from pandas import read_csv, concat, DataFrame, date_range
    from obspy import UTCDateTime

    path_to_data = "/lamont/Pegel/"

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if tbeg < UTCDateTime("2023-11-26"):
        print(f" -> no good data before 2023-11-26!")
        tbeg = UTCDateTime("2023-11-27")

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    df = DataFrame()
    for dat in date_range(dd1, dd2):
        file = f"{str(dat)[:4]}/PG"+str(dat)[:10].replace("-", "")+".dat"
        try:
            df0 = read_csv(path_to_data+file, delimiter=" ")
            df = concat([df, df0])
        except:
            print(f"error for {file}")

    ## convert data
    df['pegel'] = df.pegel*0.75
    df['temperatur'] = df.temperatur*5

    ## correct seconds
    df['times_utc'] = [UTCDateTime(f"{_d[-4:]+_d[3:5]+_d[:2]} {_t}")  for _d, _t in zip(df['day'], df['hour'])]
    df['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df['times_utc']]

    ## remove columns hour and day
    df.drop(columns=["hour", "day"], inplace=True)

    ## reset index to make it continous
    df.reset_index(inplace=True)

    if df.empty:
        print(" -> empty dataframe!")
        return df

    # trim to defined times
    df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]
    return df