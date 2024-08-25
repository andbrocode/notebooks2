def __read_wromy_data(t1, t2, cha, path_to_data):

    from os import path
    from pandas import DataFrame, read_csv, concat
    from numpy import nan
    from obspy import UTCDateTime


    datapath = f"{path_to_data}{UTCDateTime(t1).year}/BW/WROMY/{cha}.D/"

    if not path.isdir(datapath):
        print(f" -> Path: {datapath}, does not exists!")
        return

    j1, j2 = UTCDateTime(t1).julday, UTCDateTime(t2).julday
    year = UTCDateTime(t1).year

    df = DataFrame()

    for doy in range(j1, j2+1):

        doy = str(doy).rjust(3, "0")

        filename = f'BW.WROMY.{cha}.D.{year}.{doy}'

        # print(f'   reading {filename} ...')

        try:
            df0 = read_csv(datapath+filename)

            # replace error indicating values (-9999, 999.9) with NaN values
            df0.replace(to_replace=-9999, value=nan, inplace=True)
            df0.replace(to_replace=999.9, value=nan, inplace=True)


            if doy == j1:
                df = df0
            else:
                df = concat([df, df0])
        except:
            print(f" -> file: {filename}, does not exists!")

    df.reset_index(inplace=True, drop=True)

    # add columns with total seconds
    if 'Seconds' in df.columns:
        totalSeconds = df.Seconds + (df.Date - df.Date.iloc[0]) * 86400
        df['totalSeconds'] = totalSeconds

    return df