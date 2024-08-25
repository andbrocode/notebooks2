def __read_DEM(filename):

    from osgeo import gdal
    from utm import from_latlon
    from numpy import linspace, nan

    gdal_data = gdal.Open(filename)
    gdal_band = gdal_data.GetRasterBand(1)
    nodataval = gdal_band.GetNoDataValue()

    # convert to a numpy array
    data_array = gdal_data.ReadAsArray().astype(float)
    data_array

    # replace missing values if necessary
    if (data_array == nodataval).any():
        data_array[data_array == nodataval] = nan

    Nx, Ny = data_array.shape

    lon_step = gdal_data.GetGeoTransform()[1]
    lat_step = gdal_data.GetGeoTransform()[5]

    out = {}
    out["longitude"] = linspace(gdal_data.GetGeoTransform()[0], Nx*lon_step+gdal_data.GetGeoTransform()[0], Nx)
    out["latitude"] = linspace(gdal_data.GetGeoTransform()[3], Ny*lat_step+gdal_data.GetGeoTransform()[3], Ny)
    out["data"] = data_array
    out["utm_e"], out["utm_n"], _, _ = from_latlon(out["latitude"], out["longitude"])

    return out