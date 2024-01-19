def __load_bathymetry(path_to_data, box=(-180, -90, 180, 90)):

    from numpy import array
    from glob import glob
    import cartopy.io.shapereader as shpreader
    import matplotlib

    # Read shapefiles, sorted by depth
    shp_dict = {}
    files = glob(path_to_data+'*.shp')

    files.sort()

    depths = []
    for f in files:
        depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
        depths.append(depth)
        bbox = box  # (x0, y0, x1, y1)
        nei = shpreader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei

    depths_str = array(depths)[::-1]  # sort from surface to bottom

    # Construct a discrete colormap with colors corresponding to each depth
    depths = depths_str.astype(int)
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))

    out = {}
    out['depths'] = depths
    out['shp_dict'] = shp_dict
    out['colors_depths'] = colors_depths
    out['blues_cm'] = blues_cm
    out['depths_str'] = depths_str
    out['norm'] = norm

    return out