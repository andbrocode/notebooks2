def __sagnac_to_geometry(data, ring):

    from numpy import sin, pi, deg2rad, sqrt, nanmedian

    dip = {"Z":0, "U":109.5, "V":70.5, "W":70.5}

    L = {"Z":11.2, "U":12, "V":12, "W":12}

    ## triangular height
    H = data * 3/2 * 632.8e-9 / (2*pi/86400 * sin(deg2rad(48.162941 + dip[ring])))

    ## side length
    a =  data * 3 / sqrt(3) * 632.8e-9 / (2*pi/86400 * sin(deg2rad(48.162941 + dip[ring])))

    H_expected = L[ring]/2*sqrt(3)

    H_relative = abs(H-H_expected)
    H_relative = H - nanmedian(H)

    out = {}
    out['triangular_height'] = H
    out['triangular_height_nominal'] = H_expected
    out['side_length'] = a

    return out