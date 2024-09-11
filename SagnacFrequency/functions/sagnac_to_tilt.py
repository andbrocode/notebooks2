def __sagnac_to_tilt(data=None, ring="Z", tilt="n-s"):

    from numpy import pi, sqrt, arccos, deg2rad, arcsin, cos, sin, array, zeros

    # angle in horizontal plane
    h_rot = {"Z":0, "U":0, "V":60, "W":60}

    # angle from vertical
    v_rot = {"Z":0, "U":109.5, "V":70.5, "W":70.5}
    # v_rot = {"Z":-90, "U":19.5, "V":-19.5, "W":-19.5}

    # side length
    L = {"Z":11.2, "U":12, "V":12, "W":12}

    # Scale factor
    S = (sqrt(3)*L[ring])/(3*632.8e-9)

    # ROMY latitude
    lat = deg2rad(48.162941)
    lon = deg2rad(11.275501)

    # nominal Earth rotation
    omegaE = 2*pi/86400 * array([0, 0, 1])

    # matrix 1
    D = array([[-sin(lat)*cos(lon), -sin(lon), cos(lat)*cos(lon)],
               [sin(lat)*sin(lon), cos(lon), cos(lat)*sin(lon)],
               [cos(lat), 0, sin(lat)]
              ])

    # tilt
    da = deg2rad(0)
    dz = deg2rad(0)

    # tilt matrix
    R = array([[1, -da, -dz], [da,  1, 0], [dz, 0, 1]])

    pv = deg2rad(v_rot[ring])
    ph = deg2rad(h_rot[ring])

    # normal vector of ring
    nx = array([[sin(pv)*cos(ph)], [sin(pv)*sin(ph)], [cos(pv)]])

    # terms
    # term1 = cos(v_rot[ring])*sin(lat)
    # term2 = cos(lat)*sin(v_rot[ring])*cos(h_rot[ring])
    term1 = cos(pv)*sin(lat)
    term2 = cos(lat)*sin(pv)*cos(ph)

    # tilt factor
    # fz = sin(lat)*sin(v_rot[ring])*cos(h_rot[ring]) - cos(v_rot[ring])*cos(lat)
    # fa = sin(v_rot[ring])*sin(h_rot[ring])*cos(lat)
    fz = sin(lat)*sin(pv)*cos(ph) - cos(pv)*cos(lat)
    fa = sin(pv)*sin(ph)*cos(lat)

    if data is None:
        out = S * ( omegaE @ ( D @ (R @ nx) ) )[0]

    else:
        if tilt == "n-s":
            out = ( (data /S /omegaE[2]) - term1 - term2 ) / fz
        elif tilt == "e-w":
            out = ( (data /S /omegaE[2]) - term1 - term2 ) / fa
        elif tilt == "rad_to_hz":
            out = zeros(len(data))
            for n in range(len(data)):
                out[n] = S * ( (omegaE + data[n]) @ ( D @ (R @ nx) ) )

    return out