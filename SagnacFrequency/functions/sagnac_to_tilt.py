def __sagnac_to_tilt(data, ring):

    from numpy import pi, sqrt, arccos, deg2rad, arcsin

    dip = {"Z":0, "U":109.5, "V":70.5, "W":70.5}

    L = {"Z":11.2, "U":12, "V":12, "W":12}

    ## Scale factor
    S = (sqrt(3)*L[ring])/(3*632.8e-9)

    ## ROMY latitude
    lat = 48.162941

    ## nominal Earth rotation
    omegaE = 2*pi/86400

    return arcsin(data /omegaE /S) - deg2rad(lat) - deg2rad(dip[ring])