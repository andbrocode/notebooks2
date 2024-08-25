def __get_angle(N, E, out="deg"):
    
    import numpy as np

    ang = np.zeros(len(E))
    for i, (n, e) in enumerate(zip(N, E)):
        # Q2
        if n >= 0 and e < 0:
            ang[i] = np.rad2deg(np.arctan(n/e)) + 180
        # Q3
        elif n < 0 and e < 0:
            ang[i] = np.rad2deg(np.arctan(n/e)) + 180
        # Q4
        elif n < 0 and e >= 0:
            ang[i] = np.rad2deg(np.arctan(n/e)) + 360
        # Q1
        else:
            ang[i] = np.rad2deg(np.arctan(n/e))
    if out == "deg":
        return ang
    elif out == "rad":
        return np.deg2rad(ang)