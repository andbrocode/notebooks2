class sagnacf:

    import numpy as np
    from obspy import UTCDateTime

    def __init__(self, delta=60, ring="Z"):

        self.delta = delta

        self.ring = ring

    def load_data(self, path, filename):
        self.data = read_pickle(path+filename)

    def relative_timing(self, tbeg):
        self.data['time_sec'] = self.data.time1 - UTCDateTime(tbeg) + self.delta/2

    def trim(self, tbeg=None, tend=None):
        if tbeg is not None:
            _df = self.data
            self.data = _df[_df.time1 >= UTCDateTime(tbeg)]
        if tend is not None:
            _df = self.data
            self.data = _df[_df.time2 <= UTCDateTime(tend)]

    def apply_backscatter_correction(self):

        # copy data
        _data = self.data

        # compute unwrapped phase difference
        phase_difference = np.unwrap(_data.f1_phw) - np.unwrap(_data.f2_phw)

        # compute backscatter correction
        _data['fj_bs'], _, _ = self.bs_correction(_data.f1_ac / _data.f1_dc,
                                                  _data.f2_ac / _data.f2_dc,
                                                  phase_difference,
                                                  _data.fj_fs,
                                                  np.nanmedian(_data.fj_fs),
                                                  cm_filter_factor=1.033,
                                                 )
        # reasign updated dataframe
        self.data = _data

    def get_scalefactor(self, output):

        from numpy import pi, sqrt, arccos, deg2rad, arcsin, cos, sin, array, zeros

        # angle in horizontal plane
        h_rot = {"Z":0, "U":0, "V":60, "W":60}

        # angle from vertical
        v_rot = {"Z":0, "U":109.5, "V":70.5, "W":70.5}
        # v_rot = {"Z":-90, "U":19.5, "V":-19.5, "W":-19.5}

        # side length
        L = {"Z":11.2, "U":12, "V":12, "W":12}

        # wavelength
        lamda = 632.8e-9

        # Scale factor
        S = (sqrt(3)*L[self.ring])/(3*lamda)

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

        pv = deg2rad(v_rot[self.ring])
        ph = deg2rad(h_rot[self.ring])

        # normal vector of ring
        nx = array([[sin(pv)*cos(ph)], [sin(pv)*sin(ph)], [cos(pv)]])

        one = array([0, 0, 1])

        out = S * ( one @ ( D @ (R @ nx) ) )[0]

        self.get_scalefactor = out

        if output:
            return out

    @staticmethod
    def sagnac_to_tilt(data, ring="Z", tilt="n-s"):

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

        if tilt == "n-s":
            out = ( (data /S /omegaE[2]) - term1 - term2 ) / fz
        elif tilt == "e-w":
            out = ( (data /S /omegaE[2]) - term1 - term2 ) / fa
        elif tilt == "tilt_to_hz":
            out = zeros(len(data))
            for n in range(len(data)):
                out[n] = S * ( (omegaE + data[n]) @ ( D @ (R @ nx) ) )

        return out

    @staticmethod
    def bs_correction(m01, m02, phase0, w_obs, fs0, cm_filter_factor=1.033):

        from numpy import array, sin, cos

        # Correct for bias
        m1 = m01 * ( 1 + m01**2 / 4 )
        m2 = m02 * ( 1 + m02**2 / 4 )

        # angular correction for phase
        phase = phase0 + 0.5 * m1 * m2 * sin( phase0 )

        # compute squares of common-mode modulations
        m2c = ( m1**2 + m2**2 + 2*m1*m2*cos( phase ) ) / 4

        # compute squares of differential-mode modulations
        m2d = ( m1**2 + m2**2 - 2*m1*m2*cos( phase ) ) / 4  ## different angle!

        # correct m2c for gain saturation of a HeNe laser
        # m2c = m2c * ( 1 + ( beta + theta )**2 * fL**2 * I0**2 / ws**2 )
        m2c = m2c * cm_filter_factor

        # compute backscatter correction factor
        M = m2c - m2d + 0.25 * m1**2 * m2**2 * sin(phase)**2

        # correction term
        term = ( 4 + M ) / ( 4 - M )

        # backscatter correction
        correction = -1 * ( term - 1 ) * fs0
        # w_corrected = np.array(w_obs) + correction

        # apply backscatter correction
        w_corrected = array(w_obs) * term

        return w_corrected, correction, term
