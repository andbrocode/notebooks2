class sagnacf:

    import numpy as np
    from obspy import UTCDateTime

    def __init__(self, delta=60):

        self.delta = delta

    def load_data(self, path, filename):
        from pandas import read_pickle
        self.data = read_pickle(path+filename)

    def relative_timing(self, tbeg):
        from obspy import UTCDateTime
        self.data['time_sec'] = self.data.time1 - UTCDateTime(tbeg) + self.delta/2

    def trim(self, tbeg=None, tend=None):
        from obspy import UTCDateTime
        if tbeg is not None:
            _df = self.data
            self.data = _df[_df.time1 >= UTCDateTime(tbeg)]
        if tend is not None:
            _df = self.data
            self.data = _df[_df.time2 <= UTCDateTime(tend)]

    def apply_backscatter_correction(self):
        from numpy import unwrap, nanmedian

        # copy data
        _data = self.data

        # compute unwrapped phase difference
        phase_difference = unwrap(_data.f1_phw) - unwrap(_data.f2_phw)

        # compute backscatter correction
        _data['fj_bs'], _, _ = self.bs_correction(_data.f1_ac / _data.f1_dc,
                                                  _data.f2_ac / _data.f2_dc,
                                                  phase_difference,
                                                  _data.fj_fs,
                                                  nanmedian(_data.fj_fs),
                                                  cm_filter_factor=1.033,
                                                 )
        # reasign updated dataframe
        self.data = _data

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
