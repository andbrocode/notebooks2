class baroArray:

    def __init__(self, seeds=[], coords=None,
                 path_to_figs="./", verbose=False,
                 fmin=None, fmax=None, out_seed="BW.BRMY..BDX"):

        # specify station codes
        self.seeds = seeds

        # station coordinates
        self.coordinates = coords

        # path to output figures
        self.path_to_figs = path_to_figs

        self.verbose = verbose

        self.fmin = fmin

        self.fmax = fmax

        self.apply_bandpass = True

        self.plot = False

        self.out_seed = out_seed

        self.aperture = None

        # adr parameters
        self.vp = 1
        self.vs = 1
        self.sigmau = 1e-7

    def load_data(self, tbeg, tend, path_to_sds, verbose=False):

        from obspy import UTCDateTime, Stream
        from numpy import isnan

        # starttime and endtime
        self.tbeg = UTCDateTime(tbeg)
        self.tend = UTCDateTime(tend)

        # specify path to data
        self.path_to_sds = path_to_sds

        # establish data stream
        self.st0 = Stream()

        for _i, seed in enumerate(self.seeds):

            # if "ROMY" in seed:
            #     seed = "BW.PROMY.03.LDI"

            ps = self.read_sds(self.path_to_sds, seed, self.tbeg, self.tend)

            for tr in ps:
                if isnan(tr.data).any():
                    if self.verbose or verbose:
                        print("-> NaN found! Interpolating NaNs ...")
                    tr.data = self.interpolate_nan(tr.data)

            try:

                # convert from Pa to hPa
                if "PROMY.03" in seed:
                    ps[0].data = ps[0].data / 100

#                 ps = ps.detrend("simple")

#                 ps = ps.taper(0.01, type="cosine")

#                 ps = ps.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4, zerophase=True)

#                 ps = ps.resample(self.fmax*10, no_filter=True)

                # convert from hPa to Pa
                ps[0].data = ps[0].data * 100

                if self.verbose or verbose:
                    print(ps)

                self.st0 += ps

                del ps

            except:
                print(f" -> Error encountered for {seed}")

    def preprocessing(self, fmin, fmax, resample=False):

        self.fmin = fmin
        self.fmax = fmax

        self.st = self.st0.copy()

        self.st.detrend("simple")

        self.st.taper(0.01, type="cosine")

        self.st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4, zerophase=True)

        if resample:
            self.st.resample(self.fmax*10, no_filter=True)

    def compute_gradient(self, reference, fmin=None, fmax=None, verbose=False):

        import matplotlib.pyplot as plt

        from obspy import Stream
        from obspy.signal.util import util_geo_km
        from numpy import array

        # assign reference
        self.reference = reference

        # get data stream
        st = self.st.copy()

        # change channel name
        for tr in st:
            tr.stats.channel = "LDZ"

        # add placebo N component
        _stN = st.copy()
        for tr in _stN:
            tr.stats.channel = "LDN"

        # add placebo E component
        _stE = st.copy()
        for tr in _stE:
            tr.stats.channel = "LDE"

        st += _stN.copy()
        st += _stE.copy()

        # processing
        st = st.detrend("linear")
        st = st.detrend("demean")

        # bandpass filter
        if self.apply_bandpass:
            st = st.taper(0.02, type="cosine")
            if fmin is None and fmax is None:
                fmin, fmax = self.fmin, self.fmax
                st = st.filter('bandpass', freqmin=self.fmin, freqmax=self.fmax, corners=4, zerophase=True)
            else:
                st = st.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
            if verbose:
                print(f" -> applying bandpass: {fmin} - {fmax} Hz")

        # compute relataive distances
        dist = []
        for k in self.coordinates.keys():

            coo = self.coordinates[k]

            ref_sta = self.reference.split(".")[1]

            # reference
            coo_ref = self.coordinates[ref_sta]
            ref_lon, ref_lat, ref_height = coo_ref['lon'], coo_ref['lat'], coo_ref['height']

            # convert
            lon, lat = util_geo_km(ref_lon, ref_lat, coo['lon'], coo['lat'])

            # convert unit from km to m
            dist.append([lon*1000, lat*1000, coo['height']-ref_height])

        # assign distances
        self.distances = array(dist)

        # check if enough stations for ADR are available otherwise continue
        if len(st) < 3*3:
            print(" -> not enough stations (< 3) for ADR computation!")
            return
        else:
            if verbose:
                print(f" -> continue computing ADR for {int(len(st)/3)} of {len(self.seeds)} stations ...")

        # homogenize the time line
        # st = __adjust_time_line(st, reference=config['reference_station'])

        # check for same amount of samples
        self.check_samples_in_stream()

        # compute array derived rotation (ADR)
        try:
            self.compute_adr(st)
        except Exception as e:
            print(f"failed")
            self.gradient = Stream()
            if verbose:
                print(e)

        # trim to requested interval
        self.gradient.trim(self.tbeg, self.tend)

        # remove Z trace (not possible to compute for pressure)
        for tr in self.gradient:
            if "Z" in tr.stats.channel:
                self.gradient.remove(tr);

    def compute_adr(self, st):

        from obspy.signal import array_analysis as AA
        from obspy import Stream
        from numpy import array, arange, transpose

        # reference stream
        sta = self.reference.split(".")[1]
        ref_st = st.select(station=sta)[0]

        # prepare data arrays
        tsz, tsn, tse = [], [], []
        for tr in st:
            try:
                if "Z" in tr.stats.channel:
                    tsz.append(tr.data)
                elif "N" in tr.stats.channel:
                    tsn.append(tr.data)
                elif "E" in tr.stats.channel:
                    tse.append(tr.data)
            except:
                if self.verbose:
                    print(" -> stream data could not be appended!")

        # make sure input is array type
        tse, tsn, tsz = array(tse), array(tsn), array(tsz)

        # define array for subarray stations with linear numbering
        substations = arange(len(self.seeds))

        try:
            result = AA.array_rotation_strain(substations,
                                              transpose(tse),
                                              transpose(tsn),
                                              transpose(tsz),
                                              self.vp,
                                              self.vs,
                                              self.distances,
                                              self.sigmau,
                                             )
        except Exception as e:
            if self.verbose:
                print(e)
                print("\n -> failed to compute ADR...")
                self.gradient = Stream()
                return

        # create rotation stream and add data
        out = Stream()
        out += ref_st.copy()
        out += ref_st.copy()
        out += ref_st.copy()

        out[0].data = result['ts_w3']
        out[1].data = result['ts_w2']
        out[2].data = result['ts_w1']

        out[0].stats.network = self.out_seed.split(".")[0]
        out[1].stats.network = self.out_seed.split(".")[0]
        out[2].stats.network = self.out_seed.split(".")[0]

        out[0].stats.station = self.out_seed.split(".")[1]
        out[1].stats.station = self.out_seed.split(".")[1]
        out[2].stats.station = self.out_seed.split(".")[1]

        out[0].stats.location = self.out_seed.split(".")[2]
        out[1].stats.location = self.out_seed.split(".")[2]
        out[2].stats.location = self.out_seed.split(".")[2]

        out[0].stats.channel = 'BDZ'
        out[1].stats.channel = 'BDN'
        out[2].stats.channel = 'BDE'

        out = out.detrend('linear')

        self.gradient = out

    def check_samples_in_stream(self):

        Rnet, Rsta = self.reference.split(".")

        Rsamples = self.st.select(network=Rnet, station=Rsta)[0].stats.npts

        for tr in self.st:
            if tr.stats.npts != Rsamples:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {Rsamples})")
                self.st.remove(tr)

    def write_stream_to_sds(self, path_to_dir=None, verbose=False):

        if path_to_dir is None:
            path_to_sds = self.path_to_sds
        else:
            path_to_sds = path_to_dir
        import os

        # check if output path exists
        if not os.path.exists(path_to_sds):
            print(f" -> {path_to_sds} does not exist!")
            return

        for tr in self.gradient:
            nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
            yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

            if not os.path.exists(path_to_sds+f"{yy}/"):
                os.mkdir(path_to_sds+f"{yy}/")
                print(f"creating: {path_to_sds}{yy}/")
            if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
                os.mkdir(path_to_sds+f"{yy}/{nn}/")
                print(f"creating: {path_to_sds}{yy}/{nn}/")
            if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
                os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
                print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
            if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
                os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
                print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

        for tr in self.gradient:
            nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
            yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

            try:
                st_tmp = self.gradient.copy()
                st_tmp.select(network=nn, station=ss, location=ll, channel=cc).write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
            except:
                if self.verbose or verbose:
                    print(f" -> failed to write: {cc}")
            finally:
                if self.verbose or verbose:
                    print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")

    def get_absolute_gradient(self):

        from numpy import sqrt

        E = self.gradient.select(channel="*E")[0].data
        N = self.gradient.select(channel="*N")[0].data

        self.abs_gradient = sqrt(N**2 + E**2)

    def get_angle(self, outunit="deg", relative_to_north=True):

        import numpy as np

        E = self.gradient.select(channel="*E")[0].data
        N = self.gradient.select(channel="*N")[0].data

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

        # angle relative to north
        if relative_to_north:
            ang = ((ang + 90) % 360) - 180

        if outunit == "deg":
            self.angles = ang
        elif outunit == "rad":
            self.angles = np.deg2rad(ang)

    def makeplot(self, plot=False):

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        t = self.gradient.select(channel="*N")[0].times()
        N = self.gradient.select(channel="*N")[0].data
        E = self.gradient.select(channel="*E")[0].data

        # pressure data
        tt = self.st.select(station="PROMY")[0].times()
        pp = self.st.select(station="PROMY")[0].data

        # get angles
        self.get_angle()
        brmy_ang = self.angles

        # get absolute gradient
        self.get_absolute_gradient()
        brmy_abs = self.abs_gradient

        # normalized absolute gradient
        r = brmy_abs / max(brmy_abs)

        # specify limit for r
        rlim = 0.2

        # angle as radians
        theta = np.deg2rad(brmy_ang)

        # ___________________________________________

        tscale, tunit = 1/3600, "hour"

        gscale, gunit = 1e6, ""

        Ncol, Nrow = 9, 4

        font = 12

        fig = plt.figure(figsize=(15, 5))

        gs1 = GridSpec(Nrow, Ncol, figure=fig, hspace=0.1, wspace=0.2)

        ax0 = fig.add_subplot(gs1[0, :5])
        ax1 = fig.add_subplot(gs1[1, :5])
        ax2 = fig.add_subplot(gs1[2, :5])
        ax3 = fig.add_subplot(gs1[3, :5])
        ax4 = fig.add_subplot(gs1[0:4, 5:], polar=True)

        axes = [ax0, ax1, ax2, ax3, ax4]

        # ___________________________________________

        ax0.plot(tt*tscale, pp, color="k")

        ax0.set_xticklabels([])
        ax0.set_ylabel(f"Pressure\n(Pa)", fontsize=font)
        # ax0.legend(loc=1)
        ax0.text(0.99, 0.97, self.reference, ha="right", va="top", transform=ax0.transAxes, fontsize=font)

        # ___________________________________________

        ax1.scatter(t*tscale, N*gscale, c=t*tscale, s=1, label="N-S")
        ax1.set_xticklabels([])
        ax1.set_ylim(-max(abs(N))*gscale, max(abs(N))*gscale)
        ax1.set_ylabel(f"Spatial\nGradient", fontsize=font)
        ax1.text(0.99, 0.97, "N-S", ha="right", va="top", transform=ax1.transAxes, fontsize=font)

        # ___________________________________________

        ax2.scatter(t*tscale, E*gscale, c=t*tscale, s=1, label="E-W")
        ax2.set_xticklabels([])
        ax2.set_ylim(-max(abs(E))*gscale, max(abs(E))*gscale)
        ax2.set_ylabel(f"Spatial\nGradient", fontsize=font)
        ax2.text(0.99, 0.97, "E-W", ha="right", va="top", transform=ax2.transAxes, fontsize=font)

        # ___________________________________________

        ax3.plot(t*tscale, r, color='k', alpha=0.8)
        ax3.fill_between(t*tscale, r, color='k', alpha=0.2)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel(f"norm. abs.\nGradient", fontsize=font)

        # ___________________________________________

        ax4.scatter(theta[r>rlim], r[r>rlim], c=t[r>rlim], s=1, zorder=2)
        ax4.set_theta_zero_location("N")
        ax4.set_rmin(0.2)
        ax4.set_rmax(1.05)
        ax4.set_rlabel_position(-122.5)
        ax4.grid(True)

        ax4.set_xticklabels(["N", "-45°", "-90°", "-135°", "180°", "135°", "90°", "45°"])

        ax3.set_xlabel(f"Time ({tunit}) since {self.tbeg.date} {str(self.tbeg.time)[:10]} UTC", fontsize=font)

        ax0.set_title(f"{self.fmin*1e3} - {self.fmax*1e3} mHz", fontsize=font)

        for _k, (ll, ax) in enumerate(zip(['(a)', '(b)', '(c)', '(d)', '(e)'], axes)):
            ax.text(0.005, 0.97, ll, ha="left", va="top", transform=ax.transAxes, fontsize=font+1)

        self.fig = fig

        if plot:
            plt.show();
        else:
            plt.close();

    def get_aperture(self, output=False):

        from numpy import argmax, array, sqrt, reshape
        import obspy

        dists, idx = [], []

        coords = self.coordinates

        for ii, ki in enumerate(coords.keys()):

            for jj, kj in enumerate(coords.keys()):

                lon1, lat1, height1 = coords[ki]['lon'], coords[ki]['lat'], coords[ki]['height']
                lon2, lat2, height2 = coords[kj]['lon'], coords[kj]['lat'], coords[kj]['height']

                # compute distances
                dist_x, dist_y = obspy.signal.util.util_geo_km(lon1, lat1, lon2, lat2)

                # convert unit from km to m
                dists.append(sqrt((dist_x*1000)**2+(dist_y*1000)**2))
                idx.append((ii, jj))

        # prepare matrix of distances
        _N = len(coords.keys())
        _dists = np.array(dists)
        self.distmatrix = _dists.reshape((_N, _N))

        self.aperture = round(max(abs(array(dists))), 2)

        print(f" -> Aperture of Array: {self.aperture} m")

        if output:
            return self.aperture

    def get_fband_of_array(self, apparent_velocity, output=False):

        if self.aperture is None:
            self.get_aperture()

        self.flower = round(0.01 * apparent_velocity / self.aperture, 4)
        self.fupper = round(0.25 * apparent_velocity / self.aperture, 4)

        if output:
            return self.flower, self.fupper

    def equalize_samples(self):

        # equalize number of samples
        npts_min = min([tr.stats.npts for tr in self.st])
        for tr in self.st:
            if tr.stats.npts < npts_min:
                npts_min
        for tr in self.st:
            diff = abs(tr.stats.npts - npts_min)
            if diff != 0:
                tr.data = tr.data[:-diff]

    def get_apparent_velocity(self, verbose=False):

        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import arange, roll, array, sqrt, nanmean, nanstd

        # shift traces to compute mean of array
        shifts = []
        dists = []
        pairs = []
        velos = []

        coords = self.coordinates

        for ii, ki in enumerate(coords.keys()):

            for jj, kj in enumerate(coords.keys()):

                lon1, lat1, height1 = coords[ki]['lon'], coords[ki]['lat'], coords[ki]['height']
                lon2, lat2, height2 = coords[kj]['lon'], coords[kj]['lat'], coords[kj]['height']

                # compute distances
                dist_x, dist_y = obspy.signal.util.util_geo_km(lon1, lat1, lon2, lat2)

                # convert unit from km to m
                dist = sqrt((dist_x*1000)**2+(dist_y*1000)**2)

                # cross-correlate
                arr0 = self.st.select(station=ki)[0].data
                arr1 = self.st.select(station=kj)[0].data

                dt = self.st.select(station=ki)[0].stats.delta
                Nshift = len(arr0)
                ccf = correlate(arr0, arr1, shift=Nshift, demean=False, normalize='naive', method='fft')

                # get shifts
                cclags = arange(-Nshift, Nshift+1) * dt

                # find maximum
                shift_max, value_max = xcorr_max(ccf)

                # assign time shift
                shifts.append(round(shift_max,1))

                # assign station pair
                pairs.append(f"{ki}_{kj}")

                # assign velocity
                velos.append(abs(round(dist/shift_max, 1)))

                # assign distance
                dists.append(round(dist, 1))

        self.velocities = velos
        self.distances = dists
        self.timeshifts = shifts

        self.mean_velocity = round(nanmean(velos), 0)
        self.std_velocity = round(nanstd(velos), 0)

        if self.verbose or verbose:
            for i in range(len(pairs)):
                print(pairs[i], f"shift: {shifts[i]}s", f"vel: {velos[i]}m/s")

            print(f"\nmean velocity: {self.mean_velocity} +- {self.std_velocity} m/s")

    def get_mean_pressure(self, plot=False):

        import matplotlib.pyplot as plt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import arange, roll, array

        # shift traces to compute mean of array
        shifted = []

        for i, seed in enumerate(self.seeds):

            sta = seed.split(".")[1]

            if i == 0:
                arr0 = self.st.select(station=sta)[0].data
                shifted.append(arr0)
                continue
            else:
                arr1 = self.st.select(station=sta)[0].data

            Nshift = len(arr0)

            dt = self.st[0].stats.delta

            ccf1 = correlate(arr0, arr1, shift=Nshift, demean=False, normalize='naive', method='fft')

            cclags = arange(-Nshift, Nshift+1) * dt

            shift1, value1 = xcorr_max(ccf1)

            if self.verbose:
                print(sta, f"shift: {round(shift1/60, 2)}min", f"CC: {round(value1, 2)}")

            arr1_shifted = roll(arr1, shift1)

            shifted.append(arr1_shifted)

            # compute mean
            _mean = array([])
            for i, arr in enumerate(shifted):
                if i == 0:
                    _mean = arr
                else:
                    _mean = _mean + arr

        mean = self.st[0].copy()
        mean.stats.station = "RMY"
        mean.stats.location = "00"
        mean.stats.channel = "LDO"
        mean.data = _mean/(i+1)

        self.st_mean = mean

        # checkup plot
        if plot:
            times = self.st[0].times()/3600
            fig = plt.figure(figsize=(15, 5))
            for i, x in enumerate(shifted):
                plt.plot(times, x, label=self.seeds[i], zorder=2)
            plt.plot(times, mean.data, "k", zorder=2)
            plt.legend()
            plt.grid(ls="--", color="grey", alpha=0.4)
            plt.ylabel("Pressure (Pa)", fontsize=12)
            plt.xlabel("Time (hour)", fontsize=12)
            plt.show();

    @staticmethod
    def read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED"):

        '''
        @params path_to_archive
        @params seed
        @params tbeg
        @params tend
        @params data_format

        DEPENDENCIES:
         - from obspy.core import UTCDateTime
         - from obspy.clients.filesystem.sds import Client

        OUTPUT:
         - stream

        EXAMPLE:
        >>> st = __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED")

        '''

        import os
        from obspy.core import UTCDateTime, Stream
        from obspy.clients.filesystem.sds import Client

        tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

        if not os.path.exists(path_to_archive):
            print(f" -> {path_to_archive} does not exist!")
            return

        # separate seed id
        net, sta, loc, cha = seed.split(".")

        # define SDS client
        client = Client(path_to_archive, sds_type='D', format=data_format)

        # read waveforms
        try:
            st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
        except:
            print(f" -> failed to obtain waveforms!")
            st = Stream()

        return st

    @staticmethod
    def interpolate_nan(array_like):

        from numpy import isnan, interp

        array = array_like.copy()

        nans = isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = interp(get_x(nans), get_x(~nans), array[~nans])

        return array

    @staticmethod
    def __adjust_time_line(st0, reference):

        from numpy import interp

        Rnet, Rsta, _, _ = reference.split(".")

        ref_start = st0.select(network=Rnet, station=Rsta)[0].stats.starttime
        ref_times = st0.select(network=Rnet, station=Rsta)[0].times()

        dt = st0.select(network=Rnet, station=Rsta)[0].stats.delta

        for tr in st0:
            times = tr.times(reftime=ref_start)

            tr.data = interp(ref_times, times, tr.data)
            tr.stats.starttime = ref_start

        return st0