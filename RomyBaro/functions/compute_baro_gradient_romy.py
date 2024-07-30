def __compute_baro_gradient_romy(tbeg, tend, status=False, excluded_stations=[], verbose=False):

    ######################
    """
    rotation_X = -u_nz
    rotation_Y =  u_ez
    rotation_Z = 0.5*(u_ne-u_en)
    """
    ######################

    import os
    import numpy as np
    import timeit
    import matplotlib.pyplot as plt
    import matplotlib.colors

    from obspy import UTCDateTime, Stream, read_inventory
    from obspy.clients import fdsn
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.geodetics import locations2degrees
    from obspy.clients.fdsn import Client, RoutingClient
    from obspy.signal import array_analysis as AA
    from obspy.signal.util import util_geo_km
    from obspy.signal.rotate import rotate2zne
    from datetime import datetime

    import warnings
    warnings.filterwarnings('ignore')

    if os.uname().nodename == 'lighthouse':
        root_path = '/home/andbro/'
        data_path = '/home/andbro/kilauea-data/'
        archive_path = '/home/andbro/freenas/'
        bay_path = '/home/andbro/ontap-ffb-bay200/'
        lamont_path = '/home/andbro/lamont/'
    elif os.uname().nodename == 'kilauea':
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/import/ontap-ffb-bay200/'
        lamont_path = '/lamont/'
    elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/import/ontap-ffb-bay200/'
        lamont_path = '/lamont/'

    # _____________________________________________________

    # start timer for runtime
    start_timer = timeit.default_timer()

    # _____________________________________________________

    # generate configuration object
    config = {}

    # convert to utcdatetime object
    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    # select the fdsn client for the stations
    # config['fdsn_client'] = {"BW":Client('http://jane'), "GR":Client('BGR')}
    config['fdsn_client'] = {"BW":Client('LMU'), "GR":Client('BGR')}

    # define output seed
    config['out_seed'] = "BW.BRMY"

    config['location'] = "00"

    # specify frequency range
    config['freq2'] = 0.01
    config['freq1'] = 0.0001
    config['apply_bandpass'] = True

    # decide if information is printed while running the code
    config['verbose'] = verbose

    # BRMY array information
    config['array_stations'] = [
                               'BW.PROMY.03.LDI',
                               'BW.GELB..LDO',
                               'BW.GRMB..LDO',
                               'BW.ALFT..LDO',
                               'BW.BIB..LDO',
                               'BW.TON..LDO',
                                ]

    # specify reference station
    config['reference_station'] = 'BW.PROMY.03.LDI'

    # exclude stations
    config['subarray_stations'] = [_sta for _sta in config['array_stations'] if _sta not in excluded_stations]

    # create subarray mask
    config['subarray_mask'] = range(len(config['subarray_stations']))

    # set coordinates
    config['coo'] = {  "ALFT":{"lon":11.2795 , "lat":48.142334, "height":593.0},
                       "GELB":{"lon":11.2514 , "lat":48.1629, "height":628.0},
                       "GRMB":{"lon":11.2635 , "lat":48.1406, "height":656.0},
                       "TON":{"lon":11.288809 , "lat":48.173897, "height":564.0},
                       "BIB":{"lon":11.2473 , "lat":48.1522, "height":599.0},
                       "PROMY":{"lon":11.275501 , "lat":48.162941, "height":571.0},
                      }

    # adr parameters
    config['vp'] = 1 # 5000 #6264. #1700
    config['vs'] = 1 # 3500 #3751. #1000
    config['sigmau'] = 1e-7 # 0.0001

    # _____________________________________________________

    def __interpolate_nan(array_like):

        from numpy import isnan, interp

        array = array_like.copy()

        nans = isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = interp(get_x(nans), get_x(~nans), array[~nans])

        return array

    def __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED"):

        '''
        VARIABLES:
         - path_to_archive
         - seed
         - tbeg, tend
         - data_format

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

        ## separate seed id
        net, sta, loc, cha = seed.split(".")

        ## define SDS client
        client = Client(path_to_archive, sds_type='D', format=data_format)

        ## read waveforms
        try:
            st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
        except:
            print(f" -> failed to obtain waveforms!")
            st = Stream()

        return st

    def __get_inventory_and_distances(config):

        coo = []
        for i, station in enumerate(config['subarray_stations']):

            net, sta, _, _ = station.split(".")

            loc, cha = "", "*HZ"

            # change PROMY
            if sta == "PROMY":
                sta, cha = "ROMY", "*JN"

            try:
                inven = config['fdsn_client'][net].get_stations(
                                                                network=net,
                                                                station=sta,
                                                                channel=cha,
                                                                location=loc,
                                                                starttime=config['tbeg'],
                                                                endtime=config['tend'],
                                                                level='response'
                                                                )
            except:
                return [], []
                print("fail")

            # get coordinates
            l_lon =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])

            print(l_lon, l_lat, height)

            if station == config['reference_station']:
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon, o_lat, l_lon, l_lat)

            # convert unit from km to m
            coo.append([lon*1000, lat*1000, height-o_height])

        return inven, np.array(coo)

    def __check_samples_in_stream(st, config):

        Rnet, Rsta, _, _ = config['reference_station'].split(".")

        Rsamples = st.select(network=Rnet, station=Rsta)[0].stats.npts

        for tr in st:
            if tr.stats.npts != Rsamples:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {Rsamples})")
                st.remove(tr)

        return st

    def __get_data(config):

        config['subarray'] = []

        st = Stream()

        for k, station in enumerate(config['subarray_stations']):

            net, sta, loc, cha = station.split(".")

            print(f" -> requesting {net}.{sta}.{loc}.{cha}") if config['verbose'] else None

            # try to get waveform data
            try:
                st00 = __read_sds(archive_path+"temp_archive/", station, config['tbeg'], config['tend'])
                # print(data)

            except Exception as E:
                print(E) if config['verbose'] else None
                print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...")
                continue

            # merge if masked
            if len(st00) > 1:
                print(f" -> merging stream. Length: {len(st00)} -> 1") if config['verbose'] else None
                st00 = st00.merge(method=1, fill_value="interpolate")

            # interpolate nan values
            for tr in st00:
                if np.isnan(tr.data).any():
                    print(" -> NaN found")
                    tr.data = __interpolate_nan(tr.data)

            # successfully obtained
            if len(st00) == 3:
                print(f" -> obtained: {net}.{sta}")

            # correect scaling for PROMY.03
            if sta == "PROMY" and loc == "03":
                for tr in st00:
                    tr.data /= 100

            # resampling using decitmate
            st00 = st00.detrend("linear");

            # assign reference station stream
            if station == config['reference_station']:
                # ref_station = stats.copy().resample(40, no_filter=False)
                ref_station = st00.copy()

            st += st00

            # create station list for obtained stations
            config['subarray'].append(f"{st00[0].stats.network}.{st00[0].stats.station}")

        if verbose:
            st.plot(equal_scale=False);

        # update subarray stations
        config['subarray_stations'] = config['subarray']

        print(f" -> obtained: {len(st)} of {len(config['subarray_stations'])} stations!") if config['verbose'] else None

        if len(st) == 0:
            return st, Stream(), config
        else:
            return st, ref_station, config

    def __compute_ADR(st, config, ref_station):

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
                print(" -> stream data could not be appended!")

        # make sure input is array type
        tse, tsn, tsz = np.array(tse), np.array(tsn), np.array(tsz)

        # define array for subarray stations with linear numbering
        substations = np.arange(len(config['subarray_stations']))

        try:
            result = AA.array_rotation_strain(substations,
                                              np.transpose(tse),
                                              np.transpose(tsn),
                                              np.transpose(tsz),
                                              config['vp'],
                                              config['vs'],
                                              config['dist'],
                                              config['sigmau'],
                                             )
        except Exception as E:
            print(E)
            print("\n -> failed to compute ADR...")
            return None

        # create rotation stream and add data
        out = Stream()
        out += ref_station.copy()
        out += ref_station.copy()
        out += ref_station.copy()

        out[0].data = result['ts_w3']
        out[1].data = result['ts_w2']
        out[2].data = result['ts_w1']

        out[0].stats.channel = 'BDZ'
        out[1].stats.channel = 'BDN'
        out[2].stats.channel = 'BDE'

        out[0].stats.station = config['out_seed'].split(".")[1]
        out[1].stats.station = config['out_seed'].split(".")[1]
        out[2].stats.station = config['out_seed'].split(".")[1]

        out[0].stats.network = config['out_seed'].split(".")[0]
        out[1].stats.network = config['out_seed'].split(".")[0]
        out[2].stats.network = config['out_seed'].split(".")[0]

        out[0].stats.location = config['location']
        out[1].stats.location = config['location']
        out[2].stats.location = config['location']

        out = out.detrend('linear')

        return out

    def __adjust_time_line(st0, reference="BW.ROMY"):

        Rnet, Rsta, _, _ = reference.split(".")

        ref_start = st0.select(network=Rnet, station=Rsta)[0].stats.starttime
        ref_times = st0.select(network=Rnet, station=Rsta)[0].times()

        dt = st0.select(network=Rnet, station=Rsta)[0].stats.delta

        for tr in st0:
            times = tr.times(reftime=ref_start)

            tr.data = np.interp(ref_times, times, tr.data)
            tr.stats.starttime = ref_start

        return st0

    # __________________________________________________________
    # MAIN

    # launch a times
    start_timer1 = timeit.default_timer()

    # status of stations loaded
    config['stations_loaded'] = np.ones(len(config['subarray_stations']))

    # request data for pfo array
    st, ref_station, config = __get_data(config)

    # get inventory and coordinates/distances
    # inv, config['coo'] = __get_inventory_and_distances(config)

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
    if config['apply_bandpass']:
        st = st.taper(0.02, type="cosine")
        st = st.filter('bandpass', freqmin=config['freq1'], freqmax=config['freq2'], corners=4, zerophase=True)
        print(f" -> bandpass: {config['freq1']} - {config['freq2']} Hz")

    # compute relataive distances
    dist = []
    for k in config['coo'].keys():

        coo = config['coo'][k]

        ref_sta = config['reference_station'].split(".")[1]

        # reference
        ref = config['coo'][ref_sta]
        ref_lon, ref_lat, ref_height = ref['lon'], ref['lat'], ref['height']

        # convert
        lon, lat = util_geo_km(ref_lon, ref_lat, coo['lon'], coo['lat'])

        # convert unit from km to m
        dist.append([lon*1000, lat*1000, coo['height']-ref_height])

    config['dist'] = np.array(dist)


    # plot station coordinates for check up
    if verbose:
        import matplotlib.pyplot as plt
        for station in config['subarray_stations']:
            print(station)
            _net, _sta = station.split(".")
            plt.scatter(config['coo'][_sta]['lon'], config['coo'][_sta]['lat'], label=_sta)
            plt.legend()
        plt.show();

    # check if enough stations for ADR are available otherwise continue
    if len(st) < 3*3:
        print(" -> not enough stations (< 3) for ADR computation!")
        return
    else:
        print(f" -> continue computing ADR for {int(len(st)/3)} of {len(config['subarray_mask'])} stations ...")

    # homogenize the time line
    # st = __adjust_time_line(st, reference=config['reference_station'])

    # trim to requested interval
    # st = st.trim(config['tbeg'], config['tend'])

    # check for same amount of samples
    __check_samples_in_stream(st, config)

    # compute array derived rotation (ADR)
    try:
        rot = __compute_ADR(st, config, ref_station)
    except Exception as e:
        print(e)
        # return None

    # get mean starttime
    # tstart = [tr.stats.starttime - tbeg for tr in st]
    # for tr in st:
    #     tr.stats.starttime = tbeg + np.mean(tstart)

    # trim to requested interval
    rot = rot.trim(config['tbeg'], config['tend'])

    # remove Z trace (not possible to compute for pressure)
    for tr in rot:
        if "Z" in tr.stats.channel:
            rot.remove(tr);

    # stop times
    stop_timer1 = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer1 - start_timer1)/60, 2)} minutes\n")

    return rot