def __compute_adr_romy(tbeg, tend, submask='all', status=False, excluded_stations=[]):

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
        bay_path = '/home/andbro/bay200/'
    elif os.uname().nodename == 'kilauea':
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/bay200/'
    elif os.uname().nodename == 'lin-ffb-01':
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/bay200/'

    ## _____________________________________________________

    ## start timer for runtime
    start_timer = timeit.default_timer()


    ## _____________________________________________________

    ## generate configuration object
    config = {}

    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    ## select the fdsn client for the stations
    config['fdsn_client'] = {"BW":Client('http://jane'), "GR":Client('BGR')}

    ## define output seed
    config['out_seed'] = "BW.ADR"

    ## add submask
    config['submask'] = submask

    if config['submask'] == "inner":
        config['location'] = "01"
    elif config['submask'] == "outer":
        config['location'] = "02"
    elif config['submask'] == "all":
        config['location'] = "03"


    ## select stations to consider: 
    if submask is not None:

        if submask == "inner":
            config['subarray_mask'] = [0, 1, 2, 3]
            config['freq1'] = 0.2
            config['freq2'] = 1.0

        elif submask == "outer":
            config['subarray_mask'] = [0, 4, 5, 6, 7, 8]
            config['freq1'] = 0.01
            config['freq2'] = 0.1

        elif submask == "all":
            config['subarray_mask'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            config['freq1'] = 0.01
            config['freq2'] = 0.1



    ## decide if information is printed while running the code
    config['print_details'] = False

    ## _____________________
    ## PFO array information

    config['reference_station'] = 'GR.FUR' ## reference station

    config['array_stations'] = ['GR.FUR', 'BW.FFB1', 'BW.FFB2', 'BW.FFB3',
                                'BW.BIB', 'BW.TON', 'BW.GELB', 'BW.ALFT', 'BW.GRMB',
                               ]


#     config['misorientations'] =  [0, 0. ,-1.375 ,0.25 ,0.125 ,-0.6875 ,-0.625 ,-1.9375 ,0.375 
#                                   ,-6.5625 ,0.3125 ,-1.125 ,-2.5625 ,0.1875]

#     config['subarray_misorientation'] = [config['misorientations'][i] for i in config['subarray_mask']]

    ## select stations of subarray
    config['subarray_stations'] = []
    for i in config['subarray_mask']:
        if config['array_stations'][i] not in excluded_stations:
            config['subarray_stations'].append(config['array_stations'][i])

    config['subarray_sta'] = config['subarray_stations']

    ## ______________________________
    ## parameter for array-derivation

    #config['prefilt'] = (0.001, 0.01, 5, 10)
    config['apply_bandpass'] = False


    # adr parameters
    config['vp'] = 1 # 5000 #6264. #1700
    config['vs'] = 1 # 3500 #3751. #1000
    config['sigmau'] = 1e-7 # 0.0001


    ## _____________________________________________________


    def __get_inventory_and_distances(config):

        coo = []
        for i, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            loc, cha = "", "BH*"


            try:
                ## load local version
                inven = read_inventory(root_path+f"Documents/ROMY/stationxml_ringlaser/station_{net}_{sta}.xml")

            except:
                inven = config['fdsn_client'][net].get_stations(
                                                                network=net,
                                                                station=sta,
                                                                channel=cha,
                                                                location=loc,
                                                                starttime=config['tbeg'],
                                                                endtime=config['tend'],
                                                                level='response'
                                                                )

            l_lon =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])


            if sta == config['reference_station'].split(".")[1]:
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon, o_lat, l_lon, l_lat)

            coo.append([lon*1000, lat*1000, height-o_height])  ## convert unit from km to m

        return inven, np.array(coo)


    def __check_samples_in_stream(st, config):

        Rnet, Rsta = config['reference_station'].split(".")

        Rsamples = st.select(network=Rnet, station=Rsta)[0].stats.npts

        for tr in st:
            if tr.stats.npts != Rsamples:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {Rsamples})")
                st.remove(tr)

        return st


    def __get_data(config):


        config['subarray'] = []

        st = Stream()

        coo = []

        for k, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            loc, cha = "", "BH*"

            print(f" -> requesting {net}.{sta}.{loc}.{cha}") if config['print_details'] else None


            ## querry inventory data
            try:
                try:
                    ## load local version
                    inventory = read_inventory(root_path+f"Documents/ROMY/stationxml_ringlaser/station_{net}.{sta}.xml")
                    print(f" -> load local inventory")

                except:
                    if sta == "FUR":
                            inventory = read_inventory(root_path+f"Documents/ROMY/stationxml_ringlaser/station_{net}_{sta}.xml")
                    else:
                        inventory = config['fdsn_client'][net].get_stations(
                                                                            network=net,
                                                                            station=sta,
                                                                            location=loc,
                                                                            channel=cha,
                                                                            starttime=config['tbeg']-3600,
                                                                            endtime=config['tend']+3600,
                                                                            level="response"
                                                                            )
                    print(f" -> load jane inventory")

            except:
                print(f" -> {sta} Failed to load inventory!")
                inventory = None

            ## add coordinates
            l_lon =  float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])

            if sta == config['reference_station'].split(".")[1]:
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon, o_lat, l_lon, l_lat)

            coo.append([lon*1000, lat*1000, height-o_height])  ## convert unit from km to m


            ## try to get waveform data
            try:
                stats = config['fdsn_client'][net].get_waveforms(
                                                                network=net,
                                                                station=sta,
                                                                location=loc,
                                                                channel=cha,
                                                                starttime=config['tbeg']-3600,
                                                                endtime=config['tend']+3600,
                                                                attach_response=False,
                                                                )
            except Exception as E:
                print(E) if config['print_details'] else None
                print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...")
                config['stations_loaded'][k] = 0
                continue

            ## merge if masked
            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3") if config['print_details'] else None
                stats.merge(method=1, fill_value="interpolate")

            ## successfully obtained
            if len(stats) == 3:
                print(f" -> obtained: {net}.{sta}")


            ## remove response [VEL -> rad/s | DISP -> rad]
            stats = stats.remove_sensitivity(inventory)
            # stats = stats.remove_response(inventory, output="VEL", water_level=None, pre_filt=[0.005, 0.008, 8, 10])
            # stats = stats.remove_response(inventory, output="VEL", water_level=1)


            ## rotate to ZNE
            try:
                if len(stats.select(component="2")) > 0:
                    stats = stats.rotate(method='->ZNE', inventory=inventory, components=['Z23'])
                else:
                    stats = stats.rotate(method='->ZNE', inventory=inventory, components=['ZNE'])
            except:
                print(f" -> {sta} failed to rotate to ZNE")
                continue

            ## resampling using decitmate
            stats = stats.detrend("linear");
            stats = stats.detrend("simple");

            # stats = stats.taper(0.01);
            stats = stats.filter("highpass", freq=0.001, corners=4, zerophase=True);
            # stats = stats.filter("highpass", freq=0.001);



            if station == config['reference_station']:
                # ref_station = stats.copy().resample(40, no_filter=False)
                ref_station = stats.copy()

            st += stats
            config['subarray'].append(f"{stats[0].stats.network}.{stats[0].stats.station}")

            print(stats)

        ## trim to interval
        # stats.trim(config['tbeg'], config['tend'], nearest_sample=False)

        config['coo'] = np.array(coo)

        # st.plot(equal_scale=False);

        config['subarray_stations'] = config['subarray']

        print(f" -> obtained: {len(st)/3} of {len(config['subarray_stations'])} stations!") if config['print_details'] else None

        if len(st) == 0:
            return st, Stream(), config
        else:
            return st, ref_station, config


    def __compute_ADR(st, config, ref_station):

        ## prepare data arrays
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


        ## make sure input is array type
        tse, tsn, tsz = np.array(tse), np.array(tsn), np.array(tsz)

        ## define array for subarray stations with linear numbering
        substations = np.arange(len(config['subarray_stations']))

        try:
            result = AA.array_rotation_strain(substations,
                                              np.transpose(tse),
                                              np.transpose(tsn),
                                              np.transpose(tsz),
                                              config['vp'],
                                              config['vs'],
                                              config['coo'],
                                              config['sigmau'],
                                             )
        except Exception as E:
            print(E)
            print("\n -> failed to compute ADR...")
            return None

        ## create rotation stream and add data
        rotsa = ref_station.copy()

        rotsa[0].data = result['ts_w3']
        rotsa[1].data = result['ts_w2']
        rotsa[2].data = result['ts_w1']

        rotsa[0].stats.channel='BJZ'
        rotsa[1].stats.channel='BJN'
        rotsa[2].stats.channel='BJE'

        rotsa[0].stats.station=config['out_seed'].split(".")[1]
        rotsa[1].stats.station=config['out_seed'].split(".")[1]
        rotsa[2].stats.station=config['out_seed'].split(".")[1]

        rotsa[0].stats.network=config['out_seed'].split(".")[0]
        rotsa[1].stats.network=config['out_seed'].split(".")[0]
        rotsa[2].stats.network=config['out_seed'].split(".")[0]

        rotsa[0].stats.location=config['location']
        rotsa[1].stats.location=config['location']
        rotsa[2].stats.location=config['location']

        rotsa = rotsa.detrend('linear')

    #     gradient_ZNE = result['ts_ptilde'] #u1,1 u1,2 u1,3 u2,1 u2,2 u2,3
    #     u_ee=gradient_ZNE[:,0]
    #     u_en=gradient_ZNE[:,1]
    #     u_ez=gradient_ZNE[:,2]
    #     u_ne=gradient_ZNE[:,3]
    #     u_nn=gradient_ZNE[:,4]
    #     u_nz=gradient_ZNE[:,5]


        #(Gradient trace)
        #      Gradient = o_stats.copy()        #information of the central station
        #      Gradient.append(o_stats[0].copy())
        #      Gradient.append(o_stats[0].copy())
        #      Gradient.append(o_stats[0].copy())
        #      Gradient[0].data = u_ee
        #      Gradient[1].data = u_en
        #      Gradient[2].data = u_ez
        #      Gradient[3].data = u_ne
        #      Gradient[4].data = u_nn
        #      Gradient[5].data = u_nz
        #      Gradient[0].stats.channel='uee'
        #      Gradient[1].stats.channel='uen'
        #      Gradient[2].stats.channel='uez'
        #      Gradient[3].stats.channel='une'
        #      Gradient[4].stats.channel='unn'
        #      Gradient[5].stats.channel='unz'

        return rotsa


    def __adjust_time_line(st0, reference="GR.FUR"):

        Rnet, Rsta = reference.split(".")

        ref_start = st0.select(network=Rnet, station=Rsta)[0].stats.starttime
        ref_times = st0.select(network=Rnet, station=Rsta)[0].times()

        dt = st0.select(network=Rnet, station=Rsta)[0].stats.delta

        for tr in st0:
            times = tr.times(reftime=ref_start)

            tr.data = np.interp(ref_times, times, tr.data)
            tr.stats.starttime = ref_start

        return st0


    ## __________________________________________________________
    ## MAIN ##

    ## launch a times
    start_timer1 = timeit.default_timer()

    ## status of stations loaded
    config['stations_loaded'] = np.ones(len(config['subarray_stations']))

    ## request data for pfo array
    st, ref_station, config = __get_data(config)


    ## get inventory and coordinates/distances
    # inv, config['coo'] = __get_inventory_and_distances(config)

    ## processing
    st = st.detrend("linear")
    st = st.detrend("demean")

    ## bandpass filter
    if config['apply_bandpass']:
        st = st.taper(0.02)
        st = st.filter('bandpass', freqmin=config['freq1'], freqmax=config['freq2'], corners=4, zerophase=True)
        print(f" -> bandpass: {config['freq1']} - {config['freq2']} Hz")


    ## plot station coordinates for check up
    # import matplotlib.pyplot as plt
    # stas = []
    # for tr in st:
    #     if tr.stats.station not in stas:
    #         stas.append(tr.stats.station)
    # plt.figure()
    # for c, s in zip(config['coo'], stas):
    #     print(s, c)
    #     plt.scatter(c[0], c[1], label=s)
    #     plt.legend()


    ## check if enough stations for ADR are available otherwise continue
    if len(st) < 9:
        print(" -> not enough stations (< 3) for ADR computation!")
        return
    else:
        print(f" -> continue computing ADR for {int(len(st)/3)} of {len(config['subarray_mask'])} stations ...")

    ## homogenize the time line
    st = __adjust_time_line(st, reference=config['reference_station'])


    ## trim to requested interval
    # st = st.trim(config['tbeg'], config['tend'])

    ## check for same amount of samples
    __check_samples_in_stream(st, config)

    ## compute array derived rotation (ADR)
    try:
        rot = __compute_ADR(st, config, ref_station)
    except Exception as e:
        print(e)
        return None, None

    ## get mean starttime
    # tstart = [tr.stats.starttime - tbeg for tr in st]
    # for tr in rot:
    #     tr.stats.starttime = tbeg + np.mean(tstart)


    ## trim to requested interval
    rot = rot.trim(config['tbeg'], config['tend'])


    ## plot status of data retrieval for waveforms of array stations
    if status:

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

        ax.pcolormesh(np.array([config['stations_loaded'], np.ones(len(config['stations_loaded']))*0.5]).T, cmap=cmap, edgecolors="k", lw=0.5)

        ax.set_yticks(np.arange(0, len(config['subarray_sta']))+0.5, labels=config['subarray_sta'])

        # ax.set_xlabel("Event No.",fontsize=12)
        ax.set_xticks([])
        ax.set_xlim(0, 1)

        plt.show();


    ## stop times
    stop_timer1 = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer1 - start_timer1)/60, 2)} minutes\n")

    if status:
        return rot, config['stations_loaded']
    else:
        return rot, []