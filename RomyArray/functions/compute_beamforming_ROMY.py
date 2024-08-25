def __compute_beamforming_ROMY(tbeg, tend, submask="all", fmin=None, fmax=None, component="Z", bandpass=True, plot=False, reference_station=None):

    import os
    import numpy as np
    import timeit
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import scipy.stats as sts

    from obspy import UTCDateTime, Stream, read_inventory
    from obspy.clients import fdsn
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.geodetics import locations2degrees
    from obspy.clients.fdsn import Client, RoutingClient
    from obspy.signal import array_analysis as AA
    from obspy.signal.util import util_geo_km
    from obspy.signal.rotate import rotate2zne
    from obspy.core.util import AttribDict
    from obspy.imaging.cm import obspy_sequential
    from obspy.signal.invsim import corn_freq_2_paz
    from obspy.signal.array_analysis import array_processing
    from datetime import datetime
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    import warnings
    warnings.filterwarnings('ignore')

    from andbro__read_sds import __read_sds

    username = os.environ.get('USER')
    if username == "brotzer":
        bay_path = "/bay200/"
    elif username == "andbro":
        bay_path = "/home/andbro/bay200/"

    ## _____________________________________________________

    def __get_data(config):

        config['subarray'] = []

        st = Stream()

        for k, station in enumerate(config['subarray_stations']):

            net, sta, loc, cha = station.split(".")[0], station.split(".")[1], "", f"HH{config['component']}"

            # print(f"-> requesting {net}.{sta}.{loc}.{cha}")


            ## querry inventory data
            try:
                try:
                    # print(" -> loading inventory via archive")
                    # inventory = read_inventory(f"/home/{username}/Documents/ROMY/stationxml_ringlaser/dataless.seed.{net}_{sta}", format="SEED")
                    inventory = read_inventory(f"/home/{username}/Documents/ROMY/stationxml_ringlaser/station_{net}_{sta}", format="STATIONXML")

                except:
                    # print(" -> loading inventory via Client")
                    inventory = Client(config['fdsn_clients'][k]).get_stations(
                                                                                 network=net,
                                                                                 station=sta,
                                                                                 # channel=cha,
                                                                                 starttime=config['tbeg']-20,
                                                                                 endtime=config['tend']+20,
                                                                                 level="response"
                                                                                )
            except Exception as e:
                # print(e)
                print(f" -> {station}: Failed to load inventory!")
                inventory = None

            ## try to get waveform data
            try:
                try:
                    # print(" -> loading waveforms via Client")
                    stats = Client(config['fdsn_clients'][k]).get_waveforms(
                                                                            network=net,
                                                                            station=sta,
                                                                            location=loc,
                                                                            channel=cha,
                                                                            starttime=config['tbeg']-20,
                                                                            endtime=config['tend']+20,
                                                                            # attach_response=True
                                                                            )
                except Exception as e:
                    # print(" -> loading waveforms via archive")
                    stats = __read_sds(f"{bay_path}mseed_online/archive/", f"{net}.{sta}.{loc}.{cha}", config['tbeg']-20, config['tend']+20)

            except Exception as e:
                print(e)
                print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...")
                continue

            ## if empty
            if len(stats) == 0:
                print(f" -> stream empty!")
                continue

            ## merge if masked
            if len(stats) > 1:
                print(f" -> merging stream. Length: {len(stats)}")
                stats.merge(method=1, fill_value="interpolate")

            if stats[0].stats.sampling_rate != 20.:
                stats.detrend("demean")
                stats.filter("lowpass", freq=18, corners=4, zerophase=True)
                stats.decimate(2, no_filter=True)

            ## remove response [ACC -> m/s/s | VEL -> m/s | DISP -> m]
            try:
                stats = stats.remove_response(inventory=inventory, output="VEL", water_level=60)
            except:
                print(" -> failed to remove response")
                continue


            ## rotate to ZNE
            try:
                stats = stats.rotate(method="->ZNE", inventory=inventory)
            except:
                print(" -> failed to rotate to ZNE")
                continue


            l_lon =  float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inventory.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])


            # for c in ["Z", "N", "E"]:
            stats.select(station=sta, channel=f"*{config['component']}")[0].stats.coordinates = AttribDict({
                                                                                      'latitude': l_lat,
                                                                                      'elevation': height/1000,
                                                                                      'longitude': l_lon
                                                                                       })

            if station == config['reference_station']:
                ref_station = stats.copy()


        ## make sure all have same sampling rate for HH* only
        if cha[0] == "H":
            st = st.resample(20, no_filter=False)

        # print(st.__str__(extended=True))

        print(f" -> obtained: {int(len(st)/1)} of {len(config['subarray_stations'])} stations!")

        ## update subarray stations if data could not be requested for all stations
        if len(st) < 1*len(config['subarray_stations']):
            config['subarray_stations'] = [f"{tr.stats.network}.{tr.stats.station}" for tr in st]
            config['subarray_stations'] = list(set(config['subarray_stations']))

        if len(st) == 0:
            return st, config
        else:
            return st, config



    ## _____________________________________________________

    ## start timer for runtime
    start_timer = timeit.default_timer()


    ## generate configuration object
    config = {}

    ## time period of event
    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    ## add component
    config['component'] = component

    ## select the fdsn client for the stations


    ## select stations to consider:
    if submask == "inner":
        config['subarray_mask'] = [0,1,2,3]
        # config['freq1'] = 0.4
        # config['freq2'] = 3.7
    elif submask == "outer":
        config['subarray_mask'] = [0,4,5,6,7,8]
        # config['freq1'] = 0.04
        # config['freq2'] = 0.3
    else:
        config['subarray_mask'] = [0,1,2,3,4,5,6,7,8]


    ## decide if information is printed while running the code
    config['print_details'] = False

    ## apply bandpass to data
    config['apply_bandpass'] = bandpass

    ## _____________________
    ## PFO array information

    config['reference_station'] = 'GR.FUR'  # reference station

    config['array_stations'] = ['GR.FUR', 'BW.FFB1', 'BW.FFB2', 'BW.FFB3', 'BW.TON', 'BW.GELB', 'BW.BIB', 'BW.ALFT', 'BW.GRMB']
    config['fdsn_clients'] = ['BGR', 'LMU', 'LMU', 'LMU', 'LMU', 'LMU', 'LMU', 'LMU', 'LMU']

    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]
    config['fdsn_clients'] = [config['fdsn_clients'][i] for i in config['subarray_mask']]

#     config['sub_array_stations'] = seeds

#     if reference_station is not None:
#         config['reference_station'] = reference_station
#     else:
#         config['reference_station'] = seeds[0]

    ## ______________________________

    ## beamforming parameters
    config['slow_xmin'] = -0.5
    config['slow_xmax'] = 0.5
    config['slow_ymin'] = -0.5
    config['slow_ymax'] = 0.5
    config['slow_steps'] = 0.01

    config['win_length'] = 1/fmin # window length in seconds

    config['win_frac'] = 0.5  # fraction of window to use as steps

    config['freq_lower'] = fmin
    config['freq_upper'] = fmax
    config['prewhitening'] = 0  ## 0 or 1


    ## loading data
    st, config = __get_data(config)

    ## pre-pprocessing data
    st = st.detrend("demean")

    if config['apply_bandpass']:
        st = st.taper(0.1)
        st = st.filter("bandpass", freqmin=config['freq_lower'], freqmax=config['freq_upper'], corners=8, zerophase=True)

    ## add coordinates from inventories
    # st = __add_coordinates(st, config)

    ## select only one component
    st = st.select(channel=f"*{component}")

    st = st.trim(config['tbeg']-0.1, config['tend']+0.1)

    ## define parameters for beamforming
    kwargs = dict(

        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=config['slow_xmin'], slm_x=config['slow_xmax'],
        sll_y=config['slow_ymin'], slm_y=config['slow_ymax'],
        sl_s=config['slow_steps'],

        # sliding window properties
        win_len=config['win_length'], win_frac=config['win_frac'],

        # frequency properties
        frqlow=config['freq_lower'], frqhigh=config['freq_upper'], prewhiten=config['prewhitening'],

        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',

        ## time period
        stime=config['tbeg'], etime=config['tend'],
        # stime=st[0].stats.starttime, etime=st[0].stats.endtime,
    )

    ## perform beamforming
    out = array_processing(st, **kwargs)

    st = st.trim(config['tbeg'], config['tend'])


    ## stop times
    stop_timer = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer - start_timer)/60,2)} minutes")

    ## ______________________________
    ## Plotting

    if plot:

        ## PLOT 1 -----------------------------------
        labels = ['rel.power', 'abs.power', 'baz', 'slow']

        out[:, 3][out[:, 3] < 0.0] += 360

        xlocator = mdates.AutoDateLocator()

        fig1, ax = plt.subplots(5,1, figsize=(15,10))

        Tsec = config['tend']-config['tbeg']
        times = (out[:, 0]-out[:, 0][0]) / max(out[:, 0]-out[:, 0][0]) * Tsec

        for i, lab in enumerate(labels):
            # ax[i].scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
            # ax[i].scatter(times, out[:, i + 1], c=out[:, 1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
            ax[i].scatter(times, out[:, i + 1], c=out[:, 2], alpha=0.6, cmap=obspy_sequential)
            ax[i].set_ylabel(lab)
            # ax[i].set_xlim(out[0, 0], out[-1, 0])
            ax[i].set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
            ax[i].xaxis.set_major_locator(xlocator)
            ax[i].xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        ax[4].plot(st[0].times()/st[0].times()[-1]*out[:, 0][-1], st[0].data)
        ax[2].set_ylim(0, 360)
        ax[0].set_ylim(0, 1)

        fig1.autofmt_xdate()

        plt.show();



    ## PLOT 2 -----------------------------------
    cmap = obspy_sequential

    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360

    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 3, N2 + 1)


    # sum rel power in bins given by abins and sbins
    # hist2d, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    hist2d, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=abs_power)

    # transform to radian
    baz_edges = np.radians(baz_edges)

    if plot:

        # add polar and colorbar axes
        fig2 = plt.figure(figsize=(8, 8))

        cax = fig2.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig2.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])

        # circle through backazimuth
        for i, row in enumerate(hist2d):
            bars = ax.bar((i * dw) * np.ones(N2),
                          height=dh * np.ones(N2),
                          width=dw, bottom=dh * np.arange(N2),
                          color=cmap(row / hist2d.max()))

        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])

        # set slowness limits
        ax.set_ylim(0, config['slow_xmax'])
        [i.set_color('grey') for i in ax.get_yticklabels()]
        ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=hist2d.min(), vmax=hist2d.max()))

        plt.show();

    max_val = 0
    for i in range(hist2d.shape[0]):
        for j in range(hist2d.shape[1]):
            if hist2d[i,j] > max_val:
                max_val, slw_max, baz_max = hist2d[i,j], sbins[j], abins[i]

    ## prepare output
    baz = out[:, 3]
    baz[baz < 0.0] += 360

    ## compute statistics
    deltaa = 5
    angles = np.arange(0, 365, deltaa)
    angles2 = np.arange(0, 365, 1)

    baz_bf_no_nan = baz[~np.isnan(baz)]
    cc_bf_no_nan = out[:, 2][~np.isnan(out[:, 2])]

    hist = np.histogram(baz, bins=len(angles)-1, range=[min(angles), max(angles)], weights=out[:, 2], density=False)

    baz_bf_mean = round(np.average(baz_bf_no_nan, weights=cc_bf_no_nan), 0)
    baz_bf_std = np.sqrt(np.cov(baz_bf_no_nan, aweights=cc_bf_no_nan))

    kde = sts.gaussian_kde(baz_bf_no_nan, weights=cc_bf_no_nan)
    baz_bf_max = angles2[np.argmax(kde.pdf(angles2))]

    ## convert samples to time in seconds
    _t = out[:, 0] - out[:, 0][0]
    ttime = _t / _t[-1] * ( tend - tbeg )

    ## prepare output dictionary
    output = {}
    output['time'] = ttime
    output['rel_pwr'] = out[:, 1]
    output['abs_pwr'] = out[:, 2]
    output['baz'] = baz
    output['slow'] = out[:, 4]
    output['baz_max_count'] = max_val
    output['baz_max'] = baz_max
    output['slw_max'] = slw_max
    output['baz_bf_mean'] = baz_bf_mean
    output['baz_bf_max'] = baz_bf_max
    output['baz_bf_std'] = baz_bf_std

    output['num_stations_used'] = len(config['subarray_stations'])
    output['num_stations_array'] = len(config['array_stations'])

    if plot:
        output['fig1'] = fig1
        output['fig2'] = fig2

    return output