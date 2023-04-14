#!/usr/bin/env python
# coding: utf-8

######################
"""
1. demonstration of Array-derived-rotation
2. Data source from IRIS PFO array (http://www.fdsn.org/networks/detail/PY/)
3. more detail refer to https://doi.org/10.1785/0220160216
4. relationship between rotation and gradient

rotation_X = -u_nz
rotation_Y =  u_ez
rotation_Z = 0.5*(u_ne-u_en)
"""
######################


def __compute_adr_pfo(tbeg, tend):
    
    import os
    import numpy as np
    import timeit

    from obspy import UTCDateTime, Stream
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

    ## _____________________________________________________

    ## start timer for runtime
    start_timer = timeit.default_timer()


    ## _____________________________________________________

    ## generate configuration object
    config = {}

    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    ## select the fdsn client for the stations
    config['fdsn_client'] = Client('IRIS')

    ## select stations to consider: 
    ## all: [0,1,2,3,4,5,6,7,8,9,10,11,12] | optimal: [0,5,8,9,10,11,12] | inner: [0,1,2,3]
    config['subarray_mask'] = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    ## select referenc station (usually central station)
    config['reference_station'] = 'BPH01'
    
    ## decide if information is printed while running the code
    config['print_details'] = False
    
    ## _____________________
    ## PFO array information
    config['network'] = 'PY'

    config['array_stations'] = ['BPH01','BPH02','BPH03','BPH04','BPH05','BPH06','BPH07',
                                'BPH08','BPH09','BPH10','BPH11','BPH12','BPH13']

    config['misorientations'] =  [0. ,-1.375 ,0.25 ,0.125 ,-0.6875 ,-0.625 ,-1.9375 ,0.375 
                                  ,-6.5625 ,0.3125 ,-1.125 ,-2.5625 ,0.1875]


    config['subarray_misorientation'] = [config['misorientations'][i] for i in config['subarray_mask']]
    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]

#     config['subarray'] = np.arange(len(config['array_stations']))

    ## ______________________________
    ## parameter for array-derivation

    #config['prefilt'] = (0.001, 0.01, 5, 10)
    config['freq1'] = 0.014   #0.014 for Spudich    and  0.073 for Langston
    config['freq2'] = 20.0 # 1.5
    config['apply_bandpass'] = True


    # adr parameters
    config['vp'] = 6264. #1700
    config['vs'] = 3751. #1000
    config['sigmau'] = 1e-8 # 0.0001


    ## _____________________________________________________


    def __get_inventory_and_distances(config):

        coo = []
        for i, station in enumerate(config['subarray_stations']):

            inven = config['fdsn_client'].get_stations(network=config['network'],
                                                       station=station,
                                                       channel='BHZ',
                                                       starttime=config['tbeg'],
                                                       endtime=config['tend'],
                                                       level='response'
                                                      )

            l_lon =  float(inven.get_coordinates('%s.%s..BHZ'%(config['network'],station))['longitude'])
            l_lat =  float(inven.get_coordinates('%s.%s..BHZ'%(config['network'],station))['latitude'])
            height = float(inven.get_coordinates('%s.%s..BHZ'%(config['network'],station))['elevation'])

            if i == 0:
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon,o_lat,l_lon,l_lat)
            coo.append([lon*1000,lat*1000,height-o_height])  #convert unit from km to m

        return inven, np.array(coo)


    def __check_samples_in_stream(st, config):

        for tr in st:
            if tr.stats.npts != config['samples']:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {config['samples']})")
                st.remove(tr)

        return st


    def __get_data(config):


        config['subarray'] = []

        st = Stream()

        for k, station in enumerate(config['subarray_stations']):

            print(f" -> requesting {config['network']}.{station}..BH*") if config['print_details'] else None

            ## try to get waveform data
            try:
                stats = config['fdsn_client'].get_waveforms(
                                                            network=config['network'],
                                                            station=station,
                                                            location='',
                                                            channel='BH*',
                                                            starttime=config['tbeg']-10,
                                                            endtime=config['tend']+10,
                                                            attach_response=True,
                                                            )
            except Exception as E:
                print(E)
                print(f" -> geting waveforms failed ofr {station}...")
                continue


            ## try to get inventory
    #         try:
    #             inv = config['fdsn_client'].get_stations(  
    #                                                     network=config['network'],
    #                                                     station=station,
    #                                                     location='',
    #                                                     channel='BHZ',
    #                                                     starttime=config['tbeg'],
    #                                                     endtime=config['tend'],
    #                                                     level='response'
    #                                                     )

    #         except Exception as E:
    #             print(E)
    #             print(f" -> geting inventory failed ofr {station}...")
    #             continue


            ## merge if masked 
            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3")
                stats.merge(method=1, fill_value="interpolate")


            ## sorting
            stats.sort()
            stats.reverse()

            #correct mis-alignment
            stats[0].data, stats[1].data, stats[2].data = rotate2zne(stats[0],0,-90,
                                                                     stats[1],
                                                                     config['subarray_misorientation'][config['subarray_stations'].index(station)],0, 
                                                                     stats[2],90+config['subarray_misorientation'][config['subarray_stations'].index(station)],0)

            ## remove response [VEL -> rad/s | DISP -> rad]
#             stats.remove_response(inventory=inv, output="VEL")
            stats.remove_response(output="VEL")

            ## trim to interval
#             stats.trim(config['tbeg'], config['tend'], nearest_sample=False)


            if station == config['reference_station']:
                ref_station = stats.copy()
                acc = stats.copy()
                acc.differentiate()


            st += stats


        print(st.__str__(extended=True)) if config['print_details'] else None

        ## update subarray stations if data could not be requested for all stations
        if len(st) < 3*len(config['subarray_stations']):
            config['subarray_stations'] = [tr.stats.station for tr in st]
            config['subarray_stations'] = list(set(config['subarray_stations']))
                           
                           
        print(f" -> obtained: {len(st)/3} of {len(config['subarray_stations'])} stations!") if config['print_details'] else None

        if len(st) == 0:
            return st, Stream(), config
        else:
            return st, ref_station, config


    def __compute_ADR(tse, tsn, tsz, config, ref_station):

        tse, tsn, tsz = np.array(tse), np.array(tsn), np.array(tsz)

        print(' ADR is executing...') if config['print_details'] else None

        try:
            result = AA.array_rotation_strain(np.arange(len(config['subarray_stations'])),
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


        ## create rotation stream and add data    
        rotsa = ref_station.copy()

        rotsa[0].data = result['ts_w3']
        rotsa[1].data = result['ts_w2']
        rotsa[2].data = result['ts_w1']
        
        rotsa[0].stats.channel='BJZ'
        rotsa[1].stats.channel='BJN'
        rotsa[2].stats.channel='BJE'
        
        rotsa[0].stats.station='RPFO'
        rotsa[1].stats.station='RPFO'
        rotsa[2].stats.station='RPFO'

        rotsa = rotsa.detrend('simple')


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


    ## MAIN ##

    ## launch a times
    start_timer1 = timeit.default_timer()

    ## request data for pfo array
    st, ref_station, config = __get_data(config)


    ## check if enough stations for ADR are available otherwise continue
    if len(st) < 9:
        print(" -> not enough stations (< 3) for ADR computation!")
        return
    else:
        print(f" -> continue computing ADR for {int(len(st)/3)} stations ...")

    ## get inventory and coordinates/distances
    inv, config['coo'] = __get_inventory_and_distances(config)

    ## basic processing
#     st.resample(sampling_rate=20)

    if config['apply_bandpass']:
        st.filter('bandpass', freqmin=config['freq1'], freqmax=config['freq2'], corners=4, zerophase=True)

    print(ref_station)

    ## prepare data arrays
    tsz, tsn, tse = [],[],[]
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


    ## compute array derived rotation (ADR)
    rot = __compute_ADR(tse, tsn, tsz, config, ref_station)

    ## trim to requested interval
    rot.trim(config['tbeg'], config['tend'], nearest_sample=False)

    
    ## stop times      
    stop_timer1 = timeit.default_timer()
    print(f"\n Runtime: {round((stop_timer1 - start_timer1)/60,2)} minutes")


    return rot
        
## End of File
