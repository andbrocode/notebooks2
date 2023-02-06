#!/usr/bin/env python
# coding: utf-8


def __querry_seismo_data(seed_id=None, beg=None, end=None, restitute=True):
    '''
    querry seismic traces and station data
    
    Dependencies: 
        from obspy.clients.fdsn import Client, RoutingClient
        
    example:
    	>>> fur, fur_inv = getMeMyData("GR.FUR..BH*", tbeg, tend, restitute=True)
    ''' 
    
    from obspy.clients.fdsn import Client, RoutingClient
      
    ## split seed_id string 
    net, sta, loc, cha = seed_id.split(".")
    
    ## check if input variables are as expected
    for arg in [net, sta, loc, cha, beg, end]:
        if arg is None and not 'loc':
            raise NameError(print(f"\nwell, {arg} has not been defined after all!"))
            exit()
            
    ## state which data is requested        
    if loc != None:
        print(f'\nGet data:  {net}.{sta}.{loc}.{cha} ({float(end-beg)/60}) ...')
    else: 
        print(f'\nGet data:  {net}.{sta}..{cha} (traces of {float(end-beg)/60} min duration) ...')
        
    ## attempting to get data from either EIDA or IRIS.
    try: 
        route = RoutingClient("eida-routing")
        print("\nUsing Eida-Routing...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")

            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)

    except: 
        route = RoutingClient("iris-federator")
        print("\nUsing Iris-Federator...")
        if route:
            inv = route.get_stations(network=net, station=sta, location=loc, channel=cha,
                                     starttime=beg, endtime=end, level="response")
            print(f"\nRequesting from {len(inv.get_contents()['networks'])} network(s) and {len(inv.get_contents()['stations'])} stations")
        
            st = route.get_waveforms(network=net, station=sta, location=loc, channel=cha, 
                                     starttime=beg, endtime=end)
    
    ## remove response of instrument specified in inventory
    if restitute:
               
#        pre_filt = [0.001, 0.005, 45, 50]
        
        out="VEL"  # "DISP" "ACC"
        
        st.remove_response(
            inventory=inv, 
#             pre_filt=pre_fit,
            output=out
        )
        
        print(f"\nremoving response ...")
        print(f"\noutput {out}")
        
    print('\nFinished\n_______________\n')
    
    return st, inv