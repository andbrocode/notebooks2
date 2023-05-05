#!/bin/python3

def __get_theoretical_backazimuth(config, fdsn_client="USGS"):
    
    from obspy.clients.fdsn import Client
    from obspy.geodetics.base import gps2dist_azimuth

    ## get event if not provided
    if 'event' not in config.keys():
        events = Client(fdsn_client).get_events(starttime=config['eventtime']-20, endtime=config['eventtime']+20)
        if len(events) > 1:
            print(f" -> {len(events)} events found!!!")
            print(events)
            
    event = events[0]
        
    ## event location from event info
    config['source_latitude'] = event.origins[0].latitude
    config['source_longitude'] = event.origins[0].longitude
    
    
    dist, az, baz = gps2dist_azimuth(
                                    config['source_latitude'], config['source_longitude'], 
                                    config['BSPF_lat'], config['BSPF_lon'],
                                    )
    

    return baz, az, dist

## End of File