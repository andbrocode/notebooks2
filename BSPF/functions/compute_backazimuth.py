#!/bin/python3



def __compute_backazimuth(st_acc, st_rot, config, event=None, plot=True):
    
    """
    
    >>> out = __compute_backazimuth(ii_pfo, py_bspf, config, plot=True)

    """
    
    
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    from obspy import read, read_events, UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.signal.cross_correlation import xcorr
    from obspy.signal.rotate import rotate_ne_rt

    ## _______________________________    
    ## check config
    keywords=['tbeg','tend','station_latitude', 'station_longitude',
              'step', 'win_length_sec', 'overlap', 'eventtime']

    for key in keywords: 
        if key not in config.keys():
            print(f" -> {key} is missing in config!\n")
            
            
    ## _______________________________    
    ## Defaults
    if 'win_length_sec' not in config.keys():
        config['win_length_sec'] = .5    ## window length for correlation
    if 'step' not in config.keys():
        config['step'] = 1
    if 'overlap' not in config.keys():
        config['overlap'] = 25
    
    
    ## time period
    config['tbeg'], config['tend'] = UTCDateTime(config['tbeg']), UTCDateTime(config['tend'])

    ## _______________________________    
    ## streams
    ACC = st_acc.trim(config['tbeg'], config['tend'])
    ROT = st_rot.trim(config['tbeg'], config['tend']).select(channel="*Z")
    
    ## _______________________________    
    ## get event if not provided
    if not event:
        events = Client("USGS").get_events(starttime=config['eventtime']-20, endtime=config['eventtime']+20)
        if len(events) > 1:
            print(f" -> {len(events)} events found!!!")
            print(events)
            
    event = events[0]
    
    ## event location from event info
    config['source_latitude'] = event.origins[0].latitude
    config['source_longitude'] = event.origins[0].longitude
    
    print(event.event_descriptions[0]['type'], ': ',event.event_descriptions[0]['text'] + "\n")
    

    ## _______________________________        
    ## theoretical backazimuth and distance
    
    config['baz'] = gps2dist_azimuth(
                                    config['source_latitude'], config['source_longitude'], 
                                    config['station_latitude'], config['station_longitude'],
                                    )

    print('Epicentral distance [m]:       ',np.round(config['baz'][0],1))
    print('Theoretical azimuth [deg]:     ', np.round(config['baz'][1],1))
    print('Theoretical backazimuth [deg]: ', np.round(config['baz'][2],1))

    ## _______________________________    
    ## backazimuth estimation
    
    config['sampling_rate'] = int(ROT.select(channel="*Z")[0].stats.sampling_rate)
    
    config['num_windows'] = len(ROT.select(channel="*Z")[0]) // (int(config['sampling_rate'] * config['win_length_sec']))

    
    backas = np.linspace(0, 360 - config['step'], int(360 / config['step']))
   
    corrbaz = []
    
    ind = None
    
    for i_deg in range(0, len(backas)):
        
        for i_win in range(0, config['num_windows']):
            
            ## infer indices
            idx1 = int(config['sampling_rate'] * config['win_length_sec'] * i_win)
            idx2 = int(config['sampling_rate'] * config['win_length_sec'] * (i_win + 1))
            
            ## add overlap
            if i_win > 0 and i_win < config['num_windows']:
                idx1 = int(idx1 - config['overlap']/100 * config['win_length_sec'] * config['sampling_rate'])
                idx2 = int(idx2 + config['overlap']/100 * config['win_length_sec'] * config['sampling_rate'])
                    
            ## rotate NE to RT   
            R, T = rotate_ne_rt(ACC.select(channel='*N')[0].data, 
                                ACC.select(channel='*E')[0].data,
                                backas[i_deg]
                               )
            
            ## compute correlation for backazimuth
            corrbaz0 = xcorr(ROT.select(channel="*Z")[0][idx1:idx2], 
                             T[idx1:idx2], 0,
                              )
            
            corrbaz.append(corrbaz0[1])

            
    corrbaz = np.asarray(corrbaz)
    corrbaz = corrbaz.reshape(len(backas), config['num_windows'])


    ## extract maxima
    maxcorr = np.array([backas[corrbaz[:, l1].argmax()] for l1 in range(0, config['num_windows'])])

    ## create mesh grid
    mesh = np.meshgrid(np.arange(config['win_length_sec']/2, config['win_length_sec'] * config['num_windows'], config['win_length_sec']), backas)

    

    ## _______________________________
    ## Plotting
    def __makeplot():
    
        ## get rotated acceleration
        R, T = rotate_ne_rt(ACC.select(component='N')[0].data, 
                            ACC.select(component='E')[0].data,
                            config['baz'][2]
                           )

        fig, ax = plt.subplots(3, 1, figsize=(15, 10))

        ## parameters
        font = 14
        acc_scaling, acc_unit = 1e3, "mm/s$^2$"
        rot_scaling, rot_unit = 1e6, "$\mu$rad/s"

        ## create time axis
        time = np.linspace(0, len(ACC[0].data)/ACC[0].stats.sampling_rate, len(ACC[0].data))

        ## plot vertical rotation rate
        ax[0].plot(time, ROT.select(channel="*Z")[0].data*rot_scaling, label='vertical rotation rate')

        ax[0].set_xlim(time[0], time[-1])
        ax[0].set_ylabel(f'vert. rot. rate \n({rot_unit})', fontsize=font)
        ax[0].legend()

        # add P- and S-wave arrivals
        ROT_max = max(ROT.select(channel="*Z")[0].data*rot_scaling)
        
        ## plot transverse acceleration
        ax[1].plot(time, T*acc_scaling, 'k',label='transverse acceleration')
        ax[1].set_xlim(time[0], time[-1])
        ax[1].set_ylabel(f'transv. acc. \n({acc_unit})', fontsize=font)
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax[1].legend()
        

        ## backazimuth estimation plot
        im = ax[2].pcolormesh(mesh[0], mesh[1], corrbaz, cmap=plt.cm.RdYlGn_r, vmin=-1, vmax=1, shading="auto")
        ax[2].set_xlim(time[0], time[-1])
        ax[2].set_ylim(0, 360)
        ax[2].set_ylabel(u'estimated \nbackazimuth (°)', fontsize=font)
        ax[2].set_xlabel('time (s)', fontsize=font)

        ## plot maximal correclation values
        ax[2].plot(np.arange(config['win_length_sec']/2., config['win_length_sec'] * len(maxcorr), config['win_length_sec']), maxcorr, '.k')

        ## plot theoretical Backazimuth for comparison
        xx = np.arange(0, config['win_length_sec'] * len(maxcorr) + 1, config['win_length_sec'])
        tba = np.ones(len(xx)) * config['baz'][2]
        if config['baz'][2] < 330:
            x_text, y_text = time[int(0.82*len(time))], config['baz'][2]+5
        else:
            x_text, y_text = time[int(0.82*len(time))], config['baz'][2]-15

        ax[2].plot(xx, tba, c='.5', lw=1.5, alpha=0.6)
        ax[2].text(x_text, y_text, u'Theor. BAz = '+str(round(config['baz'][2],2))+'°', color='k', fontsize=font-1)

        edist = round(config['baz'][0]/1000,1)
        
        if 'fmin' in config.keys() and 'fmax' in config.keys():
            ax[0].set_title(config['title'] +f" | {edist} km" + f" | {config['fmin']}-{config['fmax']} Hz", pad=15, fontsize=font)
        else:
            ax[0].set_title(config['title'] +f" | {edist} km", pad=15, fontsize=font)
 
            
        ## add colorbar
    #     norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    #     cb1 = mpl.colorbar.ColorbarBase(fig, cmap=plt.cm.RdYlGn_r, norm=norm, orientation='vertical')
        cax = ax[2].inset_axes([1.01, 0., 0.02, 1])
        cb1 = plt.colorbar(im, ax=ax[2], cax=cax)

        plt.show();
        return fig
    
    
    if plot:
        __makeplot();
        

    ## _______________________________
    ## prepare output        
    
    output = {}
    
    output['baz_mesh'] = mesh
    output['baz_corr'] = corrbaz
    output['baz_theo'] = config['baz']
    output['acc_transverse'] = T
    output['acc_radial'] = R
    output['rot_vertical'] = ROT.select(channel="*Z")
    output['event'] = event
    
    return output
        
## End of File