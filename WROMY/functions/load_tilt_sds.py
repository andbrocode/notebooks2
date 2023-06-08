def __load_tilt_sds(config, seed_id):
    
    from tqdm.notebook import tqdm
    from obspy.clients.filesystem.sds import Client
    
    net, sta, loc, cha = seed_id.split(".")
    
    tbeg, tend = config['tbeg'], config['tend']

    st0 = Client(config['datapath'], fileborder_samples=1000).get_waveforms(net, sta, loc, cha, tbeg, tend)
    
    if len(st0) > 3:
        print(" -> split, interpolate, merge ...")
        st0.split().merge(fill_value="interpolate")
#         st0.merge()
    
    return st0
