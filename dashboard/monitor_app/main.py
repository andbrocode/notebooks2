#!/usr/bin/env python
# coding: utf-8



import obspy as obs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
import panel as pn

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource,  Div, RangeSlider, Spinner
from bokeh.layouts import column, row
from streamz.dataframe import PeriodicDataFrame

pn.extension()





config = {}

## set path for figures
config['outpath_figs'] = "/home/brotzer/Desktop/tmp/"

# config['tbeg'] = obs.UTCDateTime("2022-05-04 09:48")
# config['tend'] = obs.UTCDateTime("2022-05-04 09:51")

config['real_time_delay']  = 60 ## minutes
config['period_to_show'] = 5 ## minutes

now = obs.UTCDateTime.now()
config['tbeg'] = now - config['real_time_delay']*60 - config['period_to_show']*60
config['tend'] = now - config['real_time_delay']*60 

config['update_interval'] = 5 ## seconds

## set stations
config['seeds'] = ["BW.RLAS..BJZ", "BW.ROMY.10.BJZ", "BW.ROMY..BJU", "BW.ROMY..BJV", "BW.ROMY..BJW",]

config['repository'] = "george"




def __getStream(config, restitute=True):
    """
    
    CONFIG:     config['seeds'] list of seed names
                config['tbeg'] startime as UTCDateTime
                config['tend'] endtime as UTCDateTime
                config['repository'] data repository to call [e.g. george, archive, jane,online]


    st = __getStream(config, restitute=True)

    """

    from andbro__querrySeismoData import __querrySeismoData
    from andbro__empty_trace import __empty_trace
    from obspy import Stream
    from numpy import nan
    
    st = Stream()


    for seed in config['seeds']:
        
        net, sta, loc, cha = seed.split(".")
        
#         print(f"loading {seed}...")
        
        try:
            st0, inv0 = __querrySeismoData(  
                                            seed_id=seed,
                                            starttime=config.get("tbeg")-.5,
                                            endtime=config.get("tend")+.5,
                                            repository=config.get("repository"),
                                            path=None,
                                            restitute=False,
                                            detail=None,
                                            fill_value=nan,
                                            )
            
            if len(st0) == 1:
                st += st0
            elif len(st0) > 1:
#                 print(" -> merging stream...")
                st += st0.merge(method=1,fill_value="latest")
    
            if restitute:
                if cha[-2] == "J":
#                     print(" -> removing sensitivity...")
                    st0.remove_sensitivity(inv0)
                elif cha[-2] == "H":
#                     print(" -> removing response...")
                    st0.remove_response(inventory=inv0, output="VEL", zero_mean=True)

        except:
            print(f" -> failed to load {seed}!")
            print(f" -> substituted {seed} with NaN values! ")
            st_empty = Stream()
            st_empty.append(__empty_trace(config, seed))
            st += st_empty
    
    
#     print("\ncompleted loading")

#     print(" -> trimming stream...")
    st.trim(config['tbeg'], config['tend'])
          
 #     print(" -> demean stream...")   
    st.detrend('demean')
    
    return st



def __update_data(config):
    
    config['tbeg'] = config['tbeg'] + config['update_interval']
    config['tend'] = config['tend'] + config['update_interval']
    
    st = __getStream(config)
    
    df = pd.DataFrame()
    
    df['timeline'] = st[0].times()
#     df['timeline'] = pd.to_datetime(st[0].times(("utcdatetime")).astype(str), 
#                                     utc=True, 
#                                     format='%Y-%m-%dT%H:%M:%S',
#                                     origin="unix")
    
    try:
        for seed in config['seeds']:
            df[seed] = st.select(id=seed)[0].data
    except:
        print("failure")

    return df




def __update_data2(st, config):
    
    tbeg_old = config['tbeg']
        
    config['tbeg'] = config['tend']
    config['tend'] = config['tend'] + config['update_interval']
    
    st_new = __getStream(config)
    
    st += st_new
    st.merge(method=1)
    
    config['tbeg'] = tbeg_old+config['update_interval']
    
    st.trim(config['tbeg'], config['tend'])
    
    df = pd.DataFrame()
    
    df['timeline'] = st[0].times("utcdatetime")
#     df['timeline'] = pd.to_datetime(st[0].times(("utcdatetime")).astype(str), 
#                                     utc=True, 
#                                     format='%Y-%m-%dT%H:%M:%S',
#                                     origin="unix")
    
    try:
        for seed in config['seeds']:
            df[seed] = st.select(id=seed)[0].data
    except:
        print("failure")

    return df, st, config




def __create_streaming_panel(st, seed_id):
    
    net, sta, loc, cha = seed_id.split('.')

    panel = figure(width=800, height=200, title=f'{sta}.{cha}')

    select_st = st.select(station=sta, channel=cha)

    panel_cds = ColumnDataSource(data={'x': select_st[0].times(), 'y': select_st[0].data})

    panel.line('x', 'y', source=panel_cds)

    return panel, panel_cds


def __get_oszi_image():

    
    x_range = (-10,-10)
    y_range = (20,30)
    oszi = figure(x_range=x_range, y_range=y_range)
    img_path = 'monitor_app/static/OSZI.png'
    oszi.image_url(url=[img_path],x=x_range[0],y=y_range[1],w=x_range[1]-x_range[0],h=y_range[1]-y_range[0])
#     doc = curdoc()
#     doc.add_root(oszi)
    oszi_cds = ColumnDataSource(data={'x': [], 'y': []})

#     oszi = Div(width=150, height=150, text="""<img src="/home/brotzer/Downloads/OSZI_north_ring.png" alt="oszi_image">""",)
    return oszi, oszi_cds        



## get initial data stream
st = __getStream(config, restitute=True)


## make panels
panel_1, panel_1_cds = __create_streaming_panel(st, "BW.RLAS..BJZ")

panel_2, panel_2_cds = __create_streaming_panel(st, "BW.ROMY.10.BJZ")

panel_3, panel_3_cds = __create_streaming_panel(st, "BW.ROMY..BJU")

panel_4, panel_4_cds = __create_streaming_panel(st, "BW.ROMY..BJV")

panel_5, panel_5_cds = __create_streaming_panel(st, "BW.ROMY..BJW")


oszi = Div(width=250, height=250, text="""<img src="monitor_app/static/OSZI.png" alt="oszi_image">""",)
curdoc().add_root(oszi)


## set up time 
# time = Div(text=f""" <p>{str(obs.UTCDateTime.now())}</p> """, width=200, height=30,)



def __streaming():

    global config
    global st
    
    df = __update_data(config)
#     df, st, config = __update_data2(st, config)
        
    panel_1_cds.update(data={'x': df['timeline'], 'y': df["BW.RLAS..BJZ"]})
    
    panel_2_cds.update(data={'x': df['timeline'], 'y': df["BW.ROMY.10.BJZ"]})
    
    panel_3_cds.update(data={'x': df['timeline'], 'y': df["BW.ROMY..BJU"]})

    panel_4_cds.update(data={'x': df['timeline'], 'y': df["BW.ROMY..BJV"]})
    
    panel_5_cds.update(data={'x': df['timeline'], 'y': df["BW.ROMY..BJW"]})

    oszi = __update_oszi()



#    pn.io.push_notebook(bk_dash) # Only needed when running in notebook context

    del df


def __update_oszi():
    doc = curdoc()
    doc.clear()
#     oszi = Div(width=250, height=250, text="""<img src="monitor_app/static/OSZI.png" alt="oszi_image">""",)
    oszi.text = """<img src="monitor_app/static/OSZI.png" alt="oszi_image">"""
    doc.add_root(oszi)
    show(doc)



bk_dash = pn.pane.Bokeh(row(column(panel_1, panel_2, panel_3, panel_4, panel_5),column(oszi))
                        )
# bk_dash = pn.Row(pn.Column(panel_1, panel_1))

## create a periodic callback 
interval = int(1e3*config['update_interval']) ## to miliseconds as int
pn.state.add_periodic_callback(__streaming, period=interval)


## make it executable 
bk_dash.servable(title='ROMY Monitor')
bk_dash.show(port=8898)

