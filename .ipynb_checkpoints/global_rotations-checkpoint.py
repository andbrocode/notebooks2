# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Global Rotations
#
# Small example of how to compute rotations on a global mesh.

# +
import salvus.namespace as sn
import os
import numpy as np
import obspy as obs

from obspy.clients.fdsn import Client, RoutingClient


# + code_folding=[0, 8]
def del_project(name):
    """ delete specified project file structure"""
    
    if name: 
        # !rm -rf $name 
        print(f"deleted: {name}")
        
def paraview_string(simulation_name, mode):
    """ print string for terminal to call paraview"""
    
    if mode is "mesh": 
        path = p.simulations.get_mesh_filenames(simulation_name)
        pwd = !pwd 
        alias_paraview =  "/home/brotzer/Downloads/ParaView-5.8.1-MPI-Linux-Python3.7-64bit/bin/paraview"
        path2=str(pwd[0])+"/"+str(path["xdmf_filename"])
        return f"{alias_paraview} --data={path2} &"
    else: 
        return "no mode set "
    


# -

d = sn.domain.dim3.SphericalGlobeDomain(radius_in_meter=6371000.0)

# ## Data downloading with ObsPy

# +
clmu = Client(base_url="LMU")
route = RoutingClient("eida-routing")

inv1 = route.get_stations(network="GR", station="FUR" )
inv2 = clmu.get_stations(network="BW", station="DROMY" )



# +
cusgs = Client("USGS")


cat = cusgs.get_events(
        minmagnitude=7.0,
        starttime=obs.UTCDateTime(2019, 5, 14),
        endtime=obs.UTCDateTime(2019, 5, 15),
        includeallorigins=True,
        )
print(cat)
# -

# ## Project Setup

# + code_folding=[]
project_name="global_rotations"

#del_project(project_name)

p = sn.Project.from_domain(
        path = project_name, 
        domain = d, 
        load_if_exists = True,
)
#print(f"created: {project_name}")


# +
# parse the retrieved sources 
sc = sn.simple_config.source.seismology.parse(
        filename_or_obj = cat, 
        dimensions = 3,
)
# parse the retrieved receiver
rc = sn.simple_config.receiver.seismology.parse(
        filename_or_obj = inv,
        dimensions = 3,
        fields = ["displacement", "velocity", "gradient-of-displacement"],
)

# add sources and receivers to the project
p.add_to_project(
        sn.Event(
#             event_name = "PNG"
            sources   = sc,
            receivers = rc,
    )
)

p.events.list()

# +
p.add_to_project(
    sn.SimulationConfiguration(
        name="simulation_01",
        min_period_in_seconds=400.0,
        elements_per_wavelength=1.0,
        tensor_order=2,
        
        # add model config
        model_configuration=sn.ModelConfiguration(
            background_model="prem_ani_no_crust",
        ),
        
        # add event config
        event_configuration=sn.EventConfiguration(
            wavelet=sn.simple_config.stf.Ricker(center_frequency = 0.5),
            waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                end_time_in_seconds=4000.0,
            ),
        ),
    ), overwrite=True
)

p.viz.nb.simulation_setup(
    simulation_configuration="simulation_01", 
    events=p.events.list()
)
# -

p.simulations.launch(
    simulation_configuration="simulation_01", 
    events=p.events.list(), 
    ranks_per_job=4, 
    site_name="salvus_local"
)
p.simulations.query(block=True)

# ## Data Analysis

# get string to open mesh in paraview
paraview_string("simulation_01", mode="mesh")

# +
wf = p.waveforms.get("simulation_01", events='event_NEW_BRITAIN_REGION_P.N.G._Mag_7.58_2019-05-14-12-58')

u = wf[0].get_receiver_data("BW.FURT", receiver_field="displacement")

tbeg, tend = u[0].stats.starttime, u[0].stats.endtime

god = wf[0].get_receiver_data("BW.FURT", receiver_field="gradient-of-displacement")
#print(god)

# -


# ### Compare translation

# +
v = wf[0].get_receiver_data("BW.FURT", receiver_field="velocity")

#client = Client(base_url="http://tarzan", timeout=100)
client = RoutingClient("eida-routing")

# tbeg = obs.UTCDateTime(2020,10,25)
# tend = obs.UTCDateTime(2020,10,26)

v_furt = client.get_waveforms(
                network="GR",
                station="FURT",
                location="00",
                channel="LHZ",
                starttime=tbeg,
                endtime=tend,
)

v_furt.remove_response(output="VEL")

v_furt.plot();

# +
# We need to rotate from xyz to rtp - this is a bit of a hack - we'll
# add proper support for this soon.
#
# This is already properly handled for displacement, velocity, and
# acceleration and we'll reuse the same matrix here.
rec = p.simulations.get_input_files("simulation_01", p.events.list()[0])[0][
    0
].output.point_data.receiver[0]
#print(rec)

# Get the rotation matrix.
R = np.array(rec.rotation_on_output.matrix)


# + code_folding=[0]
def get_rotation_rate(st, R): 
    """ 
    get the rotation rate from gradient-of-displacement data 
    
    st: stream of gradient-of-displacement
    R : rotation matrix to ZNE system
    
    dependencies: 
        -> import obspy    
    """

    # Assemble 3 x 3 gradient.
    data = np.zeros((3, 3, st[0].stats.npts), dtype=np.float32)
    for i, row in enumerate([["XX", "XY", "XZ"], ["YX", "YY", "YZ"], ["ZX", "ZY", "ZZ"]]):
        for j, channel in enumerate(row):
            data[i, j] = st.select(channel=f"X{channel}")[0].data

    # Tensor rotation.
    data_r = ((R @ data).T @ R.T).T

    # Rename.
    st_r = obs.Stream()

    #for i, row in enumerate([["ZZ", "ZN", "ZE"], ["NZ", "NN", "NE"], ["EZ", "EN", "EE"]]):
    for i, row in enumerate([["11", "12", "13"], ["21", "22", "23"], ["31", "32", "33"]]):
        for j, channel in enumerate(row):
            tr = st[0].copy()
            tr.stats.channel = channel
            tr.data = data_r[i, j]
            st_r += tr


    omega = obs.Stream()
    # Compute vertical rotation rate.
    for c, i, j in [["JZ","23","32"],["JN","31","13"],["JE","12","21"]]:
        tr_j = st[0].copy()
        tr_j.stats.channel = c
        tr_j.data = 0.5* (st_r.select(channel = i)[0].data - st_r.select(channel = j)[0])
        omega += tr_j
    
    return omega


omega = get_rotation_rate(st, R)

# We simulated with a Delta STF so we need to filter out frequencies
# that the mesh did not resolve.
omega.filter('lowpass', freq=1.0/400.0, corners=4, zerophase=True)
omega.plot();
# +
# initalize george 
george = Client(base_url='http://george')

JZ=george.get_waveforms(
    network="BW",
    station="ROMY",
    location="10",
    channel="BJZ",
    starttime=tbeg,
    endtime=tend,
)

JZ[0].detrend('linear').taper(0.1)
JZ[0].filter("bandpass", freqmin=1.0/800.0, freqmax=1.0/400.0, corners=4, zerophase=True)
JZ[0].plot();

# +
inv2 = clmu.get_stations(network="BW", minlongitude=0.0, maxlongitude=15.0)

inv2.plot();

# -


# + code_folding=[]
client = RoutingClient("eida-routing")
client.get_stations(network="BW", station="FFB1")
# -





