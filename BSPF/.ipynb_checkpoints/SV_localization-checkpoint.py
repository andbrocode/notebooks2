#! /usr/bin/env python

import math, sys
import numpy as np

from obspy import *
from obspy.signal.cross_correlation import correlate
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth
from obspy.clients.fdsn import Client as Client_fdsn

import matplotlib.pyplot as plt
import matplotlib as mpl

from twistpy.polarization import TimeFrequencyAnalysis6C, EstimatorConfiguration

client_f = Client_fdsn("http://GEORGE")
client_b = Client_fdsn("BGR")

# due to instrumental problems with the rotation sensors we need to apply some time shifts
# these are appeoximations!!!
dt_BS1 = 0.005
dt_BS2 = 0.03
dt_BGR = 0.025
dt_XB100 = 0.03
dt_XB101 = 0.025
dt_XB102 = 0.025
dt_IXBLU = -0.005
dt_ISAE = 0.025

time_shifts = {'BS1':dt_BS1, 
               'BS2':dt_BS2,
              'BGR': dt_BGR,
              'XB100':dt_XB100,
              'XB101':dt_XB101,
              'XB102':dt_XB102,
              'IXBLU':dt_IXBLU,
              'ISAE':dt_ISAE,
              }


################################ 
# Station names
sta_acc = ['XF.FB7']
sta_rot = ['XF.BS1']

################################
# sweep ID (from 1301 to 1339, last digit from 1 to 3)
sweep_id = ['EXPL3']


# get the sweep coordinates
cooS = []
daytimes = []
for sweepID in sweep_id:
    inv_s = client_f.get_stations(network='XF', location='', station=sweepID, level='channel')
    comm = inv_s[0][0].comments[0].value.split(',')
    day = comm[0]
    time = comm[5]
    #daytime = day+'T'+time # sweeps
    daytime = comm[6] # EXPL2 - 5
    #daytime = '2019-11-19T10:26:05' # test explosion
    daytimes.append(daytime)
    #cooS.append(inv_s.get_coordinates('XF.'+sweepID+'..'+'FNZ')) # sweeps
    cooS.append(inv_s.get_coordinates('XF.'+sweepID+'..'+'ELZ')) # EXPL2 - 5
    #cooS.append(inv_s.get_coordinates('XF.'+sweepID+'..'+'BXZ')) # test explosion

    
t_1 = UTCDateTime(daytimes[0]) - 5

# add some time before and after the relevant time span
t_add = 3.5
t0 = t_1-t_add

# the duration of the relevant time span
T = 10
_t = t_1 + T+t_add

t0_trim = t_1
t1_trim = _t-t_add

stS = []
#for sweepID in sweep_id:
#    st_s = client_f.get_waveforms(network='XF', location='', station=sweepID, channel='BXZ', starttime=t0, endtime=_t)
#    stS.append(st_s)

stR = []
stA = []
cooR = []
cooA = []

# get the rotation waveform
cha_rot = "HJ*"
ts = 0.0
for star in sta_rot:
    sta_r = star.split('.')[1]
    net_r = star.split('.')[0]
    if sta_r in ["BS1", "BS2", "BGR", "XB100", "XB101", "XB102", "XBLU", "ISEA"]:
        ts = time_shifts[sta_r]
    st_rot = client_f.get_waveforms(network=net_r, location='', station=sta_r, channel=cha_rot, starttime=t0+ts, endtime=_t+ts)
# handle BS1 and BS2:
    if sta_r in ["BS1", "BS2"]:
        _st = st_rot.copy()
        st_rot = Stream()
        for c in ["E", "N", "Z"]:
            comp = _st.select(channel="HJ"+c)
            if len(comp) > 0:
                comp = comp[-1]
                st_rot.append(comp)
# apply time shift:
    for tr in st_rot:
        tr.stats.starttime = tr.stats.starttime-ts

# get the meta data
    inv_rot = client_f.get_stations(network=net_r, location='', station=sta_r, starttime=t0+ts, endtime=_t+ts, level='response')

    st_rot.detrend("demean")
    st_rot.taper(0.1)
    st_rot.remove_sensitivity(inventory=inv_rot)
    #st_rot.resample(100) # only for EXLP2 and EXPLT!!!
    stR.append(st_rot)
    cooR.append(inv_rot.get_coordinates(net_r+'.'+sta_r+'..'+'HJZ'))


# get the translation waveform
cha_acc = "HH*"
for staa in sta_acc:
    sta_a = staa.split('.')[1]
    net_a = staa.split('.')[0]
    if net_a == 'XF':
        st_acc = client_f.get_waveforms(network=net_a, location='', station=sta_a, channel=cha_acc, starttime=t0, endtime=_t)
# get the meta data
        inv_acc = client_f.get_stations(network=net_a, location='', station=sta_a, channel=cha_acc, starttime=t0, endtime=_t, level='response')

    if net_a == 'GR':
        st_acc = client_b.get_waveforms(network=net_a, location='', station=sta_a, channel=cha_acc, starttime=t0, endtime=_t)
# get the meta data
        inv_acc = client_b.get_stations(network=net_a, location='', station=sta_a, channel=cha_acc, starttime=t0, endtime=_t, level='response')
    st_acc.detrend("demean")
    st_acc.taper(0.1)
    st_acc.remove_response(inventory=inv_acc, water_level=60, output='ACC')
    stA.append(st_acc)
    cooA.append(inv_acc.get_coordinates(net_a+'.'+sta_a+'..'+'HHZ'))

for i in range(len(cooR)):
    if cooR[i]['latitude'] != cooA[i]['latitude'] or cooR[i]['longitude'] != cooA[i]['longitude']:
        print(sta_rot[i], sta_acc[i])
        print(cooR[i]['latitude'], cooA[i]['latitude'])
        print(cooR[i]['longitude'], cooA[i]['longitude'])
        print('ERROR: translation sensor and rotation sensor are not co-located!')
        #sys.exit(1)
coo = cooR

dist, az_theo, baz_theo = gps2dist_azimuth(cooS[0]['latitude'], cooS[0]['longitude'], coo[0]['latitude'], coo[0]['longitude'])
print(az_theo, baz_theo)


waveform = st_acc + st_rot

#waveform.resample(100)

print(waveform)
waveform.trim(t_1, t_1+T)
waveform.plot(equal_scale=False)

waveform_fil = waveform.copy()
#waveform_fil.filter('lowpass', freq=90, zerophase=True, corners=6)
f_min = 1
f_max = 45
waveform_fil.filter('bandpass', freqmin=f_min, freqmax=f_max, zerophase=True, corners=6)

# set cross-correlation coefficient threshold for median and visualization
CC = 0.0

# Starting and ending time (in seconds)
t1 = 0 
t2 = T

# The length of the moving window and moving steps (in seconds)
win_len = 0.5
win_step = 0.3 
################################ 

dt = waveform[0].stats.delta
t3 = np.int32(t1/dt)
t4 = np.int32(t2/dt)
wins = np.arange(0,t2-t1,win_step)
num_windows = len(wins) 

result_baz = np.empty((num_windows,3))
result_corr = np.empty((num_windows,2))

baz = []
baz_std = []

for num_sta,sta in enumerate(sta_acc):

    baz_cal = []
    t = []
    corrbaz = []
    _baz_cal = []
    
    for i_win in wins:

        t1_1 = t1 + i_win
        t2_1 = t1_1 + win_len
        t3_1 = np.int32(t1_1/dt)  
        t4_1 = np.int32(t2_1/dt)         

        dataE_r = waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJE')[0].data[t3_1:t4_1]
        dataN_r = waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJN')[0].data[t3_1:t4_1]
                
        data1  = (np.zeros((t4_1-t3_1,2)))    
        data1  = (np.zeros((len(dataE_r),2)))    
        data1[:,0] = dataE_r
        data1[:,1] = dataN_r
        #data1[:,0] = waveform_fil.select(station=sta_rot[num_sta],channel='HJE')[0].data[t3_1:t4_1]
        #data1[:,1] = waveform_fil.select(station=sta_rot[num_sta],channel='HJN')[0].data[t3_1:t4_1]

        #relate north and east component of rotation rate to compute backzimuth
        C = np.cov(data1, rowvar=False)
        Cprime,Q = np.linalg.eigh(C,UPLO='U')
        loc = np.argsort(np.abs(Cprime))[::-1]
        Q = Q[:,loc]
        baz_tmp = -np.arctan((Q[1,0]/Q[0,0]))*180/np.pi 

        if baz_tmp <= 0:
            baz_tmp = baz_tmp + 180.

            
        #remove 180° ambiguity
        corrbazz = correlate(waveform_fil.select(station=sta_acc[num_sta].split('.')[1],channel='HHZ')[0].data[t3_1:t4_1],
                             rotate_ne_rt(waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJN')[0].data[t3_1:t4_1], 
                                          waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJE')[0].data[t3_1:t4_1], baz_tmp)[1][:],0)
        
        if (corrbazz[0] > 0):
            baz_tmp = (180. + baz_tmp)

        baz_cal.append(baz_tmp)
        t.append(t1_1+win_len/2)
        corrbaz.append(np.abs(corrbazz[0]))
    
    baz_cal = np.array(baz_cal)
    t = np.array(t)
    corrbaz = np.array(corrbaz)
    
    result_baz[:,num_sta+1] = baz_cal
    result_corr[:,num_sta] = corrbaz
    for i in range(len(baz_cal)):
        if corrbaz[i] >= CC:
            _baz_cal.append(baz_cal[i])
    _baz_cal = np.asarray(_baz_cal)
    baz.append(np.median(_baz_cal))
    baz_std.append(np.std(_baz_cal))
    print("Station: %s-%s: %s" %(sta_acc[num_sta].split('.')[1],sta_rot[num_sta].split('.')[1], np.median(baz_cal)))
result_baz[:,0] = t


plt.figure(figsize=(12,8))
#compute time axis
time0 = waveform_fil.select(station=sta_acc[num_sta].split('.')[1],channel='HHZ')[0].times()
time1 = waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJE')[0].times() 
time2 = np.linspace(t1+win_len/2,t2+win_len/2,num_windows)
ylim1 = np.max([np.abs(waveform_fil.select(station=sta_acc[num_sta].split('.')[1],channel='HHZ')[0].data)])
ylim2 = np.max([np.abs(waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJN')[0].data),\
                np.abs(waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJE')[0].data),
                np.abs(waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJZ')[0].data)])


title = 'SV: '+sweep_id[0]+', '+sta_acc[0]+'-'+sta_rot[0]+', filtered '+str(f_min)+' Hz to '+str(f_max)+' Hz'

# plot time series of Az
ax0 = plt.subplot2grid((3,50),(0,0), colspan=49)
ax0.plot(time0[:],waveform_fil.select(station=sta_acc[num_sta].split('.')[1],channel='HHZ')[0].data,'k',
         linewidth=0.5,label="Acceleration - Z")
ax0.set_title("STATION: %s-%s" %(sta_acc[num_sta],sta_rot[num_sta]))
ax0.set_xlim(t1,t2)
ax0.set_ylim(-ylim1, ylim1)
ax0.set_ylabel("Acceleration\n(m/s$^2$)")
ax0.legend(loc=1,ncol=2,prop={'size':10})
ax0.set_title(title)

# plot time series of Re and Rn
ax1 = plt.subplot2grid((3,50),(1,0), colspan=49,sharex=ax0)
ax1.plot(time1,waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJE')[0].data,'k',
         linewidth=0.5,label="Rotation Rate - E")
ax1.plot(time1,waveform_fil.select(station=sta_rot[num_sta].split('.')[1],channel='HJN')[0].data,'r',
         linewidth=0.5,label="Rotation Rate - N")
ax1.set_xlim(t1,t2)
ax1.set_ylim(-ylim2,ylim2)
ax1.set_ylabel("Rotational rate\n(rad/s)")
ax1.legend(loc=1,ncol=2,prop={'size':10})   


# add colorbar
fig = plt.subplot2grid((3,50),(2,49))
norm = mpl.colors.Normalize(vmin=0.0,vmax=1)
cb1 = mpl.colorbar.ColorbarBase(fig,cmap=plt.cm.RdYlGn_r,
                                norm=norm,orientation='vertical',label="CC coefficient")

# plot backazimuth from Re/Rn
ax2 = plt.subplot2grid((3,50),(2,0),colspan=49,sharex=ax0)
index = np.where(result_corr[:,num_sta] > CC)  
ax2.scatter(t[index],result_baz[index,num_sta+1],c=result_corr[index,num_sta],cmap=plt.cm.RdYlGn_r,
            vmin=0.0,vmax=1,marker='.',s=55,alpha=0.7,label='Estimated BAz from Re/Rn')
ax2.set_xlim(t1,t2)
ax2.set_ylim(0,360)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Back azimuth\n(°)")
ax2.hlines(y=baz_theo,xmin=t1+win_len,xmax=t2+win_len,linestyle=':',
           linewidth=2,color='royalblue',label='Theor. BAz')
#ax2.hlines(y=110,xmin=t1+win_len,xmax=t2+win_len,linestyle='--',
#           linewidth=2,color='royalblue',label='110°')
ax2.legend(loc=6,ncol=2,prop={'size':8})
ax2.grid(linestyle='--', linewidth=0.5)

plt.subplots_adjust(left=0.09,bottom=0.07,right=0.95,top=0.96,hspace=0.0)
figname = 'SV_'+sweep_id[0]+'_'+sta_acc[0]+'_'+sta_rot[0]+'_'+str(f_min)+'_'+str(f_max)+'.png'
#plt.savefig('./plots/'+figname, dpi=100)
plt.show()
