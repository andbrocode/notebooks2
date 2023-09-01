#! /usr/bin/env python
import sys, os
import math as M
import numpy as np
import scipy as sp

from obspy import *
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from matplotlib import rc, font_manager
import matplotlib.cm as cm
import datetime as dt
import matplotlib.dates as mdates

from scipy.odr import *
from scipy.stats import pearsonr

def get_data(sta, cha, starttime, endtime, dt, fmin, fmax):
    mainpath = '/bay_event/BAM_BLEIB_2021/mseed/'
    yyyy = str(starttime.year)
    #if sta in ['CBAM2', 'CBAM3', 'CBAM4']:
    #    DDD = '286'
    #else:
    DDD = str(starttime.julday).zfill(3)
    #print(DDD)
    fname = 'XX.'+sta+'..'+cha+'.D.'+yyyy+'.'+DDD
    if sta == 'CBAM3':
        _starttime = starttime - 407
        _endtime = endtime - 407
        st = read(mainpath+fname, starttime=_starttime-2.5, endtime=_endtime+2.5)
        print(st)
    else:
        st = read(mainpath+fname, starttime=starttime-2.5, endtime=endtime+2.5)
    
    inv = read_inventory('../stationXML/stationXML/XX.'+sta+'.BLEIB.xml')
    st.attach_response(inv)
    #print(dt)    
    for tr in st:
    #    if DDD == '280':
        tr.stats.starttime = tr.stats.starttime + dt
    #    if DDD == '281':
    #        tr.stats.starttime = tr.stats.starttime + dt[1]
    st.trim(starttime, endtime)

    st.detrend('demean')
    st.taper(0.1)
    
    if sta.startswith('BB'):
        st.remove_response(water_level=60, output='ACC')
    else:
        st.remove_sensitivity()
    #st.filter('bandpass', freqmin=1.5, freqmax=5.0, zerophase=True, corners=4)
    st.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True, corners=4)
    #st.filter('bandpass', freqmin=7.5, freqmax=22.5, zerophase=True, corners=4)
    if not sta in ['CBAM2', 'CBAM3', 'CBAM1']:
        st.resample(100)
    print(st)
    if cha[1] == 'J':
        st[0].data = np.imag(sp.signal.hilbert(st[0].data))
    return st

def read_events(path_to_file):
    evt_file = open(path_to_file, 'r')
    evt_start = []
    evt_end = []
    ps = []
    load = []
    source = []
    dt_rr = []
    toUTC = 0
    #toUTC = -7200
    while True:
        evt = evt_file.readline()
        if not evt:
            break
        for i in range(4):
            evt = evt_file.readline().split(' ')
            evt_start.append(UTCDateTime(evt[0])+toUTC)
            evt_end.append(UTCDateTime(evt[1])+toUTC)
            ps.append(float(evt[2]))
            load.append(float(evt[3]))
            source.append(evt[4])
            dt_rr.append(float(evt[5]))

    return evt_start, evt_end, ps, load, source, dt_rr



def nearestPow2(x):
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b

def linear_func(p, x):
   m, c = p
   return m*x + c

# This function calculates the ratio between rotation rate and acceleration as a linear least squares regression.
def vel_from_amps(rr, acc, event_length, thresh, srate, offset, doplot, model):
    win_len = 1.0 # seconds
    win_nsamp = int(win_len * srate)
    cc_thresh = 0.85
    npts = len(rr)
    first = 0
    last = win_nsamp
    start = int(offset)
    stop = int(offset+event_length)
    nwins = int(npts / win_nsamp)
# initialize lists:
    VEL, _VEL, ccorr = np.zeros(nwins), np.zeros(nwins), np.zeros(nwins)
    #VEL, _VEL, ccorr = [], [], []
# we loop over the windows until we reach the end of the trace
    j = 0
    while last < npts:
        r = rr[first:last]
        a = acc[first:last]
        c, p = sp.stats.pearsonr(r, a)
        ccorr[j] = c
        _vel = []
        A = []
        R = []
# we check if the cross-correlation coefficient for this window is larger than 0.9
        if np.abs(c) >= cc_thresh:
            for i in range(len(r)):
# we take only rotation rate values that are above the self-noise
# assuming that acceleration is above self-noise if rotation rate is above self-noise
                if np.abs(r[i]) >= thresh:
                    R.append(r[i])
                    A.append(a[i])
            R = np.asarray(R)
            A = np.asarray(A)
# ODR
            data = RealData(R, A)
            odr = ODR(data, model, beta0=[0., 1.])
            out = odr.run()
            #out.pprint()
            #print(out.beta)
            VEL[j] = np.abs(out.beta[0])
# move to the next window
        first = int(first+win_nsamp)
        last = int(first+win_nsamp)
        j += 1
        
    ccorr = np.asarray(ccorr)
# now we cut out the interesting part of the event
    VEL = np.asarray(VEL)
    _VEL = []
    for val in VEL:
        if val != 0:
            _VEL.append(val)
    _VEL = np.asarray(_VEL)
    vel = np.nanmedian(_VEL)

# We implement the option to plot the analysis step from each single event
    if doplot == True:
        rgba1 = (216/255,27/255,96/255,1)
        rgba3 = (255/255,183/255,7/255,1)

        fig, ax0 = plt.subplots(1, 1, figsize=(6, 5))
        ax1 = ax0.twinx()
        ax0.plot(np.arange(0,len(rr),1)*(1/srate), rr/max(rr), color='r', linewidth=1.5)
        ax0.plot(np.arange(0,len(acc),1)*(1/srate), acc/max(acc), color='k', linewidth=1)
        t_vel = (np.arange(0, len(VEL), 1)) * win_len
        t_corr = np.arange(0, len(ccorr), 1) * win_len
        ax1.plot(t_vel, np.abs(VEL), color=rgba1)
        ax0.plot(t_corr, ccorr, color=rgba3)
        ax0.set_xlim(0,len(rr)*(1/srate))
        ax0.set_ylim(-1.1,1.1)
        #ax1.set_ylim(0, 120)
        ax0.set_xlabel('time [s]')
        ax0.set_ylabel('norm. amp / ccorr')
        ax1.set_ylabel(r'acc/rr [m/s]', color=rgba1)
        ax0.tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
        ax0.tick_params(axis='y', which='both', direction='in', left=True, labelleft=True)
        ax1.tick_params(axis='y', which='both', direction='in', right=True, labelright=True, left=False, labelleft=False, color=rgba1, labelcolor=rgba1)

        plt.show()
    
    print(vel) 
    return vel



def main():
    sta_rr = 'RBAM1'
    evt_file = 'events_manual_'+sta_rr+'.txt'
    #evt_file = 'events_manual_RBAM4.txt'
    cha_rr = 'HJX'
    sta_acc = 'BBAM1'
    cha_acc = 'HHZ'
    dt_acc = np.zeros(150)
    sourceType = 'HH'
    win_len = 1  # seconds
    win_nsamp = win_len * 100
    evt_len = 5
    overlap = 0.0
    thresh = 1e-6
    offset = 7 # seconds
    fmin = 16.0
    fmax = 20.0
    start_times, end_times, pre_stress, load, source, dt_rr = read_events(evt_file)
    vels = []
    stds = []
    VELS = []
    mins = []
    maxs = []
    times = []
    pre_stresses = []
    loads = []
    Sstart_times = []
    load_color = []
    _v0 = []
    rgba0 = (255/255,255/255,255/255,1)
    rgba1 = (216/255,27/255,96/255,1)
    rgba2 = (30/255,136/255,229/255,1)
    rgba3 = (255/255,183/255,7/255,1)
    rgba4 = (0/255,77/255,64/255,1)
    rgba5 = (0/255,0/255,0/255,1)
    #doplot = True
    doplot = False
    k = 0
    for i in range(len(start_times)):
        if source[i] == sourceType:
            k+=1
            print('#####################')
            print('number of event: ',k)
            pre_stresses.append(pre_stress[i])
            print('pre-stress: ', pre_stress[i])
            if float(load[i]) == 900.0:
                load_color.append(rgba4)
                print(load[i])
            if float(load[i]) == 600.0:
                load_color.append(rgba3)
                print(load[i])
            if float(load[i]) == 300.0:
                load_color.append(rgba2)
                print(load[i])
            if float(load[i]) == 0.0:
                load_color.append(rgba1)
                print(load[i])
            print('#####################')
            st_rr = get_data(sta_rr, cha_rr, start_times[i], end_times[i], dt_rr[i], fmin, fmax)
            #st_rr = get_data(sta_rr, cha_rr, start_times[i], end_times[i], -0.03)
            st_acc = get_data(sta_acc, cha_acc, start_times[i], end_times[i], dt_acc[i], fmin, fmax)
            #print(start_times[i])
            rr = st_rr[0].data
            acc = st_acc[0].data
            #if k > 42:
            #    doplot = True
            linear_model = Model(linear_func)
            vel = vel_from_amps(rr, acc, evt_len, thresh, st_rr[0].stats.sampling_rate, offset, doplot, linear_model)
            vels.append(vel)
            #stds.append(std)
            #mins.append(Min)
            #maxs.append(Max)
            #VELS.append(_vel)
            #time = np.arange(0, len(_vel), 1) * (win_len*(1-overlap)) + start_times[i].timestamp#+offset
            #times.append(time)
            loads.append(load[i])
            if load[i] == 0.0 and pre_stress[i] == 400:
                _v0.append(vel)
            Sstart_times.append(start_times[i].datetime)
    st_rr.plot()
    vels = np.asarray(vels)
    #Stds = np.asarray(stds)
    #Mins = np.asarray(mins)
    #Maxs = np.asarray(maxs)
    pre_stresses = np.asarray(pre_stresses)
    loads = np.asarray(loads)
    _v0 = np.asarray(_v0[:3])
    v0 = np.nanmean(_v0)
    dv_v = ((vels - v0) / v0) * 100
    #print(vels)
    #print(dv_v)
    #mins = ((Mins - v0) / v0) * 100
    #maxs = ((Maxs - v0) / v0) * 100
    #stds = ((Stds) / v0) * 100

    #print(load_color)    

    stress_changes0 = [
                        dt.datetime(2021, 10, 7, 9, 8, 0),   # 450 -> 400
                        dt.datetime(2021, 10, 7, 10, 25, 0), # 400 -> 350
                        dt.datetime(2021, 10, 7, 11, 33, 0), # 350 -> 300
                        dt.datetime(2021, 10, 7, 12, 38, 0), # 300 -> 250
                        dt.datetime(2021, 10, 7, 13, 49, 0), # 250 -> 200
                    ]
    stress_changes1 = [
                        dt.datetime(2021, 10, 8, 7, 15, 0), # 200 -> 250
                        dt.datetime(2021, 10, 8, 7, 35, 0), # 250 -> 300
                        dt.datetime(2021, 10, 8, 7, 57, 0), # 300 -> 350
                        dt.datetime(2021, 10, 8, 8, 20, 0), # 350 -> 400
                        dt.datetime(2021, 10, 8, 9, 31, 0), # 400 -> 450
                    ]
   
    stress_colors = [
                    (100/255,100/255,100/255,1),
                    (128/255,128/255,128/255,1),
                    (182/255,182/255,182/255,1),
                    (210/255,210/255,210/255,1),
                    (225/255,225/255,225/255,1),
                    (250/255,250/255,250/255,1),
                    ]

# Plots
    params = {'text.usetex': True, 
          'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
    plt.rcParams.update(params)
    sizeOfFont = 14
    fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}
    rc('font',**fontProperties)

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12, 7))

#    #norm = matplotlib.colors.Normalize(vmin=min(l_med), vmax=max(l_med), clip=True)
#    #mapper = cm.ScalarMappable(norm=norm, cmap='magma')
#    #load_color = np.array([(mapper.to_rgba(v)) for v in l_med])
#

    #print(len(vels))
    #print(len(start_times)) 
    #print(len(load_color))
    for i in range(len(Sstart_times)):
        #line1 = ax0.plot(times[i], VELS[i], color='k', alpha=1)
        line2 = ax0.plot(Sstart_times[i], dv_v[i], marker='o', color=load_color[i], markersize=4, alpha=1)
        #ax0.vlines(x=Sstart_times[i], ymin=mins[i], ymax=maxs[i], colors='k', lw=0.5)
        #ax0.vlines(x=Sstart_times[i], ymin=dv_v[i]-0.5*stds[i], ymax=dv_v[i]+0.5*stds[i], colors=load_color[i], lw=0.8)
        #line1 = ax1.plot(times[i], VELS[i], color='k', alpha=1)
        line2 = ax1.plot(Sstart_times[i], dv_v[i], marker='o', color=load_color[i], markersize=4, alpha=1)
        #ax1.vlines(x=Sstart_times[i], ymin=mins[i], ymax=maxs[i], colors='k', lw=0.5)
        #ax1.vlines(x=Sstart_times[i], ymin=dv_v[i]-0.5*stds[i], ymax=dv_v[i]+0.5*stds[i], colors=load_color[i], lw=0.8)

    ax0.axvspan(Sstart_times[0]-dt.timedelta(seconds=120), stress_changes0[0], color=stress_colors[0])
    ax0.axvspan(stress_changes0[0], stress_changes0[1], color=stress_colors[1])
    ax0.axvspan(stress_changes0[1], stress_changes0[2], color=stress_colors[2])
    ax0.axvspan(stress_changes0[2], stress_changes0[3], color=stress_colors[3])
    ax0.axvspan(stress_changes0[3], stress_changes0[4], color=stress_colors[4])
    ax0.axvspan(stress_changes0[4], Sstart_times[-25]+dt.timedelta(seconds=120), color=stress_colors[5])

    ax1.axvspan(Sstart_times[-24]-dt.timedelta(seconds=120), stress_changes1[0], color=stress_colors[5])
    ax1.axvspan(stress_changes1[0], stress_changes1[1], color=stress_colors[4])
    ax1.axvspan(stress_changes1[1], stress_changes1[2], color=stress_colors[3])
    ax1.axvspan(stress_changes1[2], stress_changes1[3], color=stress_colors[2])
    ax1.axvspan(stress_changes1[3], stress_changes1[4], color=stress_colors[1])
    ax1.axvspan(stress_changes1[4], Sstart_times[-1]+dt.timedelta(seconds=120), color=stress_colors[0])

    ax0.set_xlim(Sstart_times[4]-dt.timedelta(seconds=120), Sstart_times[-29]+dt.timedelta(seconds=120))    
    ax1.set_xlim(Sstart_times[-24]-dt.timedelta(seconds=120), Sstart_times[-5]+dt.timedelta(seconds=120))    
    
    #print(Sstart_times[4], Sstart_times[-29])

    ax0.set_ylim(-30, 10)    
    ax1.set_ylim(-30, 10)    


    ticks0 = [#dt.datetime(2021, 10, 7, 9, 0, 0), 
                dt.datetime(2021, 10, 7, 10, 0, 0), 
                dt.datetime(2021, 10, 7, 11, 0, 0),
                dt.datetime(2021, 10, 7, 12, 0, 0),
                dt.datetime(2021, 10, 7, 13, 0, 0)]
                #dt.datetime(2021, 10, 7, 14, 0, 0)]
    ticks1 = [dt.datetime(2021, 10, 8, 7, 30, 0),
                dt.datetime(2021, 10, 8, 8, 30, 0),
                dt.datetime(2021, 10, 8, 9, 30, 0)]
    ax0.set_xticks(ticks0)
    ax1.set_xticks(ticks1)

    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax0.set_xlabel('2021-10-07')
    ax1.set_xlabel('2021-10-08')

    ax0.tick_params(axis='x', which='both', direction='in')
    ax1.tick_params(axis='x', which='both', direction='in')
    ax0.tick_params(axis='y', which='both', direction='in', left=True, labelleft=True)
    ax1.tick_params(axis='y', which='both', direction='in', right=True, labelright=True, left=False, labelleft=False)
    ax1.yaxis.set_label_position("right")

    ax0.spines.right.set_visible(False)
    ax1.spines.left.set_visible(False)

    ax0.set_ylabel(r'dv/v [\%]')
    ax1.set_ylabel(r'dv/v [\%]')

    legend_elements = [
                    Line2D([0], [0], linestyle='none', label=r'\textbf{load:}'),
                    Line2D([0], [0], color=rgba1, marker='o', lw=4, linestyle='none', label=r'0\,kg'),
                    Line2D([0], [0], color=rgba2, marker='o', lw=4, linestyle='none', label=r'300\,kg'),
                    Line2D([0], [0], linestyle='none', label=''),
                    Line2D([0], [0], color=rgba3, marker='o', lw=4, linestyle='none', label=r'600\,kg'),
                    Line2D([0], [0], color=rgba4, marker='o', lw=4, linestyle='none', label=r'900\,kg'),
                    #Line2D([0], [0], linestyle='none', label=''),
                    #Line2D([0], [0], linestyle='none', label=''),
                    ]
    ax1.legend(handles=legend_elements, ncol=2, loc='lower left', bbox_to_anchor=(-0.2, 0.1), frameon=True)

    ax0.text(0.9, 1.06,  r'\textbf{pre-stress [kN]}', ha='right', va='bottom', transform=ax0.transAxes)
    #ax0.text(0.03, 1.01,  r'\textbf{450}', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.18, 1.01,  r'\textbf{400}', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.41, 1.01,  r'\textbf{350}', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.63, 1.01,  r'\textbf{300}', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.85, 1.01,  r'\textbf{250}', ha='right', va='bottom', transform=ax0.transAxes)
    #ax0.text(1.0, 1.01,  r'\textbf{200}', ha='right', va='bottom', transform=ax0.transAxes)
    ax1.text(0.05, 1.01,  r'\textbf{250}', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.21, 1.01,  r'\textbf{300}', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.38, 1.01,  r'\textbf{350}', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.71, 1.01,  r'\textbf{400}', ha='right', va='bottom', transform=ax1.transAxes)
    #ax1.text(0.99, 1.01,  r'\textbf{450}', ha='right', va='bottom', transform=ax1.transAxes)
 
    filt_str = str(int(fmin))+'_'+str(int(fmax))
    ax0.set_title(sta_rr+' '+filt_str.split('_')[0]+'Hz - '+filt_str.split('_')[1]+'Hz', loc='left', pad=30)
    
    plt.subplots_adjust(
    top=0.9,
    bottom=0.08,
    left=0.06,
    right=0.94,
    hspace=0.2,
    wspace=0.2
    )
    plt.savefig('plots/dv_v_fromamp_'+sta_rr+'_'+filt_str+'.png', dpi=300)    
    plt.show()
#
#


main()
