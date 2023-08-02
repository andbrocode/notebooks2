#!/bin/python3

def __compute_velocity_from_amplitude_ratio(rot0, acc0, baz=None, mode="love", win_time_s=2.0, cc_thres=0.8, overlap=0.5, plot=False):
    
    from scipy.stats import pearsonr
    from numpy import zeros, nan, ones, nanmean, array, nanmax, linspace, std
    from scipy import odr
    from functions.compute_linear_regression import __compute_linear_regression
    from obspy.signal.rotate import rotate_ne_rt
    
    npts = rot0[0].stats.npts
    
    df = rot0[0].stats.sampling_rate
    
    ## windows
    t_win = win_time_s
    n_win = int(win_time_s*df)
    nover = int(0.5*n_win)
        
    ## define windows
    n, windows = 0, []
    while n < npts:
        windows.append((n,n+n_win))        
        n+=n_win
        
    ## rotate channels
    if mode == "love":
        r_acc, t_acc = rotate_ne_rt(acc0.select(channel='*N')[0].data, 
                                    acc0.select(channel='*E')[0].data,
                                    baz
                                    )
        acc = t_acc
        rot = rot0.select(channel="*JZ")[0].data 
        
    elif mode == "rayleigh":
        r_rot, t_rot = rotate_ne_rt(rot0.select(channel='*N')[0].data, 
                                    rot0.select(channel='*E')[0].data,
                                    baz
                                    )
        rot = t_rot
        acc = acc0.select(channel="*HZ")[0].data     
    
    
    ## add overlap
    windows_overlap = []
    for i, w in enumerate(windows):
        if i == 0:
            windows_overlap.append((w[0],w[1]+nover))
        elif i >= (len(windows)-nover):
            windows_overlap.append((w[0]-nover, w[1]))
        else:
            windows_overlap.append((w[0]-nover, w[1]+nover))
    
    vel, ccor = ones(len(windows_overlap))*nan, zeros(len(windows_overlap))
    
    ## compute crosscorrelation for each window
    for j, (w1, w2) in enumerate(windows_overlap):
        
        ## trying to remove very small rotation values
#         rot_win = array([r if r>5e-8 else 0 for r in rot[w1:w2]])
#         acc_win = array([a if r>5e-8 else 0 for a, r in zip(acc[w1:w2], rot[w1:w2])])
        if mode == "love":
            rot_win, acc_win = rot[w1:w2], 0.5*acc[w1:w2]        
        elif mode == "rayleigh":
            rot_win, acc_win = rot[w1:w2], acc[w1:w2]

        if len(rot_win) < 10: 
            print(f" -> not enough samples in window (<10)")
            
#         ccor[j], p = pearsonr(rot_win/nanmax(abs(rot_win)), acc_win/nanmax(abs(acc_win)))
        ccor[j], p = pearsonr(rot_win, acc_win)
        
        ## if cc value is above threshold perform odr to get velocity
        if ccor[j] > cc_thres:
            data = odr.RealData(rot_win, acc_win)
            out = odr.ODR(data, model=odr.unilinear)
            output = out.run()
            slope, intercept = output.beta
            vel[j] = abs(slope)

            
    ## define time axis
    time = [((w2-w1)/2+w1)/df for (w1, w2) in windows_overlap]

#     print(len(time), len(ccor), len(vel))
    
    if plot:
        
        cmap = plt.get_cmap("viridis", 10)
        
        fig, ax = plt.subplots(1,1,figsize=(15,5))
        
        ax.plot(array(range(len(rot)))/df, rot/max(abs(rot)), alpha=1, color="grey", label="rotation rate (rad/s)")
        ax.plot(array(range(len(acc)))/df, acc/max(abs(acc)), alpha=0.5, color="tab:red", label=r"acceleration (m/s$^2$)")
        
        
        ax.set_ylim(-1,1)
        ax.set_xlim(0, len(rot)/df)
        ax.set_xlabel("Time (s)",fontsize=14)
        ax.set_ylabel("Norm. Amplitude",fontsize=14)
        ax.grid(zorder=0)
        ax.legend(loc=2, fontsize=13)
        
        ax2 = ax.twinx()
        cax = ax2.scatter(time, vel, c=ccor, s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, label="phase velocity estimate")
        ax2.set_ylabel(r"Phase Velocity (m/s)", fontsize=14)
        ax2.set_ylim(bottom=0)
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
        ax2.legend(loc=1, fontsize=13)
        
        cbar = plt.colorbar(cax, pad=0.08)
        cbar.set_label("Cross-Correlation Coefficient", fontsize=14)
        
        cax.set_clip_on(False)
    
        out = {"time":time, "velocity":vel, "ccoef":ccor, "fig":fig}    
    else:    
        out = {"time":time, "velocity":vel, "ccoef":ccor}
        
    return out

## End of File