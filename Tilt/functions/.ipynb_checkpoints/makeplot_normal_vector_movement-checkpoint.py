def __makeplot_normal_vector_movement(set1, set2, set3):

    from numpy import deg2rad, arange
    import matplotlib as mpl

    vdirH, vnormH, vtimelineH = set1
    vdirV, vnormV, vtimelineV = set2
    vdirB, vnormB, vtimelineB = set3

    time_min = min([min(tromy_timeline), min(romyt_timeline), min(bromy_timeline)])
    time_max = max([max(tromy_timeline), max(romyt_timeline), max(bromy_timeline)])

    #-- Plot... ------------------------------------------------
    fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'), figsize=(15,5))

    font=12

    cmap = mpl.colormaps['viridis']
    # cmap = mpl.colormaps['viridis'].resampled(len(cs)+1)


    ## convert degrees to radians for plotting as polar!
    p1 = ax[0].scatter(deg2rad(vdirH), vnormH, c=vtimelineH, cmap=cmap, vmin=time_min, vmax=time_max, alpha=0.75, s=4, zorder=2)
    p2 = ax[1].scatter(deg2rad(vdirV), vnormV, c=vtimelineV, cmap=cmap, vmin=time_min, vmax=time_max, alpha=0.75, s=4, zorder=2)
    p3 = ax[2].scatter(deg2rad(vdirB), vnormB, c=vtimelineB, cmap=cmap, vmin=time_min, vmax=time_max, alpha=0.75, s=4, zorder=2)

    # cbar1 = plt.colorbar(p1, ax=ax[0], pad=0.1, orientation='horizontal')
    # cbar1.set_label('Time in days', rotation=0, fontsize=font, labelpad=10)

    # cbar2 = plt.colorbar(p2, ax=ax[1],  pad=0.1, orientation='horizontal')
    # cbar2.set_label('Time in days', rotation=0, fontsize=font, labelpad=10)

    # cbar3 = plt.colorbar(p3, ax=ax[2],  pad=0.1, orientation='horizontal')
    # cbar3.set_label('Time in days', rotation=0, fontsize=font, labelpad=10)

    cbar3 = plt.colorbar(p3, ax=ax,  pad=0.1, orientation='horizontal', fraction=0.1, shrink=5)
    # cbar3.set_label('Time from 2021-03-10 (days)', rotation=0, fontsize=font, labelpad=10)

    ## set new colorbar ticks
    ref_time = UTCDateTime("2021-03-10")
    nticks = [str((ref_time+time_min+t*86400).date) for t in cbar3.get_ticks()]
    cbar3.set_ticklabels(nticks)

# #     ax[0].set_ylim(min(vnormH)-0.1*min(vnormH), max(vnormH)+0.05*max(vnormH))
# #     ax[1].set_ylim(min(vnormV)-0.05*min(vnormV), max(vnormV)+0.01*max(vnormV))

    for i in range(3):
        ax[i].set_ylim(0, 0.1)
        ax[i].set_theta_zero_location('N')
        ax[i].set_theta_direction(-1)

    ax[0].text(deg2rad(25),0.12,r"(mrad)")
    ax[0].set_rgrids(arange(0.02, 0.12, 0.02), angle=25., zorder=0)

    ax[1].text(deg2rad(250),0.13,r"(mrad)")
    ax[1].set_rgrids(arange(0.02, 0.12, 0.02), angle=250., zorder=0)

    ax[2].text(deg2rad(205),0.12,r"(mrad)")
    ax[2].set_rgrids(arange(0.02, 0.12, 0.02), angle=205., zorder=0)

    ax[0].set_title("TROMY",fontsize=font)
    ax[1].set_title("ROMYT",fontsize=font)
    ax[2].set_title("BROMY",fontsize=font)

    plt.show();
    return fig