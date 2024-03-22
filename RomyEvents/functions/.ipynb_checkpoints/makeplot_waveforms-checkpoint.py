def __makeplot_waveforms(acc, rot):

    import matplotlib.pyplot as plt

    from matplotlib.gridspec import GridSpec
    from numpy import arange, mean, nan, nanmax
    from pandas import DataFrame
    from functions.get_octave_bands import __get_octave_bands
    from scipy.signal import coherence

    rot_scale, rot_unit = 1e9, "nrad/s"
    acc_scale, acc_unit = 1e6, "$\mu$m/s$^2$"


    Nrow, Ncol = 3, 2
    font = 12
    lw = 1

    fig = plt.figure(figsize=(12, 5))

    gs = GridSpec(Nrow, Ncol, figure=fig)

    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax3 = fig.add_subplot(gs[2, :-1])

    ax4 = fig.add_subplot(gs[0, 1:])
    ax5 = fig.add_subplot(gs[1, 1:])
    ax6 = fig.add_subplot(gs[2, 1:])

    plt.subplots_adjust(hspace=0.1)

    cha='*Z'
    lbl = f"{rot.select(channel=cha)[0].stats.station}.{rot.select(channel=cha)[0].stats.channel}"
    ax1.plot(rot.select(channel=cha)[0].times(),
             rot.select(channel=cha)[0].data*rot_scale,
             color="k", label=lbl, lw=lw,
            )

    cha='*N'
    lbl = f"{rot.select(channel=cha)[0].stats.station}.{rot.select(channel=cha)[0].stats.channel}"
    ax2.plot(rot.select(channel=cha)[0].times(),
             rot.select(channel=cha)[0].data*rot_scale,
             color="k", label=lbl, lw=lw,
            )

    cha='*E'
    lbl = f"{rot.select(channel=cha)[0].stats.station}.{rot.select(channel=cha)[0].stats.channel}"
    ax3.plot(rot.select(channel=cha)[0].times(),
             rot.select(channel=cha)[0].data*rot_scale,
             color="k", label=lbl, lw=lw,
            )

    cha='*Z'
    lbl = f"{acc.select(channel=cha)[0].stats.station}.{acc.select(channel=cha)[0].stats.channel}"
    ax4.plot(acc.select(channel=cha)[0].times(),
             acc.select(channel=cha)[0].data*acc_scale,
             color="k", label=lbl, lw=lw,
            )

    cha='*N'
    lbl = f"{acc.select(channel=cha)[0].stats.station}.{acc.select(channel=cha)[0].stats.channel}"
    ax5.plot(acc.select(channel=cha)[0].times(),
             acc.select(channel=cha)[0].data*acc_scale,
             color="k", label=lbl, lw=lw,
            )

    cha='*E'
    lbl = f"{acc.select(channel=cha)[0].stats.station}.{acc.select(channel=cha)[0].stats.channel}"
    ax6.plot(acc.select(channel=cha)[0].times(),
             acc.select(channel=cha)[0].data*acc_scale,
             color="k", label=lbl, lw=lw,
            )


    for ax in [ax1, ax2, ax3, ax4 ,ax5, ax6]:
        ax.legend(loc=1, fontsize=font-3, bbox_to_anchor=(0.95, 1.10), ncol=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ax not in [ax3, ax6]:
            ax.set(xticklabels=[])
            ax.tick_params(bottom=False)
            ax.spines['bottom'].set_visible(False)
        if ax in [ax3, ax6]:
            ax.set_xlabel(f"Time (s) from {acc[0].stats.starttime.date} {str(acc[0].stats.starttime.time).split('.')[0]} UTC")

    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel(f"$\Omega$ ({rot_unit})", fontsize=font)

    for ax in [ax4, ax5, ax6]:
        ax.set_ylabel(f"a ({acc_unit})", fontsize=font)


    return fig