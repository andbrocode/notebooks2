def __compute_backazimuth_and_velocity_noise(conf, rot0, acc0, fmin, fmax, plot=False, save=False):

    import scipy.stats as sts
    import matplotlib.pyplot as plt

    from numpy import ones, linspace, histogram, concatenate, average, argmax, isnan, sqrt, cov, nan, array, arange, zeros
    from obspy import UTCDateTime
    from obspy.signal.rotate import rotate_ne_rt
    from functions.compute_backazimuth import __compute_backazimuth
    from functions.compute_backazimuth_tangent import __compute_backazimuth_tangent

    ## minimum number of data points for histogram
    min_num_of_datapoints = 10

    ## rotation and acceleration data
    rot = rot0.copy()
    acc = acc0.copy()

    ## remove mean and filter
    rot.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax)
    acc.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax)

    config = conf

#     config['tbeg'] = rot[0].stats.starttime
#     config['tend'] = rot[0].stats.endtime

#     ## Eventtime
#     config['eventtime'] = UTCDateTime(rot[0].stats.starttime)


#     ## specify coordinates of station
#     config['station_longitude'] =  -116.455439
#     config['station_latitude']  = 33.6106

#     ## specify window length for baz estimation in seconds
#     config['win_length_sec'] = 2/fmin

#     ## define an overlap for the windows in percent (50 -> 50%)
#     config['overlap'] = 75

#     ## specify steps for degrees of baz
#     config['step'] = 1


    out1 = __compute_backazimuth(
                                acc,
                                rot,
                                config,
                                wave_type='rayleigh',
                                event=None,
                                plot=False,
                                flim=(fmin, fmax),
                                show_details=False,
    )

    out2 = __compute_backazimuth(
                                acc,
                                rot,
                                config,
                                wave_type='love',
                                event=None,
                                plot=False,
                                flim=(fmin, fmax),
                                show_details=False,
    )

    ## set random theo. baz.
    out2['baz_theo'] = 0

    out3 = __compute_backazimuth_tangent(
                                        rot,
                                        acc,
                                        win_time_s=config['win_length_sec'],
                                        overlap=config['overlap']/100,
                                        baz_theo=out2['baz_theo'],
                                        cc_thres=config['cc_thres'],
                                        plot=False,
    )

    ## filter according to cc-threshold
    if config['cc_thres']:
        for ii, _cc in enumerate(out1['cc_max']):
            if abs(_cc) <= config['cc_thres']:
                out1['cc_max'][ii], out1['cc_max_y'][ii] = nan, nan
        for ii, _cc in enumerate(out2['cc_max']):
            if abs(_cc) <= config['cc_thres']:
                out2['cc_max'][ii], out2['cc_max_y'][ii] = nan, nan
        for ii, _cc in enumerate(out3['ccoef']):
            if abs(_cc) <= config['cc_thres']:
                out3['ccoef'][ii], out3['baz_est'][ii] = nan, nan



    ## statistics of backazimuth
    deltaa = 10
    angles = arange(0, 365, deltaa)
    angles2 = arange(0, 365, 1)

    # ______________________________________
    # Rayleigh
    try:
        baz_rayleigh_no_nan = out1['cc_max_y'][~isnan(out1['cc_max_y'])]
        cc_rayleigh_no_nan = out1['cc_max'][~isnan(out1['cc_max'])]

        # compute histogram
        hist = histogram(out1['cc_max_y'], bins=len(angles)-1,
                         range=[min(angles), max(angles)],
                         weights=out1['cc_max'], density=True
                        )

        baz_rayleigh_mean = round(average(baz_rayleigh_no_nan, weights=cc_rayleigh_no_nan), 0)
        baz_rayleigh_std = sqrt(cov(baz_rayleigh_no_nan, aweights=cc_rayleigh_no_nan))

        try:
            kde1 = sts.gaussian_kde(baz_rayleigh_no_nan, weights=cc_rayleigh_no_nan)
            kde1_success = True
        except:
            kde1_success = False

        # find maximum of KDE if computation successful and more than X values are used
        if len(baz_rayleigh_no_nan) > min_num_of_datapoints and kde1_success:
            baz_rayleigh_max = angles2[argmax(kde1.pdf(angles2))]
        else:
            baz_rayleigh_max = nan

        # ______________________________________
        # Love
        baz_love_no_nan = out2['cc_max_y'][~isnan(out2['cc_max_y'])]
        cc_love_no_nan = out2['cc_max'][~isnan(out2['cc_max'])]

        # compute histogram
        hist = histogram(out2['cc_max_y'], bins=len(angles)-1,
                         range=[min(angles), max(angles)],
                         weights=out2['cc_max'], density=True
                        )

        baz_love_mean = round(average(baz_love_no_nan, weights=cc_love_no_nan), 0)
        baz_love_std = sqrt(cov(baz_love_no_nan, aweights=cc_love_no_nan))

        # baz_love_max = angles[argmax(hist[0])]+deltaa  ## add half of deltaa to be in the bin center
        try:
            kde2 = sts.gaussian_kde(baz_love_no_nan, weights=cc_love_no_nan)
            kde2_success = True
        except:
            kde2_success = False

        # find maximum of KDE if computation successful and more than X values are used
        if len(baz_love_no_nan) > min_num_of_datapoints and kde2_success:
            baz_love_max = angles2[argmax(kde2.pdf(angles2))]
        else:
            baz_love_max = nan

        # ______________________________________
        # tangent
        baz_tangent_no_nan = out3['baz_est'][~isnan(out3['ccoef'])]
        cc_tangent_no_nan = out3['baz_est'][~isnan(out3['ccoef'])]

        # compute histogram
        hist = histogram(out3['baz_est'], bins=len(angles)-1,
                         range=[min(angles), max(angles)],
                         weights=out3['ccoef'], density=True
                        )

        baz_tangent_mean = round(average(baz_tangent_no_nan, weights=cc_tangent_no_nan), 0)
        baz_tangent_std = sqrt(cov(baz_tangent_no_nan, aweights=cc_tangent_no_nan))

        try:
            kde3 = sts.gaussian_kde(baz_tangent_no_nan, weights=cc_tangent_no_nan)
            kde3_success = True
        except:
            kde3_success = False

        # find maximum of KDE if computation successful and more than X values are used
        if len(baz_tangent_no_nan) > min_num_of_datapoints and kde3_success:
            baz_tangent_max = angles2[argmax(kde3.pdf(angles2))]
        else:
            baz_tangent_max = nan

    except Exception as e:
        baz_tangent_max = nan
        print(e)
        pass

    # statistics of velocities
    delta_vel = 200
    velocities = arange(0, 4000, delta_vel)
    velocities_fine = arange(0, 400, 50)


    # ______________________________________
    # Rayleigh velocity
    try:
        vel_rayleigh_no_nan, cc_vel_rayleigh_no_nan = zeros(len(out1['vel'])), zeros(len(out1['cc_max']))
        for _k, (_v, _c) in enumerate(zip(out1['vel'], out1['cc_max'])):
            if not isnan(_v) and not isnan(_c):
                vel_rayleigh_no_nan[_k] = _v
                cc_vel_rayleigh_no_nan[_k] = _c

        # compute histogram
        hist = histogram(out1['vel'], bins=len(velocities)-1,
                         range=[min(velocities), max(velocities)],
                         weights=out1['cc_max'], density=True
                        )

        vel_rayleigh_mean = round(average(vel_rayleigh_no_nan, weights=cc_vel_rayleigh_no_nan), 0)
        vel_rayleigh_std = sqrt(cov(vel_rayleigh_no_nan, aweights=cc_vel_rayleigh_no_nan))

        try:
            vel_kde1 = sts.gaussian_kde(vel_rayleigh_no_nan, weights=cc_vel_rayleigh_no_nan)
            vel_kde1_success = True
        except:
            vel_kde1_success = False

        # find maximum of KDE if computation successful and more than X values are used
        if len(vel_rayleigh_no_nan) > min_num_of_datapoints and vel_kde1_success:
            vel_rayleigh_max = velocities_fine[argmax(vel_kde1.pdf(velocities_fine))]
        else:
            vel_rayleigh_max = nan

    except Exception as e:
        print(" -> no Rayleigh velocities")
        print(e)
        pass


    # ______________________________________
    # Love velocity
    try:

        vel_love_no_nan, cc_vel_love_no_nan = zeros(len(out2['vel'])), zeros(len(out2['cc_max']))
        for _k, (_v, _c) in enumerate(zip(out2['vel'], out2['cc_max'])):
            if not isnan(_v) and not isnan(_c):
                vel_love_no_nan[_k] = _v
                cc_vel_love_no_nan[_k] = _c

        # compute histogram
        hist = histogram(out2['vel'], bins=len(velocities)-1,
                         range=[min(velocities), max(velocities)],
                         weights=out2['cc_max'], density=True
                        )

        vel_love_mean = round(average(vel_love_no_nan, weights=cc_vel_love_no_nan), 0)
        vel_love_std = sqrt(cov(vel_love_no_nan, aweights=cc_vel_love_no_nan))
        try:
            vel_kde2 = sts.gaussian_kde(vel_love_no_nan, weights=cc_vel_love_no_nan)
            vel_kde2_success = True
        except:
            vel_kde2_success = False

        # find maximum of KDE if computation successful and more than X values are used
        if len(vel_love_no_nan) > min_num_of_datapoints and vel_kde2_success:
            vel_love_max = velocities_fine[argmax(vel_kde2.pdf(velocities_fine))]
        else:
            vel_love_max = nan

    except Exception as e:
        print(" -> no Love velocities")
        print(e)
        pass

    ## ______________________________________
    ## Plotting

    if plot or save:

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        Ncol, Nrow = 8, 7

        fig3 = plt.figure(figsize=(15, 12))

        gs = GridSpec(Nrow, Ncol, figure=fig3, hspace=0.15)

        ax0 = fig3.add_subplot(gs[0, :])
        ax1 = fig3.add_subplot(gs[1, :])
        ax2 = fig3.add_subplot(gs[2, :])

        ax3 = fig3.add_subplot(gs[3, :])
        ax4 = fig3.add_subplot(gs[4, :])
        ax5 = fig3.add_subplot(gs[5, :])

        ax6 = fig3.add_subplot(gs[3, 7:])
        ax7 = fig3.add_subplot(gs[4, 7:])
        ax8 = fig3.add_subplot(gs[5, 7:])

        ax9 = fig3.add_subplot(gs[6, :])

        ax6.set_axis_off()
        ax7.set_axis_off()
        ax8.set_axis_off()

        for _ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
            _ax.set_xticklabels([])

        rot_scaling, rot_unit = 1e9, r"nrad/s"
        trans_scaling, trans_unit = 1e6, r"$\mu$m/s$^2$"

        font = 12

        hz = acc.select(channel="*HZ")[0]
        hn = acc.select(channel="*HN")[0]
        he = acc.select(channel="*HE")[0]

        jz = rot.select(channel="*JZ")[0]
        jn = rot.select(channel="*JN")[0]
        je = rot.select(channel="*JE")[0]

        hr, ht = rotate_ne_rt(hn.data, he.data, baz_tangent_max)
        jr, jt = rotate_ne_rt(jn.data, je.data, baz_tangent_max)

        ## reverse polarity of transverse rotation!!
        jt *= -1

        ax0.plot(hz.times(), ht*trans_scaling, 'black', label=f"FUR.BHT")
        ax1.plot(hz.times(), hr*trans_scaling, 'black', label=f"FUR.BHR")
        ax2.plot(hz.times(), hz.data*trans_scaling, 'black', label=f"FUR.BHZ")

        try:
            ax0.set_ylim(-max(abs(ht*trans_scaling)), max(abs(ht*trans_scaling)))
            ax1.set_ylim(-max(abs(hr*trans_scaling)), max(abs(hr*trans_scaling)))
            ax2.set_ylim(-max(abs(hz.data*trans_scaling)), max(abs(hz.data*trans_scaling)))
        except:
            print(f" -> y axes limits failed!")

        ax00 = ax0.twinx()
        ax00.plot(jz.times(), jz.data*rot_scaling, 'darkred', label=r"ROMY.BJZ")

        ax11 = ax1.twinx()
        ax11.plot(jz.times(), jt*rot_scaling, 'darkred', label=r"-1x ROMY.BJT")

        ax22 = ax2.twinx()
        ax22.plot(jz.times(), jt*rot_scaling, 'darkred', label=r"-1x ROMY.BJT")

        try:
            ax00.set_ylim(-max(abs(jz.data*rot_scaling)), max(abs(jz.data*rot_scaling)))
            ax11.set_ylim(-max(abs(jt*rot_scaling)), max(abs(jt*rot_scaling)))
            ax22.set_ylim(-max(abs(jt*rot_scaling)), max(abs(jt*rot_scaling)))
        except:
            print(f" -> y axes limits failed!")

        cmap = plt.get_cmap("viridis", 10)

        ca3 = ax3.scatter(out1['cc_max_t'], out1['cc_max_y'], c=out1['cc_max'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)

        ca4 = ax4.scatter(out2['cc_max_t'], out2['cc_max_y'], c=out2['cc_max'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)

        ca5 = ax5.scatter(out3['t_win_center'], out3['baz_est'], c=out3['ccoef'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)

        ca9 = ax9.scatter(out1['cc_max_t'], out1['vel'], c=out1['cc_max'], marker='s', label="Rayleigh",
                          s=30, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        ca9 = ax9.scatter(out2['cc_max_t'], out2['vel'], c=out2['cc_max'], marker='^', label="Love",
                          s=30, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)

        cax3 = ax3.inset_axes([1.01, 0., 0.02, 1])
        cb3 = plt.colorbar(ca3, ax=ax3, cax=cax3)
        cb3.set_label("CC-Coeff.", fontsize=font)

        cax4 = ax4.inset_axes([1.01, 0., 0.02, 1])
        cb4 = plt.colorbar(ca4, ax=ax4, cax=cax4)
        cb4.set_label("CC-Coeff.", fontsize=font)

        cax5 = ax5.inset_axes([1.01, 0., 0.02, 1])
        cb5 = plt.colorbar(ca5, ax=ax5, cax=cax5)
        cb5.set_label("CC-Coeff.", fontsize=font)

        cax9 = ax9.inset_axes([1.01, 0., 0.02, 1])
        cb9 = plt.colorbar(ca9, ax=ax9, cax=cax9)
        cb9.set_label("CC-Coeff.", fontsize=font)

        ax3.set_ylabel(f"Rayleigh Baz. (°)")
        ax4.set_ylabel(f"Love Baz. (°)")
        ax5.set_ylabel(f"CoVar. Baz. (°)")
        ax9.set_ylabel(f"Phase Velocity (m/s)")

        ax66 = ax6.twinx()
        ax66.hist(out1['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)],
                  weights=out1['cc_max'], orientation="horizontal", density=True, color="grey")
        if kde1_success:
            ax66.plot(kde1.pdf(angles), angles, c="k", lw=2, label='KDE')
        ax66.axhline(baz_rayleigh_max, color="k", ls="--")
        ax66.set_axis_off()
        ax66.yaxis.tick_right()
        ax66.invert_xaxis()

        ax77 = ax7.twinx()
        ax77.hist(out2['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)],
                  weights=out2['cc_max'], orientation="horizontal", density=True, color="grey")
        if kde2_success:
            ax77.plot(kde2.pdf(angles), angles, c="k", lw=2, label='KDE')
        ax77.axhline(baz_love_max, color="k", ls="--")
        ax77.set_axis_off()
        ax77.yaxis.tick_right()
        ax77.invert_xaxis()

        ax88 = ax8.twinx()
        ax88.hist(out3['baz_est'], bins=len(angles)-1, range=[min(angles), max(angles)],
                  weights=out3['ccoef'], orientation="horizontal", density=True, color="grey")
        if kde3_success:
            ax88.plot(kde3.pdf(angles), angles, c="k", lw=2, label='KDE')
        ax88.axhline(baz_tangent_max, color="k", ls="--")
        ax88.set_axis_off()
        ax88.yaxis.tick_right()
        ax88.invert_xaxis()


        ax0.set_yticks(linspace(ax0.get_yticks()[0], ax0.get_yticks()[-1], len(ax0.get_yticks())))
        ax00.set_yticks(linspace(ax00.get_yticks()[0], ax00.get_yticks()[-1], len(ax0.get_yticks())))

        ax1.set_yticks(linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax1.get_yticks())))
        ax11.set_yticks(linspace(ax11.get_yticks()[0], ax11.get_yticks()[-1], len(ax1.get_yticks())))

        ax2.set_yticks(linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax2.get_yticks())))
        ax22.set_yticks(linspace(ax22.get_yticks()[0], ax22.get_yticks()[-1], len(ax2.get_yticks())))

        for _ax in [ax0, ax1, ax2]:
            _ax.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
            _ax.legend(loc=1)
            _ax.set_ylabel(f"a ({trans_unit})")
            _ax.set_xlim(0, (config['tend']-config['tbeg'])*1.15)

        for _ax in [ax3, ax4, ax5]:
            _ax.set_ylim(-5, 365)
            _ax.set_yticks(range(0, 360+60, 60))
            _ax.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
            _ax.set_xlim(0, (config['tend']-config['tbeg'])*1.15)

        for aaxx in [ax00, ax11, ax22]:
            aaxx.tick_params(axis='y', colors="darkred")
            aaxx.set_ylabel(f"$\omega$ ({rot_unit})", color="darkred")
            aaxx.legend(loc=4)

        ax0.set_title(f" {config['tbeg'].date}  {str(config['tbeg'].time).split('.')[0]}-{str(config['tend'].time).split('.')[0]} UTC | f = {round(fmin ,3)}-{round(fmax,3)} Hz | T = {config['win_length_sec']} s | CC > {config['cc_thres']} | {config['overlap']} % overlap")

        ax9.set_xlabel("Time (s)")
        ax9.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
        ax9.set_xlim(0, (config['tend']-config['tbeg'])*1.15)
        ax9.legend(loc=1, fontsize=font-1)

        if plot:
            plt.show();

    ## prepare output directory
    out = {}

    out['baz_theo'] = round(out2['baz_theo'], 0)
    out['baz_angles'] = angles
    out['baz_tangent_max'] = baz_tangent_max
    out['baz_tangent_mean'] = baz_tangent_mean
    out['baz_tangent_std'] = baz_tangent_std
    out['baz_rayleigh_max'] = baz_rayleigh_max
    out['baz_rayleigh_mean'] = baz_rayleigh_mean
    out['baz_rayleigh_std'] = baz_rayleigh_std
    out['baz_love_max'] = baz_love_max
    out['baz_love_mean'] = baz_love_mean
    out['baz_love_std'] = baz_love_std

    out['baz_rayleigh_all'] = out1['cc_max_y']
    out['baz_love_all'] = out2['cc_max_y']
    out['baz_tangent_all'] = out3['baz_est']

    out['cc_rayleigh_all'] = out1['cc_max']
    out['cc_love_all'] = out2['cc_max']
    out['cc_tangent_all'] = out3['ccoef']

    out['vel_love_max'] = vel_love_max
    out['vel_love_std'] = vel_love_std
    out['vel_rayleigh_max'] = vel_rayleigh_max
    out['vel_rayleigh_std'] = vel_rayleigh_std

    out['vel_rayleigh_all'] = out1['vel']
    out['vel_love_all'] = out2['vel']

    out['times_relative'] = out1['cc_max_t']

    if plot or save:
        out['fig3'] = fig3

    return out