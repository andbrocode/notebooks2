def __compare_backazimuth_codes(rot, acc, cat_event, fmin, fmax, cc_thres=None, plot=False):

    import scipy.stats as sts
    import matplotlib.pyplot as plt

    from numpy import ones, linspace, histogram, concatenate, average, argmax, isnan, sqrt, cov, nan, array, arange
    from obspy import UTCDateTime
    from functions.compute_backazimuth import __compute_backazimuth
    from functions.compute_backazimuth_tangent import __compute_backazimuth_tangent
    
    
    rot.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax)
    acc.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax)

    config = {}

    config['tbeg'] = rot[0].stats.starttime
    config['tend'] = rot[0].stats.endtime

    ## Eventtime
    config['eventtime'] = UTCDateTime(cat_event.origins[0].time)

    ## specify coordinates of station
    config['station_longitude'] =  -116.455439
    config['station_latitude']  = 33.6106

    ## specify window length for baz estimation in seconds
    config['win_length_sec'] = 1/fmin

    ## define an overlap for the windows in percent (50 -> 50%)
    config['overlap'] = 50

    ## specify steps for degrees of baz
    config['step'] = 1


    out1 = __compute_backazimuth(
                                acc,
                                rot,
                                config,
                                wave_type='rayleigh',
                                event=cat_event,
                                plot=False,
                                flim=(fmin, fmax),
                                show_details=False,
    )

    out2 = __compute_backazimuth(
                                acc,
                                rot,
                                config,
                                wave_type='love',
                                event=cat_event,
                                plot=False,
                                flim=(fmin, fmax),
                                show_details=False,
    )

    out3 = __compute_backazimuth_tangent(
                                        rot,
                                        acc,
                                        win_time_s= config['win_length_sec'],
                                        overlap=config['overlap']/100,
                                        baz_theo=out2['baz_theo'],
                                        cc_thres=0,
                                        plot=False,
    )

    ## filter according to cc-threshold
    if cc_thres:
        for ii, _cc in enumerate(out1['cc_max']):
            if _cc <= cc_thres:
                out1['cc_max'][ii], out1['cc_max_y'][ii] = nan, nan
        for ii, _cc in enumerate(out2['cc_max']):
            if _cc <= cc_thres:
                out2['cc_max'][ii], out2['cc_max_y'][ii] = nan, nan
        for ii, _cc in enumerate(out3['ccoef']):
            if _cc <= cc_thres:
                out3['ccoef'][ii], out3['baz_est'][ii] = nan, nan



    if plot:

        NN = 6
        rot_scaling, rot_unit = 1e6, r"$\mu$rad/s"
        trans_scaling, trans_unit = 1e3, r"mm/s$^2$"

        font = 12

        fig1, ax = plt.subplots(NN, 1, figsize=(15, 10), sharex=True)

        plt.subplots_adjust(hspace=0.2)

        hz = acc.select(channel="*HZ")[0]
        hn = acc.select(channel="*HN")[0]
        he = acc.select(channel="*HE")[0]

        jz = rot.select(channel="*JZ")[0]
        jn = rot.select(channel="*JN")[0]
        je = rot.select(channel="*JE")[0]

        t1, t2 = hz.times().min(), hz.times().max()

        ax[0].plot(hz.times(), hz.data*trans_scaling, 'black', label=f"XPFO.Z")
        ax[1].plot(hn.times(), hn.data*trans_scaling, 'black', label=f"XPFO.N")
        ax[2].plot(he.times(), he.data*trans_scaling, 'black', label=f"XPFO.E")

        ax[0].set_ylim(-max(abs(hz.data*trans_scaling)), max(abs(hz.data*trans_scaling)))
        ax[1].set_ylim(-max(abs(hn.data*trans_scaling)), max(abs(hn.data*trans_scaling)))
        ax[2].set_ylim(-max(abs(he.data*trans_scaling)), max(abs(he.data*trans_scaling)))

        ax00 = ax[0].twinx()
        ax00.plot(jz.times(), jz.data*rot_scaling, 'darkred', label=r"BSPF.Z")

        ax11 = ax[1].twinx()
        ax11.plot(jn.times(), jn.data*rot_scaling, 'darkred', label=r"BSPF.N")

        ax22 = ax[2].twinx()
        ax22.plot(je.times(), je.data*rot_scaling, 'darkred', label=r"BSPF.E")

        ax00.set_ylim(-max(abs(jz.data*rot_scaling)), max(abs(jz.data*rot_scaling)))
        ax11.set_ylim(-max(abs(jn.data*rot_scaling)), max(abs(jn.data*rot_scaling)))
        ax22.set_ylim(-max(abs(je.data*rot_scaling)), max(abs(je.data*rot_scaling)))

        cmap = plt.get_cmap("viridis", 10)

        ca3 = ax[3].scatter(out1['cc_max_t'], out1['cc_max_y'], c=out1['cc_max'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        ax[3].plot([t1, t2], ones(2)*out1['baz_theo'], lw=1.5, alpha=0.7, color="k", ls="--", zorder=1)

        ca4 = ax[4].scatter(out2['cc_max_t'], out2['cc_max_y'], c=out2['cc_max'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        ax[4].plot([t1, t2], ones(2)*out2['baz_theo'], lw=1.5, alpha=0.7, color="k", ls="--", zorder=1)

        ca5 = ax[5].scatter(out3['t_win_center'], out3['baz_est'], c=out3['ccoef'], s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        ax[5].plot([t1, t2], ones(2)*out3['baz_theo'], lw=1.5, alpha=0.7, color="k", ls="--", zorder=1)

        cax3 = ax[3].inset_axes([1.01, 0., 0.02, 1])
        cb3 = plt.colorbar(ca3, ax=ax[3], cax=cax3)
        cb3.set_label("CC-Coeff.", fontsize=font)

        cax4 = ax[4].inset_axes([1.01, 0., 0.02, 1])
        cb4 = plt.colorbar(ca4, ax=ax[4], cax=cax4)
        cb4.set_label("CC-Coeff.", fontsize=font)

        cax5 = ax[5].inset_axes([1.01, 0., 0.02, 1])
        cb5 = plt.colorbar(ca5, ax=ax[5], cax=cax5)
        cb5.set_label("CC-Coeff.", fontsize=font)

        ax[3].set_ylabel(f"Rayleigh Baz.(°)")
        ax[4].set_ylabel(f"Love Baz.(°)")
        ax[5].set_ylabel(f"CoVar. Baz.(°)")


        ax[0].set_yticks(linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-1], len(ax[0].get_yticks())))
        ax00.set_yticks(linspace(ax00.get_yticks()[0], ax00.get_yticks()[-1], len(ax[0].get_yticks())))

        ax[1].set_yticks(linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-1], len(ax[1].get_yticks())))
        ax11.set_yticks(linspace(ax11.get_yticks()[0], ax11.get_yticks()[-1], len(ax[1].get_yticks())))

        ax[2].set_yticks(linspace(ax[2].get_yticks()[0], ax[2].get_yticks()[-1], len(ax[2].get_yticks())))
        ax22.set_yticks(linspace(ax22.get_yticks()[0], ax22.get_yticks()[-1], len(ax[2].get_yticks())))

        for i in [0,1,2]:
            ax[i].grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
            ax[i].legend(loc=1)
            ax[i].set_ylabel(f"a ({trans_unit})")

        for i in [3,4,5]:
            ax[i].set_ylim(-5, 365)
            ax[i].set_yticks(range(0,360+60,60))
            ax[i].grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
            # ax[i].set_ylabel(f"Baz.(°)")

        for aaxx in [ax00, ax11, ax22]:
            aaxx.tick_params(axis='y', colors="darkred")
            aaxx.set_ylabel(f"$\omega$ ({rot_unit})", color="darkred")
            aaxx.legend(loc=4)

        ax[0].set_title(f" {config['tbeg'].date}  {str(config['tbeg'].time).split('.')[0]}-{str(config['tend'].time).split('.')[0]} UTC | f = {fmin}-{fmax} Hz | T = {config['win_length_sec']} s | {config['overlap']} % ")

        plt.show();



    ## compute statistics
    deltaa = 10
    angles = arange(0, 365, deltaa)

    ## ______________________________________
    ## Rayleigh
    baz_rayleigh_no_nan = out1['cc_max_y'][~isnan(out1['cc_max_y'])]
    cc_rayleigh_no_nan = out1['cc_max'][~isnan(out1['cc_max'])]

    hist = histogram(out1['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out1['cc_max'], density=True)

    baz_rayleigh_mean = round(average(baz_rayleigh_no_nan, weights=cc_rayleigh_no_nan), 0)
    baz_rayleigh_std = sqrt(cov(baz_rayleigh_no_nan, aweights=cc_rayleigh_no_nan))

    # baz_rayleigh_max = angles[argmax(hist[0])]+deltaa  ## add half of deltaa to be in the bin center
    kde1 = sts.gaussian_kde(baz_rayleigh_no_nan, weights=baz_rayleigh_no_nan)
    baz_rayleigh_max = angles[argmax(kde1.pdf(angles))] + deltaa/2

    ## ______________________________________
    ## Love
    baz_love_no_nan = out2['cc_max_y'][~isnan(out2['cc_max_y'])]
    cc_love_no_nan = out2['cc_max'][~isnan(out2['cc_max'])]

    hist = histogram(out2['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out2['cc_max'], density=True)

    baz_love_mean = round(average(baz_love_no_nan, weights=cc_love_no_nan), 0)
    baz_love_std = sqrt(cov(baz_love_no_nan, aweights=cc_love_no_nan))

    # baz_love_max = angles[argmax(hist[0])]+deltaa  ## add half of deltaa to be in the bin center
    kde2 = sts.gaussian_kde(baz_love_no_nan, weights=cc_love_no_nan)
    baz_love_max = angles[argmax(kde2.pdf(angles))] + deltaa/2

    ## ______________________________________
    ## Tangent
    baz_tangent_no_nan = out3['baz_est'][~isnan(out3['ccoef'])]
    cc_tangent_no_nan = out3['baz_est'][~isnan(out3['ccoef'])]

    hist = histogram(out3['baz_est'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out3['ccoef'], density=True)

    baz_tangent_mean = round(average(baz_tangent_no_nan, weights=cc_tangent_no_nan), 0)
    baz_tangent_std = sqrt(cov(baz_tangent_no_nan, aweights=cc_tangent_no_nan))

    # baz_tangent_max = angles[argmax(hist[0])]+deltaa  ## add half of deltaa to be in the bin center
    kde3 = sts.gaussian_kde(baz_tangent_no_nan, weights=cc_tangent_no_nan)
    baz_tangent_max = angles[argmax(kde3.pdf(angles))] + deltaa/2

    if plot:

        fig2, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].hist(out1['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out1['cc_max'], density=True)
        ax[0].plot(angles, kde1.pdf(angles), c='C1', lw=2, label='KDE')
        ax[0].axvline(baz_rayleigh_max, color="r")
        ax[0].axvline(baz_rayleigh_mean, color="g")
        ax[0].set_title("Rayleigh")
        ax[0].set_xlabel("Backazimuth")
        ax[0].set_ylabel("Density")

        ax[1].hist(out2['cc_max_y'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out2['cc_max'], density=True)
        ax[1].plot(angles, kde2.pdf(angles), c='C1', lw=2, label='KDE')
        ax[1].axvline(baz_love_max, color="r")
        ax[1].axvline(baz_love_mean, color="g")
        ax[1].set_title("Love")
        ax[1].set_xlabel("Backazimuth")

        ax[2].hist(out3['baz_est'], bins=len(angles)-1, range=[min(angles), max(angles)], weights=out3['ccoef'], density=True)
        ax[2].plot(angles, kde3.pdf(angles), c='C1', lw=2, label='KDE')
        ax[2].axvline(baz_tangent_max, color="r")
        ax[2].axvline(baz_tangent_mean, color="g")
        ax[2].set_title("Co.Var.")
        ax[2].set_xlabel("Backazimuth")
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


    if plot:
        out['fig1'] = fig1
        out['fig2'] = fig2

    return out