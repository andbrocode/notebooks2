def __makeplot_density(out):

    import matplotlib.pyplot as plt
    import numpy as np


    def __get_median_psd(psds):

        from numpy import median, zeros, isnan

        med_psd = zeros(psds.shape[1])

        for f in range(psds.shape[1]):
            a = psds[:,f]
            med_psd[f] = median(a[~isnan(a)])

        return med_psd

    psd_median = __get_median_psd(dat)

    font = 14

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ## add Frequency Axis
    # g = lambda x: 1/x
    # ax2 = ax.secondary_xaxis("top", functions=(g, g))
    # ax2.set_xlabel("Frequency (Hz)", fontsize=font, labelpad=5)
    # ax2.set_xticklabels(ff, fontsize=11)


    out['dist'] = np.ma.masked_array(out['dist'], out['dist'] == 0)

    y_axis = 10**(out['bin_mids']/10)
    x_axis = out['frequencies']

    if x_axis[0] == 0:
        x_axis[0] == 1e-20

    ## plotting

    cmap = plt.colormaps.get_cmap('viridis')
    # cmap.set_under(color='white')

    _tmp = out['dist'].reshape(out['dist'].size)
    im = plt.pcolormesh(out['frequencies'], out['bin_mids'], out['dist'].T, cmap=cmap, shading="auto",
                        antialiased=True, vmin=min(_tmp[np.nonzero(_tmp)]), zorder=2)


    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(1e-3, 1e1)

    plt.ylim(1e-24, 1e-15)

    plt.tick_params(axis='both', labelsize=font-1)

    plt.grid(axis="both", which="both", ls="--", zorder=1)
    # plt.legend()

    plt.xlabel("Frequency (Hz)", fontsize=font)
    plt.ylabel(r"PSD ($rad^2 /s^2 /Hz$)", fontsize=font)

    ## add colorbar
    cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.77]) #[left, bottom, width, height]
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Propability Density", fontsize=font, labelpad=-50, color="white")

    plt.show();
    return fig