def get_phase_jumps_sta_lta(arr, times, LT, ST, amp_threshold, plot=True):

    import numpy as np
    from scipy.signal import hilbert
    from functions.smoothing import __smooth

    def sta_lta_detect(_data, threshold_upper, threshold_lower=None):

        N = len(_data)

        detection = np.ones(N)
        ratio = np.zeros(N)

        triggered = False
        for n in range(N):

            if n < LT:
                continue

            # compupte LTA, STA and ratio
            LTA = np.nanmean(abs(_data[n-int(LT):n]))
            STA = np.nanmean(abs(_data[n-int(ST):n]))
            ratio[n] = abs(STA-LTA)

            if triggered and ratio[n] > threshold_lower and threshold_lower is not None:
                detection[n] = 0
                continue
            else:
                triggered = False

            if ratio[n] > threshold_upper and threshold_lower is not None:
                detection[n] = np.nan
                detection[n-1] = np.nan
                triggered = True

        # detect events based on threshold
        for n, a in enumerate(ratio):
            if a > threshold_upper:
                detection[n] = np.nan
                detection[n-1] = np.nan

        return detection, ratio


    amp_threshold_lower = 0.5

    # dpha = __smooth(dpha, 10)

    # 1st derivative of amplitude
    damp = np.gradient(arr)

    # 2nd derivative
    # damp = np.gradient(damp)

    # absolute values
    damp = abs(damp)

    detect, amp_ratio = sta_lta_detect(damp, amp_threshold)

    if plot:

        import matplotlib.pyplot as plt

        Nrow, Ncol = 2, 1

        font = 12

        tscale, tunit = 1/86400, "days"
        
        fig, ax = plt.subplots(Nrow, Ncol, figsize=(12, 5), sharex=True)

        plt.subplots_adjust(hspace=0.1)

        ax[0].plot(times*tscale, arr, label=f"$\delta$f w/ jumps")
        ax[0].plot(times*tscale, arr*np.array(detect), label=f"$\delta$f w/o jumps")

        ax[1].plot(times*tscale, amp_ratio*1e6, "k", label=f"phase ratio (x10$^6$)")

        ax[1].axhline(amp_threshold*1e6, color="darkred", alpha=0.5, ls="--", label=f"detection threshold")
        ax[1].set_ylim(0, 3*amp_threshold*1e6)

        for _k in range(Nrow):
            ax[_k].grid(which="both", ls="--", color="grey", alpha=0.5, zorder=0)
            ax[_k].legend(loc=1, fontsize=font-2)

        for _n, d in enumerate(detect):
            if np.isnan(d):
                ax[0].axvline(times[_n]*tscale, 0, np.nanmax(arr)*2, color="grey", alpha=0.2, zorder=1)
                ax[1].axvline(times[_n]*tscale, 0, 10, color="grey", alpha=0.2, zorder=1)

        ax[0].ticklabel_format(useOffset=False)
        ax[0].set_ylabel(f"$\delta$f (Hz)", fontsize=font)

        ax[1].set_ylabel("Phase Ratio", fontsize=font)
        ax[1].set_xlabel(f"Time ({tunit})", fontsize=font)
        plt.show();

    # change detections to one and everything else to zero
    detect2 = abs(np.nan_to_num(detect, 0) - 1)

    if plot:
        return np.array(amp_ratio), np.array(detect2), fig
    else:
        return np.array(amp_ratio), np.array(detect2)