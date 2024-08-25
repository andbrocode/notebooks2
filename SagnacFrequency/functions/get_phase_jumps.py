def __get_phase_jumps(arr, time, fraction_of_pi=6, plot=True):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert

    data = arr.copy()

    # define threshold for phase jump identification
    threshold = np.pi/fraction_of_pi

    # remove mean of data to avoid jumps at start and end
    data -= np.nanmean(data)

    # avoid having to deal with nan values
    data = np.nan_to_num(data, nan=0)

    # compute Hilbert transform
    H = hilbert(data)

    # obtain angle of hilibert
    # pha = np.angle(H)
    pha = np.unwrap(np.angle(H))

    # derivative of angle
    dpha = np.gradient(pha, edge_order=1)

    # dpha /= np.real(H)
    # dpha /= max(abs(pha))
    # dpha /= sum((np.gradient(test)))

    # detect phase changes
    dpha_y = [_x if abs(_x) > threshold and abs(_x) < 3.14 else np.nan for _x in dpha]

    # detect phase indices
    dpha_idx = [list(dpha).index(_x) for _x in dpha if abs(_x) > threshold and abs(_x) < 3.14]

    # apply filter
    out = [np.nan if _i in dpha_idx else arr[_i] for _i in range(len(arr))]

    # checkup plot
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(15, 8))

        ax[0].plot(time, dpha)
        ax[0].scatter(time, dpha_y, color="tab:orange", s=10, zorder=2)
        ax[0].axhline(threshold, color="red", ls="--", alpha=0.5)
        ax[0].axhline(-threshold, color="red", ls="--", alpha=0.5)

        ax[1].plot(time, data)
        for line in time[dpha_idx]:
            ax[1].axvline(line, -1000, 1000, zorder=0, alpha=0.3, color="tab:orange")

        plt.show();

    return np.array(out), time[dpha_idx], np.array(dpha_y)[dpha_idx], np.array(dpha_idx)