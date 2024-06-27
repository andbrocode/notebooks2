def __get_mlti_statistics(mlti_times, times, plot=True, ylog=False):

    import numpy as np
    import matplotlib.pyplot as plt

    # relative times
    mlti_times_sec = np.array(mlti_times - times[0]).astype(int)

    times_sec = np.array(times - times[0]).astype(int)

    # start mlti array
    _mlti = np.zeros(len(times))

    # switch 0 to 1 for each mlti time
    _t0 = 0
    for _m in mlti_times_sec:
        for _j, _t in enumerate(times_sec):
            if _t < _t0:
                continue
            if _t >= _m:
                _mlti[_j] = 1
                _t0 = _t
                break

    # sum it up
    mlti_cumsum = np.cumsum(_mlti)

    # to percent
    mlti_cumsum = mlti_cumsum / max(mlti_cumsum) * 100

    # inter mlti times
    mlti_times_sec_shift = np.roll(mlti_times_sec, 1)
    mlti_times_sec_shift[0] = 0
    mlti_inter_sec = mlti_times_sec - mlti_times_sec_shift

    # plotting
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(np.arange(0, len(mlti_cumsum))/86400, mlti_cumsum)
        ax[1].hist(mlti_inter_sec/60, bins=int(1440/10), range=(0, 1440))

        if ylog:
            ax[1].set_yscale("log")

        ax[0].grid(color="grey", ls="--", alpha=0.4)
        ax[1].grid(color="grey", ls="--", alpha=0.4)

        ax[0].set_xlabel("Time (days)", fontsize=12)
        ax[0].set_ylabel("Amount MLTI (%)", fontsize=12)

        ax[1].set_xlabel("Inter-MLTI-Time (min)", fontsize=12)
        ax[1].set_ylabel("Amount MLTI", fontsize=12)

        plt.show();

    return mlti_cumsum, mlti_inter_sec