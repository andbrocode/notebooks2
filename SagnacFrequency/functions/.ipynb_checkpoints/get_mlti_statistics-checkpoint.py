def __get_mlti_statistics(mlti, starttime, endtime, plot=True, ylog=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime

    # convert to array with UTCDateTime objects
    mlti_times = np.array([UTCDateTime(_t) for _t in mlti.time_utc])

    # relative time in seconds
    mlti_times_sec = np.array(mlti_times - starttime)

    # timeline in seconds
    times_sec = np.array(np.arange(0, UTCDateTime(endtime)-UTCDateTime(starttime), 1))

    # start mlti array
    _mlti = np.zeros(len(times_sec))

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
    mlti_cumsum = mlti_cumsum
    mlti_cumsum_percent = mlti_cumsum / max(mlti_cumsum) * 100

    # inter mlti times
    mlti_times_sec_shift = np.roll(mlti_times_sec, 1)
    mlti_times_sec_shift[0] = 0
    mlti_inter_sec = mlti_times_sec - mlti_times_sec_shift

    # plotting
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(times_sec/86400, mlti_cumsum_percent)
        ax[1].hist(mlti_inter_sec/3600, bins=int(24/1), range=(0, 24), rwidth=0.8)

        if ylog:
            ax[1].set_yscale("log")

        ax[0].grid(color="grey", ls="--", alpha=0.4)
        ax[1].grid(color="grey", ls="--", alpha=0.4)

        ax[0].set_xlabel("Time (days)", fontsize=12)
        ax[0].set_ylabel("MLTI Count (%)", fontsize=12)

        ax[1].set_xlabel("Inter-MLTI-Time (hours)", fontsize=12)
        ax[1].set_ylabel("MLTI Count", fontsize=12)

        plt.show();

    output = {"cumsum":mlti_cumsum,
              "cumsumsec":mlti_cumsum*30,
              "cumsump":mlti_cumsum_percent,
              "intersec":mlti_inter_sec,
              "tsec":times_sec,
              "mlti_series":_mlti
             }

    return output