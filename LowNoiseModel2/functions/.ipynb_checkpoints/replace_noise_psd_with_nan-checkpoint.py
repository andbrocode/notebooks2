def __replace_noisy_psds_with_nan(arr, threshold_mean=1e-16, ff=None, flim=None):

    from numpy import delete, shape, sort, array, ones, nan, nanmean, array

    idx_min = 0
    idx_max = arr.shape[1]-1

    if flim is not None and ff is not None:
        for n, f in enumerate(ff):
            if f > flim[0]:
                if n == 0:
                    idx_min = 0
                else:
                    idx_min = n-1
                break
        for n, f in enumerate(ff):
            if f > flim[1]:
                idx_max = n
                break

    l1 = shape(arr)[0]

    rejected = []
    for ii in range(shape(arr)[0]):

        ## appy upper threshold
        if ff is not None:
            if nanmean(arr[ii, idx_min:idx_max]) > threshold_mean:
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan


    l2 = len(rejected)

    print(f" -> removed {l1-l2} rows due to mean thresholds ({round(ff[idx_min],4)} and {round(ff[idx_max],4)} Hz)!")
    print(f" -> {l2} / {l1} psds removed")

    return arr, array(rejected)