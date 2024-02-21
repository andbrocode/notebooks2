def __replace_noisy_psds_with_nan(arr, ff=None, threshold_mean=1e-16, flim=[None, None], threshold_min=None, threshold_max=None):

    from numpy import delete, shape, sort, array, ones, nan, nanmean, array, isnan

    idx_min = 0
    idx_max = arr.shape[1]-1

    if flim[0] is not None and flim[1] is not None and ff is not None:

        flim = array(flim)

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

        if isnan(arr[ii, :]).all():
            rejected.append(arr[ii, :])
            continue

        ## appy upper threshold
        if ff is not None:
            if nanmean(arr[ii, idx_min:idx_max]) > threshold_mean:
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

        ## appy minimum threshold
        if ff is not None and threshold_min is not None:
            if any(arr[ii, :] < threshold_min):
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

        ## appy maximum threshold
        if ff is not None and threshold_max is not None:
            if any(arr[ii, :] > threshold_max):
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

    l2 = len(rejected)

    print(f" -> removed {l2} rows due to mean thresholds ({round(ff[idx_min],4)} and {round(ff[idx_max],4)} Hz)!")
    print(f" -> {l1-l2} / {l1} psds remain")

    return arr, array(rejected)