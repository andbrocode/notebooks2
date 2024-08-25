def __replace_noisy_psds_with_nan(arr, ff, threshold_mean=None, flim=[None, None], threshold_min=None, threshold_max=None):

    from numpy import delete, shape, sort, array, ones, nan, nanmean, array, isnan

    arr = array(arr)

    idx_min = 0
    idx_max = arr.shape[1]-1

    if flim[0] is not None and flim[1] is not None:

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

    rejected, all_nan = [], []
    for ii in range(shape(arr)[0]):

        if isnan(arr[ii, :]).all():
            all_nan.append(arr[ii, :])
            continue

        ## appy upper threshold
        if threshold_mean is not None:
            if nanmean(arr[ii, idx_min:idx_max]) > threshold_mean:
                # print("mean threshold")
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

        ## appy minimum threshold
        if threshold_min is not None:
            if any(arr[ii, :] < threshold_min):
                # print("min threshold")
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

        ## appy maximum threshold
        if threshold_max is not None:
            if any(arr[ii, :] > threshold_max):
                # print("max threshold")
                rejected.append(arr[ii, :])
                arr[ii, :] = ones(shape(arr)[1]) * nan

    l2 = len(rejected)
    l3 = len(all_nan)

    print(f" -> {l3} are all NaN")
    print(f" -> {l2} rows removed due to mean thresholds ({round(ff[idx_min],4)} and {round(ff[idx_max],4)} Hz)!")
    print(f" -> {l1-l2-l3} / {l1} psds remain")

    return arr, rejected