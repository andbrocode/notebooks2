def __find_max_min(lst, pp=99, perc=0, add_percent=None):

    from numpy import nanpercentile

    maxs, mins = [], []

    for l in lst:
        maxs.append(nanpercentile(l, pp))
        mins.append(nanpercentile(l, 100-pp))

    if perc == 0:
        out_min, out_max = min(mins), max(maxs)
    else:
        _min = min(mins)
        _max = max(maxs)
        xx = _max*(1+perc) -_max
        out_min, out_max = _min-xx, _max+xx

    if add_percent is None:
        return out_min, out_max
    else:
        return out_min-out_min*add_percent, out_max+out_max*add_percent