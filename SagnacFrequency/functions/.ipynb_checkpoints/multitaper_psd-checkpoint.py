def __multitaper_psd(arr, dt, n_win=5):

    import multitaper as mt

    out_psd = mt.MTSpec(arr, nw=n_win, kspec=0, dt=dt)

    _f, _psd = out_psd.rspec()

    f = _f.reshape(_f.size)
    psd = _psd.reshape(_psd.size)

    ## 95% confidence interval
    # _psd95 = out_psd.jackspec()
    # psd95_lower, psd95_upper = psd95[::2, 0], psd95[::2, 1]

    return f, psd