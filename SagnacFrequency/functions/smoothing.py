def __smooth(y, npts, win="hanning", setpad=True):

    from numpy import ones, convolve, hanning, nan, pad

    if win == "hanning":
        win = hanning(npts)
    elif win == "boxcar":
        win = ones(npts)

    if setpad:
        y = pad(y, npts)

    y_smooth = convolve(y, win/sum(win), mode='same')

    if setpad:
        y_smooth = y_smooth[npts:-npts]

    y_smooth[:npts//2] = nan
    y_smooth[-npts//2:] = nan

    return y_smooth