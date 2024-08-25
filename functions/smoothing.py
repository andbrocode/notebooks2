def __smooth(y, npts):

    from numpy import ones, convolve, hanning, nan

    win = hanning(npts)
    y_smooth = convolve(y, win/sum(win), mode='same')

    y_smooth[:npts//2] = nan
    y_smooth[-npts//2:] = nan
    return y_smooth