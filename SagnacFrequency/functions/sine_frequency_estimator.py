def __sine_frequency_estimator(st0, nominal_sagnac, fband=2, Tinterval=20, Toverlap=2, plot=True):

    '''
    Fitting a sin-function to the data to estimate an instantaneous frequency
    '''

    import matplotlib.pyplot as plt

    from scipy import optimize
    from numpy import sin, hanning, pi, arange, array, diag, zeros, nan, isnan

    def func(x, a, f):
        return a * sin(2 * pi * f * x)

    def func(x, a, f, p):
        return a * sin(2 * pi * f * x + p)

    df = st0[0].stats.sampling_rate

    times = st0[0].times()

    # define frequency band around Sagnac Frequency
    f_lower = nominal_sagnac - fband
    f_upper = nominal_sagnac + fband

    # bandpass with butterworth around Sagnac Frequency
    st0 = st0.detrend("linear")
    # st0 = st0.taper(0.01)
    st0 = st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

    # to array
    data = st0[0].data

    # npts per interval
    Nsamples = int(Tinterval*df)
    Noverlap = int(Toverlap*df)

    # npts in data
    Ndata = data.size

    # create time reference
    tt = times

    # amount of windows
    Nwin = 0
    x2 = Nsamples
    while x2 < Ndata:
        int(Ndata / (Nsamples - Noverlap))
        x2 = x2 + Nsamples - Noverlap
        Nwin += 1

    # prepare lists
    amps = zeros(Nwin)*nan
    freq = zeros(Nwin)*nan
    phas = zeros(Nwin)*nan
    time = zeros(Nwin)*nan
    cfs = zeros(Nwin)*nan
    cas = zeros(Nwin)*nan

    # initial values
    a00 = 0.9
    f00 = nominal_sagnac
    p00 = 0

    # specify start indices
    n1, n2 = 0, Nsamples

    # looping
    for _win in range(Nwin):

        # npts = Nsamples-Noverlap

        # set start values at begin
        if _win == 0:
            a0, f0, p0 = a00, f00, p00

        # reste start values if nan
        if isnan(a0) or isnan(f0) or isnan(p0):
            a0, f0, p0 = a00, f00, p00

        _time = tt[n1:n2]
        _data = data[n1:n2]

        # xx = 0
        # cf, ca = 1, 1

        # condition for fit
        # while cf > 0.001 and ca > 0.001:
        try:
            params, params_covariance = optimize.curve_fit(func,
                                                           _time,
                                                           _data,
                                                           p0=[a0, f0, p0],
                                                           check_finite=True,
                                                          )
            f0 = params[1]
            a0 = params[0]
            p0 = params[2]

            ca, cf = diag(params_covariance)[0], diag(params_covariance)[1]

        except:
            # print(f" -> fit failed {_win}")
            f0, a0, p0, ca, cf = nan, nan, nan, nan, nan

            # # counter to avoid infinitiy loop
            # if xx > 500:
            #     break
            # else:
            #     xx += 1

        if cf > 0.001 or ca > 0.001:
            f0, a0, p0 = nan, nan, nan

        # append values
        amps[_win] = a0
        freq[_win] = f0
        phas[_win] = p0
        time[_win] = (tt[n2]-tt[n1])/2 + tt[n1]

        cfs[_win] = cf
        cas[_win] = ca

        if plot:
            if _win == Nwin - 1:
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))

                ax.plot(_time, _data, color='black')

                ax.plot(_time, func(_time, params[0], params[1], params[2]), color='red')

                plt.show();

        # update index
        n1 = n1 + Nsamples - Noverlap
        n2 = n2 + Nsamples - Noverlap

    # timeline
    # step = (Tinterval-Toverlap)
    # time = arange(step/2, Ndata/df, step)

    if plot:

        Nrow, Ncol = 3, 1

        font = 12

        fig, ax = plt.subplots(Nrow, Ncol, figsize=(12, 5), sharex=True)

        plt.subplots_adjust(hspace=0.1)

        ax[0].errorbar(time, freq, cfs)
        ax[1].errorbar(time, amps, cas)
        ax[2].plot(time, phas)

        plt.show();

    return time, freq, amps, phas