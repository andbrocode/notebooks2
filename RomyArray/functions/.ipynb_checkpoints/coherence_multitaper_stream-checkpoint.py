def __coherence_multitaper_stream(st, reference, n_win=5, time_bandwidth=3.5):

    import multitaper as mt
    import multitaper.utils as utils

    net, sta, loc, cha = reference.split(".")
    ref = st.select(network=net, station=sta, location=loc)[0]

    out = {}
    frequencies, coherences = [], []

    for tr in st:
        ## Compute multitaper PSDs
        Psd1 = mt.MTSpec(tr.data, nw=time_bandwidth, kspec=n_win, dt=tr.stats.delta, iadapt=2)
        Psd2 = mt.MTSpec(ref.data, nw=time_bandwidth, kspec=n_win, dt=tr.stats.delta, iadapt=2)

        ## extract the frequencies and PSDs
        _f1, _psd1 = Psd1.rspec()
        out['f1'], out['psd1'] = _f1.reshape(_f1.size), _psd1.reshape(_psd1.size)

        _f2, _psd2 = Psd2.rspec()
        out['f2'], out['psd2'] = _f2.reshape(_f2.size), _psd2.reshape(_psd2.size)

        ## cross-correlation, coherence, deconvolution
        P12 = mt.MTCross(Psd1, Psd2, wl=0.001)
        N = P12.freq.size
        out['fcoh'], out['coherence'] = P12.freq[:,0][:N//2], P12.cohe[:,0][:N//2]

        frequencies.append(out['fcoh'])
        coherences.append(out['coherence'])

    return frequencies, coherences