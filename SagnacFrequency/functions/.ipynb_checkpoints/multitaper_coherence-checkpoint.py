def __multitaper_coherence(dat1, dat2, Tdelta, n_taper=5, time_bandwidth=3.5, method=2):

    from multitaper import MTSpec, MTCross

    psd_st1 = MTSpec(dat1,
                     dt=Tdelta,
                     nw=time_bandwidth,
                     kspec=n_taper,
                     iadapt=method,
                    )

    _f1, _psd1 = psd_st1.rspec()
    f1, psd1 = _f1.reshape(_f1.size), _psd1.reshape(_psd1.size)


    psd_st2 = MTSpec(dat2,
                     dt=Tdelta,
                     nw=time_bandwidth,
                     kspec=n_taper,
                     iadapt=method,
                    )

    _f2, _psd2 = psd_st2.rspec()
    f2, psd2 = _f2.reshape(_f2.size), _psd2.reshape(_psd2.size)

    if psd1.size == psd2.size:
        Pxy  = MTCross(psd_st1, psd_st2, wl=0.001)
        N = Pxy.freq.size
        ff_coh, coh = Pxy.freq[:,0][:N//2+1], Pxy.cohe[:,0][:N//2+1]
    else:
        print(dat1.size, dat2.size, psd1.size, psd2.size)

    ## output dict
    out = {}
    out['ff1'] = f1
    out['ff2'] = f2
    out['psd1'] = psd1
    out['psd2'] = psd2
    out['fcoh'] = ff_coh
    out['coh'] = coh

    return out