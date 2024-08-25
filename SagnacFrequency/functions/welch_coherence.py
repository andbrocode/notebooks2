def __welch_coherence(dat1, dat2, Tdelta, twin_sec=10):

    from scipy.signal import welch
    from scipy.signal.windows import hann
    from scipy.signal import coherence


    nblock = int(1/Tdelta * twin_sec)
    overlap = int(0.5*nblock)
    win = hann(nblock, True)

    ff1, Pxx1 = welch(dat1,
                      fs=1/Tdelta,
                      window=win,
                      noverlap=overlap,
                      nperseg=nblock,
                      scaling="density",
                      return_onesided=True
                     )

    ff2, Pxx2 = welch(dat2,
                      fs=1/Tdelta,
                      window=win,
                      noverlap=overlap,
                      nperseg=nblock,
                      scaling="density",
                      return_onesided=True
                     )

    ffxy, pxy = coherence(dat1,
                          dat2,
                          fs=1/Tdelta,
                          window=win,
                          nperseg=nblock,
                          noverlap=overlap,
                          detrend='linear',
                          axis=- 1
                         )


    ## output dict
    out = {}
    out['ff1'] = ff1
    out['ff2'] = ff2
    out['psd1'] = Pxx1
    out['psd2'] = Pxx2
    out['fcoh'] = ffxy
    out['coh'] = pxy

    return out