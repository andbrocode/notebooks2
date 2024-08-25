def __coherence_welch_stream(st, reference_trace, nseg=512, nover=128):

    '''

    VARIABLES:
     - st:          stream: containing all data to compare to reference
     - refernece    trace: defining the reference for the coherence

    DEPENDENCIES:
     - from scipy.signal import coherence

    OUTPUT:
     - frequencies, coherences

    EXAMPLE:
    >>> f, c = __coherence_stream(st, reference_trace, nseg=512, nover=128)

    '''

    from scipy.signal import coherence

    frequencies, coherences = [], []
    for tr in st:
        ff, pxy = coherence(tr.data,
                            reference_trace,
                            fs=tr.stats.sampling_rate,
                            window='hann',
                            nperseg=nseg,
                            noverlap=nover,
                            nfft=None,
                            detrend='constant',
                            axis=- 1
                           )

        frequencies.append(ff)
        coherences.append(pxy)

    return frequencies, coherences