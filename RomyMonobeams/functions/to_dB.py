def __to_dB(arr, power=True):
    from numpy import log10
    if power:
        return 10*log10(arr)
    else:
        return 20*log10(arr)