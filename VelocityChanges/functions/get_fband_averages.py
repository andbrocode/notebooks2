def __get_fband_averages(_ff, _psds):

    from functions.get_fband_average import __get_fband_average
    from numpy import array

    psds = []
    for _n, _psd in enumerate(_psds):
        out0 = __get_fband_average(_ff, _psd, faction_of_octave=12 , average="median", plot=False)
        psds.append(out0['psd_means'])

    ff = out0['fcenter']
    psds = array(psds)

    return ff, psds