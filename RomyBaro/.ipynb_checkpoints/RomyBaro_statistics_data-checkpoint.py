#!/bin/python3

import os
import sys
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from obspy.clients.fdsn import Client
from obspy import read_inventory
from obspy.signal.cross_correlation import correlate, xcorr_max
from pandas import DataFrame
from scipy.signal import hilbert

from andbro__read_sds import __read_sds
from andbro__load_FURT_stream import __load_furt_stream

from functions.get_mean_promy_pressure import __get_mean_promy_pressure
from functions.get_mean_rmy_pressure import __get_mean_rmy_pressure
from functions.get_time_intervals import __get_time_intervals
from functions.estimate_linear_coefficients import __estimate_linear_coefficients
from functions.variance_reduction import __variance_reduction
from functions.smoothing import __smooth
from functions.regression import __regression

import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/ontap-ffb-bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'

# _____________________________________________________

config = {}

# output path for figures
config['path_to_figs'] = data_path+"romy_baro/outfigs/"

# path to data sds
config['path_to_sds'] = archive_path+"temp_archive/"

# path to output data
config['path_to_out_data'] = data_path+"romy_baro/data2/"

# data
if len(sys.argv) > 1:
    config['tbeg'] = obs.UTCDateTime(sys.argv[1])
    config['tend'] = config['tbeg'] + 86400
else:
    config['tbeg'] = obs.UTCDateTime("2024-06-01 00:00")
    config['tend'] = obs.UTCDateTime("2024-06-30 00:00")

config['tbuffer'] = 3600 # 7200 # seconds

# ROMY coordinates
config['sta_lon'] = 11.275501
config['sta_lat'] = 48.162941

# define frequency range
config['fmin'], config['fmax'] = 0.0005, 0.01

# define time intervals
config['interval_seconds'] = 7200 # 10800
config['interval_overlap'] = 1800 # 3600

# _____________________________________________________

def main(config):

    warnings.filterwarnings('ignore')

    # define data frame
    df = DataFrame()

    # pre-define time intervals
    times = __get_time_intervals(config['tbeg'],
                                 config['tend'],
                                 interval_seconds=config['interval_seconds'],
                                 interval_overlap=config['interval_overlap']
                                )

    NN = len(times)

    arr_t1, arr_t2 = [], []
    arr_shift_PP_Z, arr_shift_PP_N, arr_shift_PP_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    arr_shift_HP_Z, arr_shift_HP_N, arr_shift_HP_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)

    arr_ccmax_PP_Z, arr_ccmax_PP_N, arr_ccmax_PP_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    arr_ccmax_HP_Z, arr_ccmax_HP_N, arr_ccmax_HP_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)

    arr_a_Z, arr_a_N, arr_a_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    arr_b_Z, arr_b_N, arr_b_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    arr_R_Z, arr_R_N, arr_R_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)

    arr_w_dir, arr_w_vel = np.zeros(NN), np.zeros(NN)

    reg_a_Z, reg_a_N, reg_a_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    reg_b_Z, reg_b_N, reg_b_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)
    reg_R_Z, reg_R_N, reg_R_E = np.zeros(NN), np.zeros(NN), np.zeros(NN)

    status = []

    arr_t1, arr_t2 = np.zeros(len(times)), np.zeros(len(times))

    print(config['tbeg'].date)

    for _n, (t1, t2) in enumerate(tqdm(times)):

        # print(t1, t2)

        arr_t1[_n] = t1
        arr_t2[_n] = t2

        stop = False

        try:

            # ___________________________________________________________
            # load ROMY data
            st0 = obs.Stream()
            st0 += __read_sds(config['path_to_sds'], "BW.ROMY.30.BJZ", t1-config['tbuffer'], t2+config['tbuffer'])
            st0 += __read_sds(config['path_to_sds'], "BW.ROMY.30.BJN", t1-config['tbuffer'], t2+config['tbuffer'])
            st0 += __read_sds(config['path_to_sds'], "BW.ROMY.30.BJE", t1-config['tbuffer'], t2+config['tbuffer'])

            # print(st0)

            # check if any is masked
            if len(st0) > 3:
                print(f" -> masked array(s)")
                stop = True
            else:
                stop = False

#             for tr in st0:
#                 if np.ma.isMaskedArray(tr.data):
#                     print(f" -> masked array: {tr.stats.channel}")
#                     stop = True
#                     continue
#                 else:
#                     stop = False

            status.append(stop)

            # jump if traces are masked
            # if stop:
            #     continue

            st0 = st0.detrend("linear")
            st0 = st0.detrend("demean")

            # st0 = st0.merge(fill_value="interpolate")
            st0 = st0.merge(fill_value=0)

            # print(st0)
            # st0.plot();

            # ___________________________________________________________
            # load barometer data
            ffbi_inv = read_inventory(root_path+"/Documents/ROMY/ROMY_infrasound/station_BW_FFBI.xml")

#             ffbi0 = __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDF", t1-config['tbuffer'], t2+config['tbuffer'])

#             if len(ffbi0) != 2:
#                 ffbi0 = ffbi0.merge();

#             ffbi0 += __read_sds(bay_path+"mseed_online/archive/", "BW.FFBI..BDO", t1-config['tbuffer'], t2+config['tbuffer'])
#             for tr in ffbi0:
#                 if "F" in tr.stats.channel:
#                     tr = tr.remove_response(ffbi_inv, water_level=10)
#                 if "O" in tr.stats.channel:
#                     tr.data = tr.data /1.0 /6.28099e5 /1e-5   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 100 mV/hPa
            # ffbi0 = ffbi0.decimate(2)

            ffbi0 = __read_sds(archive_path+"temp_archive/", "BW.FFBI.30.LDF", config['tbeg'], config['tend'])
            ffbi0 += __read_sds(archive_path+"temp_archive/", "BW.FFBI.30.LDO", config['tbeg'], config['tend'])

            ffbi0 = ffbi0.merge();

            # ___________________________________________________________
            # load promy pressure data
            # promy = __get_mean_promy_pressure(["03", "04", "05", "07", "09"],
            #                                   t1-config['tbuffer'],
            #                                   t2+config['tbuffer'],
            #                                   plot=True
            #                                   )

            # ___________________________________________________________
            # load rmy pressure data
            # brmy = __get_mean_rmy_pressure(["PROMY", "ALFT", "TON", "BIB", "GELB", "GRMB"],
            #                                t1-config['tbuffer'],
            #                                t2+config['tbuffer'],
            #                                plot=True
            #                               )

            # ___________________________________________________________
            # integrate ROMY data to tilt

            til1 = st0.copy()
            # til1 = til1.integrate()
            # til1 = til1.detrend("demean")
            til1 = til1.integrate("spline")

            #st0.plot(equal_scale=False)
            #til1.plot(equal_scale=False)

            # ___________________________________________________________
            # prepare stream

            stt = obs.Stream()
            stt += til1.copy()
            stt += ffbi0.copy()

            # stt.plot(equal_scale=False)

            # del st0, til1, ffbi0

            # resample to 1 Hz
            # stt = stt.decimate(2, no_filter=False)
            # stt = stt.decimate(5, no_filter=False)
            # stt = stt.resample(10.0, no_filter=True)
            # stt = stt.resample(1.0, no_filter=True)

            # stt += promy.copy()
            # stt += rmy.copy()

            # pre-process data
            stt = stt.detrend("demean")
            stt = stt.detrend("linear")
            # stt = stt.detrend("simple")

            stt = stt.taper(0.05, type="cosine")

            stt = stt.filter("bandpass", freqmin=config['fmin'], freqmax=config['fmax'], corners=4, zerophase=True);

            stt = stt.trim(t1, t2, nearest_sample=False)

            # downsample to LJ*
            for tr in stt:
                if "J" in tr.stats.channel:
                    tr = tr.decimate(2, no_filter=True)
                    tr = tr.decimate(10, no_filter=True)

            # stt.plot(equal_scale=False);

            stt = stt.taper(0.05, type="cosine")

            # check if all data for this period is there
            if len(stt) != 5:
                print(f" -> missing data")
                print(stt)
                stop = True
                # continue

            # check if data has same length
            Nexpected = int((t2 - t1)*stt[0].stats.sampling_rate)
            for tr in stt:
                Nreal = len(tr.data)
                if Nreal != Nexpected:
                    tr.data = tr.data[:Nexpected]
                    # print(f" -> adjust length: {tr.stats.station}.{tr.stats.channel}:  {Nreal} -> {Nexpected}")

            # prepare arrays
            arrHP = np.imag(hilbert(stt.select(component="O")[0].data))
            arrPP = stt.select(component="O")[0].data
            arrN = stt.select(component="N")[0].data
            arrE = stt.select(component="E")[0].data
            arrZ = stt.select(component="Z")[0].data

            # number of samples
            Nshift = len(arrN)

            # extract sampling rate in seconds
            dt = stt[0].stats.delta

            # compute cross-correlation function
            ccf_PP_N = correlate(arrPP, arrN, shift=Nshift, demean=False, normalize='naive', method='fft')
            ccf_HP_N = correlate(arrHP, arrN, shift=Nshift, demean=False, normalize='naive', method='fft')
            ccf_PP_E = correlate(arrPP, arrE, shift=Nshift, demean=False, normalize='naive', method='fft')
            ccf_HP_E = correlate(arrHP, arrE, shift=Nshift, demean=False, normalize='naive', method='fft')
            ccf_PP_Z = correlate(arrPP, arrZ, shift=Nshift, demean=False, normalize='naive', method='fft')
            ccf_HP_Z = correlate(arrHP, arrZ, shift=Nshift, demean=False, normalize='naive', method='fft')

            # compute lag time
            cclags = np.arange(-Nshift, Nshift+1) * dt

            # extract maximum CC and lag time
            shift_PP_N, ccmax_PP_N = xcorr_max(ccf_PP_N)
            shift_HP_N, ccmax_HP_N = xcorr_max(ccf_HP_N)
            shift_PP_E, ccmax_PP_E = xcorr_max(ccf_PP_E)
            shift_HP_E, ccmax_HP_E = xcorr_max(ccf_HP_E)
            shift_PP_Z, ccmax_PP_Z = xcorr_max(ccf_PP_Z)
            shift_HP_Z, ccmax_HP_Z = xcorr_max(ccf_HP_Z)

            # ___________________________________________________________
            # estimation of pressure model
            c2 = "*DO" # pressure
            c3 = "*DO" # hilbert of pressure

            a_Z, b_Z, hh_Z, res_Z = __estimate_linear_coefficients(stt, c1="BJZ", c2=c2, c3=c3)

            # Vertical
            dd_Z = stt.select(component="Z")[0].data
            pp_Z = stt.select(channel=c2)[0].data

            R_Z = __variance_reduction(dd_Z, res_Z)

            tt_Z = stt.select(component="Z")[0].times()

            # North
            a_N, b_N, hh_N, res_N = __estimate_linear_coefficients(stt, c1="BJN", c2=c2, c3=c3)

            dd_N = stt.select(component="N")[0].data
            pp_N = stt.select(channel=c2)[0].data

            R_N = __variance_reduction(dd_N, res_N)

            tt_N = stt.select(component="N")[0].times()

            # East
            a_E, b_E, hh_E, res_E = __estimate_linear_coefficients(stt, c1="BJE", c2=c2, c3=c3)

            dd_E = stt.select(component="E")[0].data
            pp_E = stt.select(channel=c2)[0].data

            R_E = __variance_reduction(dd_E, res_E)

            tt_E = stt.select(component="E")[0].times()

            # ___________________________________________________________
            # add values to arrays
            arr_a_Z[_n], arr_a_N[_n], arr_a_E[_n] = a_Z, a_N, a_E

            arr_shift_PP_N[_n], arr_shift_PP_E[_n], arr_shift_PP_Z[_n] = shift_PP_N, shift_PP_E, shift_PP_Z
            arr_shift_HP_N[_n], arr_shift_HP_E[_n], arr_shift_HP_Z[_n] = shift_HP_N, shift_HP_E, shift_HP_Z

            arr_ccmax_PP_N[_n], arr_ccmax_PP_E[_n], arr_ccmax_PP_Z[_n] = ccmax_PP_N, ccmax_PP_E, ccmax_PP_Z
            arr_ccmax_HP_N[_n], arr_ccmax_HP_E[_n], arr_ccmax_HP_Z[_n] = ccmax_HP_N, ccmax_HP_E, ccmax_HP_Z

            arr_a_Z[_n], arr_a_N[_n], arr_a_E[_n] = a_Z, a_N, a_E
            arr_b_Z[_n], arr_b_N[_n], arr_b_E[_n] = b_Z, b_N, b_E
            arr_R_Z[_n], arr_R_N[_n], arr_R_E[_n] = R_Z, R_N, R_E

        except Exception as e:
            print(" -> processing failed")
            print(e)
            stop = True
            # continue

        # ___________________________________________________________
        # perform multi-variant regression
        reg_type = "ransac"
        try:

            dff = DataFrame()

            dff['time'] = stt[0].times()
            dff['ffbPP'] = stt.select(station="FFBI", channel="*O")[0].data
            dff['ffbHP'] = np.imag(hilbert(stt.select(station="FFBI", channel="*O")[0].data))

            # dff['rmyPP'] = stt.select(station="RMY", channel="*O")[0].data
            # dff['rmyHP'] = np.imag(hilbert(stt.select(station="RMY", channel="*O")[0].data))

            for c in ["N", "E", "Z"]:
                dff[c] = stt.select(station="ROMY", location="30", channel=f"*{c}")[0].data

            # model Z ffbi
            outZ = __regression(dff, ['ffbPP', 'ffbHP'], target='Z', reg=reg_type, verbose=False)

            xx_Z0 = outZ['slope'][0]*dff['ffbPP'] + outZ['slope'][1]*dff['ffbHP']
            re_Z0 = dff['Z'] - xx_Z0
            vr_Z0 = __variance_reduction(xx_Z0, dff['Z'] - xx_Z0)
            ra_Z0 = round(outZ['slope'][0]/outZ['slope'][1], 3)

            # model N ffbi
            outN = __regression(dff, ['ffbPP', 'ffbHP'], target='N', reg=reg_type, verbose=False)

            xx_N0 = outN['slope'][0]*dff['ffbPP'] + outN['slope'][1]*dff['ffbHP']
            re_N0 = dff['N'] - xx_N0
            vr_N0 = __variance_reduction(xx_N0, dff['N'] - xx_N0)
            ra_N0 = round(outN['slope'][0]/outN['slope'][1], 3)

            # model E ffbi
            outE = __regression(dff, ['ffbPP', 'ffbHP'], target='E', reg=reg_type, verbose=False)

            xx_E0 = outE['slope'][0]*dff['ffbPP'] + outE['slope'][1]*dff['ffbHP']
            re_E0 = dff['E'] - xx_E0
            vr_E0 = __variance_reduction(xx_E0, dff['E'] - xx_E0)
            ra_E0 = round(outE['slope'][0]/outE['slope'][1], 3)

            # add to arrays
            reg_a_Z[_n], reg_a_N[_n], reg_a_E[_n] = outZ['slope'][0], outN['slope'][0], outE['slope'][0]
            reg_b_Z[_n], reg_b_N[_n], reg_b_E[_n] = outZ['slope'][1], outN['slope'][1], outE['slope'][1]
            reg_R_Z[_n], reg_R_N[_n], reg_R_E[_n] = vr_Z0, vr_N0, vr_E0

        except Exception as e:
            print(" -> regression processing failed")
            print(e)
            stop = True

        # ___________________________________________________________
        # load furt for wind direction and velocity
        try:
            furt = __load_furt_stream(config['tbeg'], config['tend'],
                                      path_to_archive=bay_path+'gif_online/FURT/WETTER/'
                                     )

            # ___________________________________________________________
            # get wind velocity estimate

            dx = 1

            wind_vel = furt.select(channel="LAW")[0].data

            wind_vel_smooth = __smooth(wind_vel, 60)

            _hist = np.histogram(wind_vel_smooth, bins=int(15/dx), range=(0, 15))

            wind_vel_mean = _hist[1][np.argmax(_hist[0])] + dx/2

            arr_w_vel[_n] = wind_vel_mean

            # ___________________________________________________________
            # get wind direction estimate

            dx = 10

            wind_dir = furt.select(channel="LAD")[0].data

            wind_dir_smooth = __smooth(wind_dir, 60)

            _hist = np.histogram(wind_dir_smooth, bins=int(360/dx), range=(0, 360))

            wind_dir_mean = _hist[1][np.argmax(_hist[0])] + dx/2

            arr_w_dir[_n] = wind_dir_mean

        except Exception as e:
            print(" -> FURT failed")
            print(e)

        # check if stop is required
        if stop:
            # del stt
            continue

        # ___________________________________________________________
        # plotting
        try:

            Nrow, Ncol = 6, 1

            fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 10), sharex=True)

            font = 12

            yscale, yunit = 1e9, "nrad"

            tscale, tunit = 1/60, "min"

            dsig = r"$\Delta \sigma$"

            y_max = max([max(abs(dd_N*yscale)), max(abs(hh_N*yscale))])

            ax[0].plot(tt_N*tscale, dd_N*yscale, label="ROMY-N")
            ax[0].plot(tt_N*tscale, hh_N*yscale, label=f"P/H[P] = {round(a_N/b_N, 3)}")
            # ax[0].plot(tt_N*tscale, hh_N*yscale, label=f"{round(a_N*1e12, 2)}e12 * P+{round(b_N*1e12, 2)}e12 * H[P]")
            ax[0].set_ylim(-y_max, y_max)
            ax[0].set_ylabel(f"Tilt ({yunit})", fontsize=font)

            ax[1].plot(tt_N*tscale, ( dd_N - hh_N )*yscale, color="grey", label=f"{dsig}={R_N}%")
            ax[1].set_ylim(-y_max, y_max)
            ax[1].set_ylabel(f"Residual ({yunit})", fontsize=font)


            y_max = max([max(abs(dd_E*yscale)), max(abs(hh_E*yscale))])

            ax[2].plot(tt_E*tscale, dd_E*yscale, label="ROMY-E")
            ax[2].plot(tt_E*tscale, hh_E*yscale, label=f"P/H[P] = {round(a_E/b_E, 3)}")
            ax[2].set_ylim(-y_max, y_max)
            ax[2].set_ylabel(f"Tilt ({yunit})", fontsize=font)

            ax[3].plot(tt_E*tscale, ( dd_E - hh_E )*yscale, color="grey", label=f"{dsig}={R_E}%")
            ax[3].set_ylim(-y_max, y_max)
            ax[3].set_ylabel(f"Residual ({yunit})", fontsize=font)


            y_max = max([max(abs(dd_Z*yscale)), max(abs(hh_Z*yscale))])

            ax[4].plot(tt_Z*tscale, dd_Z*yscale, label="ROMY-Z")
            ax[4].plot(tt_Z*tscale, hh_Z*yscale, label=f"P/H[P] = {round(a_Z/b_Z, 3)}")
            ax[4].set_ylim(-y_max, y_max)
            ax[4].set_ylabel(f"Tilt ({yunit})", fontsize=font)

            ax[5].plot(tt_Z*tscale, ( dd_Z - hh_Z )*yscale, color="grey", label=f"{dsig}={R_Z}%")
            ax[5].set_ylim(-y_max, y_max)
            ax[5].set_ylabel(f"Residual ({yunit})", fontsize=font)


            ax[Nrow-1].set_xlabel(f"Time ({tunit})", fontsize=font)

            ax[0].set_title(f" {t1.date} {str(t1.time).split('.')[0]} - {str(t2.time).split('.')[0]} UTC  |  f = {config['fmin']*1e3} - {config['fmax']*1e3} mHz", fontsize=font)


            for i in range(Nrow):
                ax[i].legend(loc=1, ncol=2)

            for _k, ll in enumerate(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
                ax[_k].text(.005, .97, ll, ha='left', va='top', transform=ax[_k].transAxes, fontsize=font+2)

            fig.savefig(config['path_to_figs']+f"RB_{str(_n).rjust(3, '0')}_waveforms_corrected.png",
                        format="png", dpi=150, bbox_inches='tight')

        except Exception as e:
            print(" -> plotting failed")
            print(e)

    df['t1'] = arr_t1
    df['t2'] = arr_t2

    df['status'] = status

    df['shift_PP_N'] = arr_shift_PP_N
    df['shift_PP_E'] = arr_shift_PP_E
    df['shift_PP_Z'] = arr_shift_PP_Z

    df['shift_HP_N'] = arr_shift_HP_N
    df['shift_HP_E'] = arr_shift_HP_E
    df['shift_HP_Z'] = arr_shift_HP_Z

    df['cmax_PP_N'] = arr_ccmax_PP_N
    df['cmax_PP_E'] = arr_ccmax_PP_E
    df['cmax_PP_Z'] = arr_ccmax_PP_Z

    df['cmax_HP_N'] = arr_ccmax_HP_N
    df['cmax_HP_E'] = arr_ccmax_HP_E
    df['cmax_HP_Z'] = arr_ccmax_HP_Z

    df['r_z'] = arr_R_Z
    df['r_n'] = arr_R_N
    df['r_e'] = arr_R_E

    df['a_z'] = arr_a_Z
    df['a_n'] = arr_a_N
    df['a_e'] = arr_a_E

    df['b_z'] = arr_b_Z
    df['b_n'] = arr_b_N
    df['b_e'] = arr_b_E

    # regression output
    df['reg_r_z'] = reg_R_Z
    df['reg_r_n'] = reg_R_N
    df['reg_r_e'] = reg_R_E

    df['reg_a_z'] = reg_a_Z
    df['reg_a_n'] = reg_a_N
    df['reg_a_e'] = reg_a_E

    df['reg_b_z'] = reg_b_Z
    df['reg_b_n'] = reg_b_N
    df['reg_b_e'] = reg_b_E

    # wind direction and velocity
    df['wvel'] = arr_w_vel
    df['wdir'] = arr_w_dir

    df.to_pickle(config['path_to_out_data']+f"RB_statistics_{config['tbeg'].date}.pkl")

# _____________________________________________________

if __name__ == "__main__":
    main(config)

# End of File
