#/bin/python3

# ---------------------------------------

# run data analysis

# ---------------------------------------


from obspy import UTCDateTime
from scipy.signal import welch
from numpy import log10, zeros, pi, append, linspace, array, where, transpose, shape, histogram, arange, append, nanmedian
from numpy import logspace, linspace, log, log10, isinf, ones, nan, count_nonzero, sqrt, isnan, nanmean
from pandas import DataFrame, concat, Series, date_range, read_csv, read_pickle
from tqdm import tqdm
from pathlib import Path

import os, sys, gc
import pickle
import matplotlib.pyplot as plt

from andbro__store_as_pickle import __store_as_pickle

from functions.get_octave_bands import __get_octave_bands
from functions.replace_noise_psd_with_nan import __replace_noisy_psds_with_nan

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'

# ---------------------------------------

year = "2024"

project = "2"

path = data_path+f"LNM2/PSDS{project}/"

if len(sys.argv) > 1:
    names = [sys.argv[1]]
else:
    names = ["FFBI", "ROMY", "FUR", "DROMY", "ROMYA", "ROMYT"]

if len(sys.argv) > 2:
    t1, t2 = sys.argv[2], sys.argv[3]
else:
    t1, t2 = "2024-02-01", "2024-09-30"

codes = {"":"B", "2":"L"}

code = codes[project]

# define dates to ignore
filter_dates = {"FUR": ["20231106", "20231115"],
                "FFBI": ["20231215", "20231219", "20231220", "20231222", "20231227", "20231228", "20231229", "20231230", "20231231", "20240108", "20240110"],
                "ROMY": ["20231215", "20231219", "20231220", "20231222", "20231227", "20231228", "20231229", "20231230", "20231231", "20240106", "20240205", "20240206", "20240207"],
                "ROMYA": [],
                "DROMY": [],
                "ROMYT": [],
               }


# define rejection limits for PSDs
rejection = {"FUR": {"Z": {"tmean":1e-10, "tmin":5e-20, "flim":[0, 0.05]},
                     "N": {"tmean":1e-10, "tmin":5e-20, "flim":[0, 0.05]},
                     "E": {"tmean":1e-10, "tmin":5e-20, "flim":[0, 0.05]}
                    },
            "ROMY": {"Z": {"tmean":5e-19, "tmax":1e-16, "tmin":1e-23, "flim":[0.5, 0.9]},
                     "N": {"tmean":5e-19, "tmax":1e-16, "tmin":1e-22, "flim":[0.5, 0.9]},
                     "E": {"tmean":5e-19, "tmax":1e-16, "tmin":1e-22, "flim":[0.5, 0.9]}
                    },
            "FFBI": {"F": {"tmean":1e7, "tmin":1e-7, "flim":[0.001, 1.0]},
                     "O": {"tmean":1e7, "tmin":1e-5, "flim":[0.001, 1.0]},
                    },
            }

# ---------------------------------------

def __load_data_file(path, file):

    from tqdm.notebook import tqdm
    from numpy import array

    psds_all = []

    datafile = read_pickle(path+file)

    try:
        psds = datafile['psd']
    except:
        psds = datafile['coherence']

    ff = datafile['frequencies']

    del datafile

    for psd in psds:
        psds_all.append(psd)

    return ff, array(psds_all)

def __get_band_average(freq, data, f_center, f_upper, f_lower):

    from numpy import nanmedian

    ## get frequency indices
    fl_idx, fu_idx = [], []

    for _k, (fl, fu) in enumerate(zip(f_lower, f_upper)):
        if _k <= len(f_center):

            for _i, _f in enumerate(freq):
                if _f >= fl:
                    fl_idx.append(int(_i))
                    break

            for _i, _f in enumerate(freq):
                if _f >= fu:
                    fu_idx.append(int(_i))
                    break

    ## compute average per band
    psd_avg, fc, fu, fl = [], [], [], []
    for _n, (ifl, ifu) in enumerate(zip(fl_idx, fu_idx)):

        avg = []
        for _psd in data:
            avg.append(nanmedian(_psd[ifl:ifu]))

        psd_avg.append(array(avg))

        fc.append(f_center[_n])
        fu.append(f_upper[_n])
        fl.append(f_lower[_n])

    psd_avg = array(psd_avg)


    ## check up plot
#     plt.figure(figsize=(15, 5))

#     for _j, dd in enumerate(psd_avg):
#         plt.scatter(ones(len(dd))*fc[_j], dd, s=5)
#         plt.xscale("log")
#         plt.yscale("log")

#     plt.show();


    ## output
    out = {}
    out['psd_avg'] = psd_avg
    out['fcenter'] = array(fc)
    out['fupper'] = array(fu)
    out['flower'] = array(fl)
    out['dates'] = dates

    return out

## ---------------------------------------
## load configurations

apps = ["", f"{code}DO_coh", f"{code}DF_coh"]

for name in names:

    if name == "FUR":
        comps = ["BHZ", "BHN", "BHE"]

    elif name == "DROMY":
        comps = ["LAT", "LAN", "LAE"]

    elif name == "ROMYT":
        comps = ["MAT", "MAN", "MAE"]

    elif name == "ROMY":
        # comps = ["BJZ", "BJU", "BJV", "BJN", "BJE"]
        comps = ["BJZ", "BJN", "BJE"]
    elif name == "FFBI":
        comps = [f"{code}DF", f"{code}DO"]

    elif name == "ROMYA":
        name = "ROMY"
        comps = ["BAZ", "BAN", "BAE"]

    elif name == "BFO":
        comps = ["LDO", "LHZ", "LHN", "LHE"]

    # elif name == "BFO":
    #     comps = ["LDO"]

    print(f"\n -> {name} ...")

    for comp in tqdm(comps):

        for app in apps:

            # print(name, comp, app)

            config = {}

            # specify paths
            config['outpath_figures'] = data_path+f"LNM2/figures{project}/"

            config['path_to_outdata'] = data_path+f"LNM2/data{project}/"

            try:

                if app == "BDF_coh" and name == "BFO":
                    continue

                elif app == f"{code}DO_coh" and name == "BFO":
                    config['filename'] = f"{name}_coherence/{year}_BFO_LDO_{name}_{comp}_3600"
                    config['outname'] = f"BFO_LDO_{name}_{comp}_coherence"
                    N = 18001

                elif app == f"{code}DO_coh" and name != "FFBI":
                    config['filename'] = f"{name}_coherence/{year}_FFBI_{code}DO_{name}_{comp}_3600"
                    config['outname'] = f"FFBI_{code}DO_{name}_{comp}_coherence"
                    N = 36002

                elif app == f"{code}DF_coh" and name != "FFBI":
                    config['filename'] = f"{name}_coherence/{year}_FFBI_{code}DF_{name}_{comp}_3600"
                    config['outname'] = f"FFBI_{code}DF_{name}_{comp}_coherence"
                    N = 36002

                else:
                    config['filename'] = f"{name}/{year}_{name}_{comp}_3600"
                    config['outname'] = f"{name}_{comp}"
                    N = 36001

            except Exception as e:
                print(e)
                continue

            config['path'] = path

            config['startdate'], config['enddate'] = t1, t2

            d1, d2 = config['startdate'], config['enddate']

            psds_medians_out, times_out = [], []

            d1, d2 = config['startdate'], config['enddate']

            psds_medians_out, times_out = [], []

            dat, dates = [], []

            # if name == "DROMY":
            #     dat = ones((date_range(d1, d2).size*24, 1802))*nan
            # else:
            #     dat = ones((date_range(d1, d2).size*24, N))*nan

            # dates = ones((date_range(d1, d2).size*24))*nan


            index = 0

            missing_files_count, missing_files = 0, []

            missing_data_count, missing_data = 0, []

            for jj, day in enumerate(date_range(d1, d2)):

                day = str(day).split(" ")[0].replace("-", "")

                yr = day[:4]

                # update year if necessary
                if year != yr:
                    config['filename'] = config['filename'].replace(year, yr)

                # print(f"{config['filename']}_{day}_hourly.pkl")

                # check dates to be filtered out
                if day in filter_dates[name]:
                    # print(f" -> skip {day} due to date filter")
                    missing_data_count += 1
                    missing_data.append(day)
                    continue

                try:
                    ff, _dat = __load_data_file(config['path'], f"{config['filename']}_{day}_hourly.pkl")

                except Exception as e:
                    print(e)
                    # print(f" -> {day}: no data found")
                    missing_files_count += 1
                    missing_files.append(day)
                    continue

                try:
                    for _k, _psd in enumerate(_dat):

                        if name == "FFBI":
                            if nanmean(_psd) > 1e4:
                                print(f" -> level too high ({nanmean(_psd)} > 1e4)!")
                                dat.append(ones(len(_psd))*nan)
                                dates.append(f"{day}_{str(_k).rjust(2, '0')}")
                            else:
                                dat.append(_psd)
                                dates.append(f"{day}_{str(_k).rjust(2, '0')}")

                        else:
                            dat.append(_psd)
                            dates.append(f"{day}_{str(_k).rjust(2, '0')}")

                        index += 1

                except Exception as e:
                    print(e)
                    print(f" -> skip {day}")
                    continue

            # get frequency octave bands
            f_lower, f_upper, f_center = __get_octave_bands(1e-3, 1e0, faction_of_octave=12, plot=False)

            try:
                dat = array(dat)
            except Exception as e:
                print(e)

            try:
                out0 = __get_band_average(ff, dat, f_center, f_upper, f_lower)
            except Exception as e:
                print(e)
                continue

            # create and fill data frame
            _df_out = DataFrame()

            _df_out['dates'] = out0['dates']

            for _i, fc in enumerate(out0['fcenter']):
                _df_out[round(fc, 5)] = array(out0['psd_avg'][_i])

            df_out = _df_out.copy()

            # store as pickle file
            print(f" -> {config['outname']}.pkl")
            df_out.to_pickle(config['path_to_outdata']+f"{config['outname']}.pkl")

print("\nFINISHED")

print(f"\n{missing_files_count} files are missing:\n")
print([mf for mf in missing_files])

print(f"\n{missing_data_count} data are missing/filtered:\n")
print([md for md in missing_data])

gc.collect()

# End of File
