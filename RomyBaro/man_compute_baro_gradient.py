
import os
import gc
import numpy as np

from pandas import date_range
from obspy import UTCDateTime, Stream
from functions.baro_array import baroArray
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/ontap-ffb-bay200/'
    lamont_path = '/home/andbro/lamont/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
    lamont_path = '/lamont/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
    lamont_path = '/lamont/'


config = {}

config['tbeg'] = UTCDateTime("2024-07-01")
config['tend'] = UTCDateTime("2024-10-31")


config['seeds'] = [
                   'BW.PROMY.03.LDI',
                   'BW.GELB..LDO',
                   'BW.GRMB..LDO',
                   'BW.ALFT..LDO',
                   'BW.BIB..LDO',
                   'BW.TON..LDO',
]

config['fmin'] = 1e-4
config['fmax'] = 10e-3

config['path_to_figs'] = data_path+"/romy_baro/figures/auto_gradient/"

config['coos'] = {
                   "ALFT":{"lon":11.2795 , "lat":48.142334, "height":593.0},
                   "GELB":{"lon":11.2514 , "lat":48.1629, "height":628.0},
                   "GRMB":{"lon":11.2635 , "lat":48.1406, "height":656.0},
                   "TON":{"lon":11.288809 , "lat":48.173897, "height":564.0},
                   "BIB":{"lon":11.2473 , "lat":48.1522, "height":599.0},
                   "PROMY":{"lon":11.275501 , "lat":48.162941, "height":571.0},
                }


def main(config):

    dates = date_range(config['tbeg'].date, config['tend'].date)

    for date in tqdm(dates):

        try:

            brmy = baroArray(seeds=config['seeds'],
                            coords=config['coos'],
                            out_seed="BW.BRMY..BDX",
                            )

            brmy.load_data(tbeg=UTCDateTime(date),
                        tend=UTCDateTime(date)+86400,
                        path_to_sds=archive_path+"temp_archive/",
                        verbose=False
                        )

            brmy.preprocessing(config['fmin'], config['fmax'])

            brmy.compute_gradient(reference="BW.PROMY",
                                verbose=False
                                )

            brmy.write_stream_to_sds(verbose=True)

            brmy.makeplot(plot=False)

            brmy.fig.savefig(config['path_to_figs']+f"{date}_gradient.png", format="png", dpi=150, bbox_inches='tight')
        except:
            print(f" -> failed for {date}")

if __name__ == "__main__":
    main(config)


# EOF
