#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read_inventory
from i95_sds import I95SDSClient, I95NoDataError

i95_client = I95SDSClient('/bay200/I95_1-20Hz/')
i95_client_smooth = I95SDSClient(
                                 '/bay200/I95_1-20Hz/',
                                 smoothing_window_length_hours=24 * 7,
                                 smoothing_step_hours=12
                                )

#start = UTCDateTime(2021, 1, 1)
#end = UTCDateTime(2021, 12, 31)

start = UTCDateTime(2024, 3, 25)
end = UTCDateTime(2024, 3, 31)

inv = read_inventory('/bay200/station-metadata/output/inventory.xml')

# optionally restrict
inv = inv.select(starttime=start, endtime=end)
inv = inv.select(network='BW')
inv = inv.select(station='GRMB')
#inv = inv.select(station='SYBAD')
#inv = inv.select(station='MGSBH')

print(inv)

def plot_channel(network, station, location, channel):
    scale = 'mum/s'
    percentiles = [68, 95, 99]

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, gridspec_kw={'width_ratios': (5, 2)}, sharey=True,
        figsize=(8, 5))
    ax1.grid()
    ax2.grid()

    i95_client.plot(network, station, location, channel, start, end, ax=ax1,
                    type='line', show=False, scale=scale)
    i95_client_smooth.plot(
        network, station, location, channel, start, end, ax=ax1, type='line',
        color='r', show=False, scale=scale)
    ax1.lines[-1].set_lw(2.0)
    ax1.lines[-1].set_alpha(0.5)

    i95_client.plot(network, station, location, channel, start, end, ax=ax2,
                    type='violin', scale=scale, percentiles=percentiles,
                    show=False, verbose=True)

    ax1.set_title(f'{network}.{station}.{location}.{channel}')
    ax1.get_legend().remove()
    # ax1.set_ylim(0, 1.1)
    plt.draw()
    for filetype in ('pdf', 'png'):
        fig.savefig(
            f'/tmp/I95_{network}_{station}_{location}_{channel}.{filetype}')
    plt.close('all')


done = set()
for net in inv:
    for sta in net:
        for cha in sta:
            seed_id = '.'.join(
                (net.code, sta.code, cha.location_code, cha.code))
            if seed_id in done:
                continue
            try:
                plot_channel(net.code, sta.code, cha.location_code, cha.code)
            except I95NoDataError as e:
                print(str(e))
            finally:
                plt.close('all')
            done.add(seed_id)

print("plotting")
fig, ax = plt.subplots(figsize=(10, 4))
scale = 'mum/s'
percentiles = [68, 95, 99]
# XXX set merging to true if channel changes in the time period
# merge_streams = False
# i95_client.plot_all_data(
#     start, end, ax=ax, type='violin', percentiles=percentiles, show=False,
#     scale=scale, color='lightskyblue', violin_kwargs={'width': 1.4},
#     merge_streams=merge_streams)

# ax.grid()
# ax.set_ylim(0, 1.25)
# ax.axhline(1.0, color='blue', zorder=3)
# # fig.subplots_adjust(top=0.98, bottom=0.3)
# fig.subplots_adjust(top=0.98)
# plt.show()
# fig.savefig('/tmp/I95_ALL.pdf')
# fig.savefig('/tmp/I95_ALL.png')


fig, ax = plt.subplots(figsize=(8, 3))
# i95_client.plot_availability('BW', 'KW1', '', 'HHZ', start, end, ax=ax,
#                              show=False)
i95_client.plot_availability(start, end, ax=ax, show=False, fast=False,
                             verbose=True, vmin=60, number_of_colors=8,
                             merge_streams=True)

ax.images[0].colorbar.set_label(u'Tägliche Datenverfügbarkeit [%]')
# fig.savefig('/tmp/I95_coverage.pdf')
# fig.savefig('/tmp/I95_coverage.png')
plt.show()
