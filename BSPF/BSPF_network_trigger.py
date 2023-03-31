#!/usr/bin/env python
"""
USAGE: network-trigger network [starttime endtime]

Recursive STA/LTA trigger with similarity checking of template events and
coincidence network trigger with coincidence sum threshold and similarity
threshold.

Network can be UH, BE, KW.

If no start and endtime is specified, use the current time -3 hours.
"""
# 2009-07-23 Moritz; PYTHON2.5 REQUIRED
# 2009-11-25 Moritz
# 2010-09 Tobi
# 2013-04 Tobi

import os
import subprocess
import optparse
from copy import deepcopy

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

from obspy import UTCDateTime, Stream, read_inventory
from obspy.core import AttribDict
from obspy.signal import coincidenceTrigger
from obspy.seishub import Client as SeishubClient
from obspy.seedlink import Client as SeedlinkClient

from seishub_event_templates import returnTemplates


def getParameters(network):
    par = AttribDict()
    par.network = network
    par.filter = AttribDict(type="bandpass", freqmin=10.0, freqmax=20.0,
                            corners=1, zerophase=True)
    network = network.lower()
    trace_ids = {}
    if network == "uh":
        #trace_ids = {"BW.UH1..EHZ": 1, "BW.UH2..EHZ": 1,
        #             "BW.UH3..EHZ": 1, "BW.UH5..EHZ": 1}
        trace_ids = {"BW.UH1..EHZ": 1, "BW.UH2..EHZ": 1,
                     "BW.UH3..EHZ": 1,
                     "BW.MGS01..ELZ": 1, "BW.MGS02..ELZ": 1,
                     "BW.MGS03..ELZ": 1, "BW.MGS04..ELZ": 1,
                     "BW.MGS05..ELZ": 1,
                     }
        #trace_ids = {"BW.DHFO..EHZ": 1, "BW.UH1..EHZ": 1, "BW.UH2..EHZ": 1,
        #             "BW.UH3..EHZ": 1, "BW.UH4..EHZ": 0.5, "BW.UH5..EHZ": 1}
        coinc_sum = 2.5
    elif network == "kw1":
        trace_ids = {"BW.KW1..EHN": 1, "BW.KW1..EHE": 1}
        coinc_sum = 1.0
        par.inventory = read_inventory("/home/megies/bin/data/dataless.seed.BW_KW1.xml", "STATIONXML")
        par.filter = AttribDict(type="bandpass", freqmin=0.5, freqmax=30.0,
                                corners=4, zerophase=False)
    elif network == "be":
        trace_ids = {"BW.BE1..EHZ": 1, "BW.BE2..EHZ": 1, "BW.BE3..EHZ": 1,
                     "BW.BE4..EHZ": 1}
        coinc_sum = 3.0
    par.trace_ids = trace_ids
    # length of sta and lta in seconds
    par.coincidence = AttribDict(trigger_type=None,
                                 thr_on=0.05, thr_off=0.01,
                                 thr_coincidence_sum=coinc_sum,
                                 trace_ids=trace_ids, max_trigger_length=3,
                                 trigger_off_extension=0.5)
    par.dir = "/scratch/%s_trigger" % network
    par.logfile = os.path.join(par.dir, "%s_trigger.txt" % network)
    par.mailto = ["megies"]
    par.plot_time_range = (-2, 3)
    return par


def run(time, par, options):
    hours_before_now = (UTCDateTime() - time) / 3600.0
    try_seedlink = hours_before_now <= 12
    if options.verbose:
        print "Initializing seedlink client..."
    seedlink_client = SeedlinkClient("10.153.82.5")
    if options.verbose:
        print "Initializing seishub client..."
    seishub_client = SeishubClient("http://10.153.82.3:8080", timeout=60, retries=10)
    t1 = time
    t2 = t1 + (60 * 60 * 1) + 10
    st = Stream()
    num_stations = 0
    possible_coinc_sum = 0
    exceptions = []
    count_seedlink = 0
    count_seishub = 0
    for trace_id, weight in par.trace_ids.iteritems():
        net, sta, loc, cha = trace_id.split(".")
        #cha = cha[:-1] + "*"
        for comp in "ZNE":
            cha_ = cha[:-1] + comp
            try:
                # we request 60s more at start and end and cut them off later to
                # avoid a false trigger due to the tapering during instrument
                # correction
                try:
                    if options.verbose:
                        print "Fetching via seedlink:", net, sta, loc, cha_
                    tmp = seedlink_client.get_waveform(net, sta, loc, cha_,
                                                       t1 - 180, t2 + 180)
                    if not tmp:
                        raise Exception()
                    count_seedlink += 1
                except Exception, e:
                    if options.verbose:
                        print "Fetching via seishub:", net, sta, loc, cha_
                    tmp = seishub_client.waveform.getWaveform(net, sta, loc, cha_, t1 - 180,
                                                              t2 + 180)
                    count_seishub += 1
            except Exception, e:
                exceptions.append("%s: %s: %s" % (sta, e.__class__.__name__, e))
                if options.verbose:
                    print "Failed to fetch:", net, sta, loc, cha_
                continue
            if comp == cha[-1]:
                possible_coinc_sum += weight
            st.extend(tmp)
        num_stations += 1
    st.merge(-1)
    st.attach_response(par.inventory)
    st.sort()

    # look for events in seishub
    try:
        events = seishub_client.event.getList(min_datetime=t1 - 300, max_datetime=t2)
    except:
        events = None

    summary = []
    summary.append("#" * 79)
    summary.append("######## %s  ---  %s ########" % (t1, t2))
    summary.append("#" * 79)
    if try_seedlink:
        summary.append("%3i channels fetched via seedlink." % count_seedlink)
    else:
        summary.append("seedlink not used (not close to realtime).")
    summary.append("%3i channels fetched via seishub." % count_seishub)
    summary.append("#" * 79)
    summary.append(st.__str__(extended=True))
    if exceptions:
        summary.append("#" * 33 + " Exceptions  " + "#" * 33)
        summary += exceptions
    summary.append("#" * 79)

    st.traces = [tr for tr in st if tr.stats.npts > 1]

    st_raw = st.copy()
    for tr in st_raw:
        tr.stats.location = "RAW"

    trig = []
    mutt = []
    if st:
        # preprocessing, backup original data for plotting at end
        st.detrend("linear")
        # XXX special handling for KW1 artifacts:
        # XXX roughly a one minute break and then trace coming in from extreme
        # XXX amplitudes. see e.g. BW.KW1..EH* at 2015-04-09T16:50:34.000000Z
        # XXX filter from backwards, to get rid of huge very low frequency
        # XXX swing at the start of the trace
        st.taper(type="cosine", max_length=10, max_percentage=None, side="right")
        for tr in st:
            tr.data = tr.data[::-1]
            tr.filter("highpass", freq=0.5, corners=2)
            tr.data = tr.data[::-1]
        st.detrend("linear")
        # XXX end special handling
        for tr in st:
            perc = 1.0 / (tr.stats.endtime - tr.stats.starttime)
            perc = min(perc, 1)
            tr.taper(type="cosine", max_percentage=perc)
        #st.merge(method=1, fill_value=0)
        st.remove_response(water_level=10)
        st.sort()
        st_trigger = st.copy()
        # prepare plotting
        st.trim(t1, t2) #, pad=True, fill_value=0)
        st.filter("bandpass", freqmin=1, freqmax=20, corners=2)
        # triggering
        # go to mm/s
        for tr in st_trigger:
            tr.data *= 1e3
        st_trigger.filter(**par.filter)
        # do the triggering (with additional data at sides to avoid artifacts
        trig = coincidenceTrigger(stream=st_trigger, details=True,
                                  **par.coincidence)
        # restrict trigger list to time span of interest
        trig = [t for t in trig if (t1 <= t['time'] <= t2)]

        for t in trig:
            max_similarity = max(t['similarity'].values() + [0])
            time_str = str(t['time']).split(".")[0]
            sta_string = "-".join(t['stations'])
            info = "%s %ss %s %.2f %s"
            info = info % (time_str, ("%.1f" % t['duration']).rjust(4),
                           ("%i" % t['cft_peak_wmean']).rjust(3),
                           max_similarity, sta_string)
            summary.append(info)
            #tmp = st.slice(t['time'] - 1, t['time'] + t['duration'])
            tmp = (st + st_raw).slice(t['time'] + par.plot_time_range[0],
                                      t['time'] + par.plot_time_range[1])
            filename = "%s_%.1f_%i_%s-%s_%.2f_%s.png"
            filename = filename % (time_str, t['duration'],
                                   t['cft_peak_wmean'], t['coincidence_sum'],
                                   possible_coinc_sum, max_similarity,
                                   sta_string)
            filename = os.path.join(par.dir, filename)
            dpi = 72
            fig = plt.figure(figsize=(700.0 / dpi, 400.0 / dpi))
            stations = sorted(set([tr.id.rsplit(".", 1)[0] for tr in tmp]))
            ax = None
            for i_, netstaloc in enumerate(stations):
                net, sta, loc = netstaloc.split(".")
                ax = fig.add_subplot(len(stations), 1, i_ + 1, sharex=ax)
                for comp, color in zip("ENZ", "rrk"):
                    tmp_ = tmp.select(network=net, station=sta, location=loc,
                                      component=comp).copy()
                    tmp_.detrend("constant")
                    for tr in tmp_:
                        x = tr.times() + (tr.stats.starttime - t['time'])
                        ax.plot(x, tr.data, color=color, linewidth=1.2)
                ax.text(0.02, 0.95, netstaloc, va="top", ha="left",
                        transform=ax.transAxes)
            fig.axes[0].set_xlim(*par.plot_time_range)
            for ax in fig.axes:
                ylims = ax.get_ylim()
                ax.set_yticks(ylims)
                ax.set_yticklabels(["%.1e" % (val * 1e3) for val in ylims])
                ax.set_ylabel("<- mm/s ->")
            for ax in fig.axes[::2]:
                ax.yaxis.set_ticks_position("left")
                ax.yaxis.set_label_position("left")
            for ax in fig.axes[1::2]:
                ax.yaxis.set_ticks_position("right")
                ax.yaxis.set_label_position("right")
            for ax in fig.axes[:-1]:
                ax.set_xticks([])
            try:
                fig.tight_layout()
            except:
                pass
            fig.subplots_adjust(hspace=0)
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)
            # attach image if event has high similarity or high trigger value
            # or high coincidence sum
            if max_similarity > 0.5 or \
                    t['cft_peak_wmean'] > 10 or \
                    t['coincidence_sum'] >= 4:
                mutt += ("-a", filename)
        del tmp
        del st_trigger
        del tr
    del st

    summary.append("#" * 79)
    if events is None:
        summary.append("Could not fetch existing events from SeisHub.")
    else:
        summary.append("%d existing events in SeisHub" % len(events))
        format_str = ("{datetime!s} {resource_name} {author!s:<8.8} "
                      "{latitude!s:<8.8} {longitude!s:<8.8} "
                      "{magnitude!s:<4.4}")
        if events:
            events = sorted(events, key=lambda ev: ev['datetime'])
        for event in events:
            try:
                summary.append(format_str.format(**event))
            except:
                summary.append("  Exception occured formatting event info..")
    summary.append("#" * 79)
    summary = "\n".join(summary)
    # avoid writing long list of streams when using many event templates
    par_tmp = par.copy()
    for k, v in par_tmp.coincidence.event_templates.iteritems():
        par_tmp.coincidence.event_templates[k] = len(v)
    summary += "\n" + "\n".join(("%s=%s" % (k, v) for k, v in par_tmp.items()))
    #print summary
    with open(par.logfile, "at") as fh:
        fh.write(summary + "\n")
    # send emails
    if par.mailto:
        if len(trig) == 0:
            alert_lvl = 0
        else:
            alert_lvl = 1
        max_simil = max([max(t['similarity'].values() + [0]) for t in trig] + \
                        [0])
        if max_simil > 0.5:
            alert_lvl = 3
        if max_simil == 0.0 or len(trig) == 0:
            simil_lvl = "X"
        else:
            simil_lvl = "%i" % (10 * max_simil)
        for t in trig:
            if t['cft_peak_wmean'] > 7:
                alert_lvl = max(alert_lvl, 2)
            if t['cft_peak_wmean'] > 10:
                alert_lvl = max(alert_lvl, 3)
            if t['coincidence_sum'] >= 4:
                alert_lvl = max(alert_lvl, 3)
        if possible_coinc_sum < par.coincidence.thr_coincidence_sum:
            alert_lvl = "X"

        mutt_base = ["mutt", "-s",
                     "%s Alert %s Lv%s Sim%s" % (par.network.upper(),
                                                 str(t1).split(".")[0],
                                                 alert_lvl, simil_lvl)]
        mutt = mutt_base + mutt + ['--'] + par.mailto
        sub = subprocess.Popen(mutt, stdin=subprocess.PIPE)
        sub.communicate(summary)
        # send again without attachments if mutt had a problem (too many
        # images)
        if sub.returncode != 0:
            mutt = mutt_base + ['--'] + par.mailto
            sub = subprocess.Popen(mutt, stdin=subprocess.PIPE)
            sub.communicate(summary)

    plt.close('all')
    del summary
    del mutt


def main():
    usage = __doc__.strip()
    parser = optparse.OptionParser(usage)
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                      default=False)
    (options, args) = parser.parse_args()
    # if no time is specified, use now (rounded to hour) - 3 hours
    if len(args) == 1:
        t = UTCDateTime()
        t = UTCDateTime(t.year, t.month, t.day, t.hour) - 3 * 3600
        times = [t]
    elif len(args) == 2:
        t = UTCDateTime(args[1])
        times = [t]
    elif len(args) == 3:
        t1 = int(UTCDateTime(args[1]).timestamp)
        t2 = int(UTCDateTime(args[2]).timestamp)
        times = [UTCDateTime(t) for t in np.arange(t1, t2, 3600)]
    else:
        parser.print_usage()
        return
    network = args[0].lower()
    if network not in ("uh", "kw1", "be"):
        parser.print_usage()
        return

    par = getParameters(network)
    templates = returnTemplates(par.filter.freqmin, par.filter.freqmax,
                                par.filter.corners, par.filter.zerophase)
    par.coincidence.event_templates = templates
    par.coincidence.similarity_threshold = 0.5

    for t in times:
        if options.verbose:
            print "Starting network trigger for:", t
        run(t, par, options)
        print t


if __name__ == '__main__':
    main()
