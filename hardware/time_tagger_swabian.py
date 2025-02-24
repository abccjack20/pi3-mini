import numpy as np

from TimeTagger import CHANNEL_UNUSED, createTimeTagger, Dump, Correlation, Histogram, Counter, CountBetweenMarkers, FileWriter, Countrate, Combiner, TimeDifferences


class time_tagger_control:

    def __init__(self, serial, ch_ticks, ch_detect, ch_sync):
        self._serial = serial
        self._tagger = createTimeTagger(serial)
        self._channels = dict(
            ticks = ch_ticks,
            detect = ch_detect,
            sync = ch_sync
        )

    def Counter(self, ch_list, binwidth, TraceLength):
        return Counter(
            self._tagger,
            ch_list,
            binwidth,
            TraceLength
        )
    
    def Pulsed(self, n_bins, binwidth, n_lasers, *args):
        ch_click = self._channels['ticks']
        ch_start = self._channels['detect']
        ch_next = self._channels['detect']
        ch_sync = self._channels['sync']
        return TimeDifferences(
            self._tagger,
            ch_click,
            start_channel=ch_start,
            next_channel=ch_next,
            sync_channel=ch_sync,
            binwidth=binwidth,
            n_bins=n_bins,
            n_histograms=n_lasers,
        )