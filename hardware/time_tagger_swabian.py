import numpy as np



# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("TimeTagger", "C:\\Program Files (x86)\\Swabian Instruments\\Time Tagger\\driver\\Python3.6\\x64\\TimeTagger.py")
# tt = importlib.util.module_from_spec(spec)
# sys.modules["module.name"] = tt
# spec.loader.exec_module(tt)

# from TimeTagger import CHANNEL_UNUSED, createTimeTagger, Dump, Correlation, Histogram, Counter, CountBetweenMarkers, FileWriter, Countrate, Combiner, TimeDifferences
import TimeTagger as tt

class time_tagger_control:

    def __init__(self, serial, ch_ticks, ch_detect, ch_sync, ch_marker=None):
        self._serial = serial
        self._tagger = tt.createTimeTagger(serial)
        self._channels = dict(
            ticks = ch_ticks,
            detect = ch_detect,
            sync = ch_sync
        )
        if ch_marker: self._channels['marker'] = ch_marker

    def Counter(self, ch_list, binwidth, TraceLength):
        return tt.Counter(
            self._tagger,
            ch_list,
            binwidth,
            TraceLength
        )
    
    def Count_Between_Markers(self, n_bins):
        if not 'marker' in self._channels.keys():
            print("Marker channel is not specified!")
            return
        ch_click    = self._channels['ticks']
        ch_start    = self._channels['marker']
        ch_end      = -ch_start     # Use falling edge of the same channel
        return tt.CountBetweenMarkers(
            self._tagger,
            ch_click,
            ch_start,
            end_channel=ch_end,
            n_values=n_bins,
        )
    
    def Pulsed(self, n_bins, binwidth, n_lasers, *args):
        ch_click = self._channels['ticks']
        ch_start = self._channels['detect']
        ch_next = self._channels['detect']
        ch_sync = self._channels['sync']
        return tt.TimeDifferences(
            self._tagger,
            ch_click,
            start_channel=ch_start,
            next_channel=ch_next,
            sync_channel=ch_sync,
            binwidth=binwidth,
            n_bins=n_bins,
            n_histograms=n_lasers,
        )