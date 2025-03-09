import numpy as np



# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("TimeTagger", "C:\\Program Files (x86)\\Swabian Instruments\\Time Tagger\\driver\\Python3.6\\x64\\TimeTagger.py")
# tt = importlib.util.module_from_spec(spec)
# sys.modules["module.name"] = tt
# spec.loader.exec_module(tt)

# from TimeTagger import CHANNEL_UNUSED, createTimeTagger, Dump, Correlation, Histogram, Counter, CountBetweenMarkers, FileWriter, Countrate, Combiner, TimeDifferences
import TimeTagger as tt

# def combine_cbm(*cbm_list):
    
class task_combiner:
    def __init__(self, *task_list):
        self.task_list = task_list

    def start(self):
        for t in self.task_list:
            t.start()
    
    def stop(self):
        for t in self.task_list:
            t.stop()
    
    def clear(self):
        for t in self.task_list:
            t.clear()
        
    def isRunning(self):
        state = True
        for t in self.task_list:
            state &= t.isRunning()
        return state

class cbm_combiner(task_combiner):

    main_id = 0

    def getData(self):
        data = 0
        for t in self.task_list:
            data += t.getData()
        return data

    def getBinWidths(self):
        return self.task_list[self.main_id].getBinWidths()

    def ready(self):
        state = True
        for t in self.task_list:
            state &= t.ready()
        return state
    
class td_combiner(task_combiner):

    main_id = 0

    def getData(self):
        data = 0
        for t in self.task_list:
            data += t.getData()
        return data
    
    def ready(self):
        state = False
        for t in self.task_list:
            state &= t.ready()
        return state

def multiple_cbm(tagger, ch_list_click, ch_start, ch_end, n_bins):
    
    task_list = []
    for ch_click in ch_list_click:
        task = tt.CountBetweenMarkers(
            tagger,
            ch_click,
            ch_start,
            end_channel=ch_end,
            n_values=n_bins,
        )
        task_list.append(task)
    return cbm_combiner(*task_list)

def multiple_td(tagger, ch_list_click, ch_start, ch_next, ch_sync, binwidth, n_bins, n_lasers):

    task_list = []
    for ch_click in ch_list_click:
        task = tt.TimeDifferences(
            tagger,
            ch_click,
            start_channel=ch_start,
            next_channel=ch_next,
            sync_channel=ch_sync,
            binwidth=binwidth,
            n_bins=n_bins,
            n_histograms=n_lasers,
        )
    task_list.append(task)
    return td_combiner(*task_list)

class time_tagger_control:

    def __init__(self, serial, ch_list_ticks, ch_detect, ch_sync, ch_marker=None):
        self._serial = serial
        self._tagger = tt.createTimeTagger(serial)
        self._channels = dict(
            ticks = ch_list_ticks,
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
        ch_list_ticks   = self._channels['ticks']
        ch_start        = self._channels['marker']
        ch_end          = -ch_start     # Use falling edge of the same channel
        return multiple_cbm(
            self._tagger,
            ch_list_ticks,
            ch_start,
            ch_end,
            n_bins,
        )
    
    def Pulsed(self, n_bins, binwidth, n_lasers, *args):
        ch_list_ticks   = self._channels['ticks']
        ch_start        = self._channels['detect']
        ch_next         = self._channels['detect']
        ch_sync         = self._channels['sync']
        return multiple_td(
            self._tagger,
            ch_list_ticks,
            ch_start,
            ch_next,
            ch_sync,
            binwidth,
            n_bins,
            n_lasers,
        )