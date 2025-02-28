import pulsestreamer as ps
import numpy as np


CH_MAP_DUMMY = {
    'aom': 0,
    'ch0': 0, 'ch1': 1, 'ch2': 2, 'ch3': 3
}

class PulseStreamer:

    def __init__(self, ip, channel_map=CH_MAP_DUMMY):
        self.channel_map = channel_map
        self.pulse_streamer = ps.PulseStreamer(ip)
        self.pulse_streamer.reset()
        self.seq = self.pulse_streamer.createSequence()
        self.pulse_streamer.selectClock(ps.ClockSource.INTERNAL)

    def Continuous(self, channels):
        # Turn on the specified channels indefinitely
        ch_list = [self.channel_map[ch] for ch in channels]
        self.pulse_streamer.constant((ch_list, 0, 0))
    
    def Sequence(self, sequence, start=True):
        # Convert sequence to pulsestreamer format
        pattens_by_channels = dict()
        for ch_name in self.channel_map.keys(): pattens_by_channels[ch_name] = []

        min_timestep = 1.
        for ch_high, duration in sequence:
            if duration < min_timestep: continue
            for ch_name, ch in self.channel_map.items():
                output = 1 if ch_name in ch_high else 0
                pattens_by_channels[ch_name].append((duration, output))

        del self.seq
        self.seq = self.pulse_streamer.createSequence()

        for ch_name, patt in pattens_by_channels.items():
            self.seq.setDigital(self.channel_map[ch_name], patt)
        if start: self.Run()

    def Run(self, n_runs=None):
        if self.pulse_streamer.isStreaming():
            self.Night()
        
        if not n_runs: n_runs = self.pulse_streamer.REPEAT_INFINITELY
        self.pulse_streamer.stream(self.seq, n_runs=n_runs)
    
    def Night(self):
        # Turn off all channels
        self.pulse_streamer.constant()

    def Light(self):
        # Turn on aom channel only
        self.pulse_streamer.constant(([self.channel_map['aom']], 0, 0))

    def checkUnderflow(self):
        # PulseStream do not underflow anymore
        return 0
