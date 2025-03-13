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
            self.Light()
        
        if not n_runs: n_runs = self.pulse_streamer.REPEAT_INFINITELY
        self.pulse_streamer.stream(self.seq, n_runs)
    
    def Night(self):
        # Turn off all channels
        self.pulse_streamer.constant()

    def Light(self):
        # Turn on aom channel only
        self.pulse_streamer.constant(([self.channel_map['aom']], 0, 0))

    def checkUnderflow(self):
        # PulseStream do not underflow anymore
        return 0


class PulseStreamer_clock:

    def __init__(self,
        pstreamer,
        samps_per_chan=None, period=0.01, duty_cycle=0.9,
        laser_init=0., T_pi=100., T_readout=2.e3, T_init=30.e3, wait=20.e3
    ):
        self.pstreamer = pstreamer
        self.samps_per_chan = samps_per_chan
        self.period = period
        self.duty_cycle = duty_cycle
        self.laser_init = laser_init
        self.T_pi = T_pi
        self.T_readout = T_readout
        self.T_init = T_init
        self.wait = wait
        self.mode = 'cw'
        self.N_per_samp = 1
        self.sec = 1.e9

    @property
    def sample_rate(self):
        return 1/self.period
    
    @sample_rate.setter
    def sample_rate(self, rate):
        self.period = 1/rate

    def prepare_task(self):
        if self.mode.lower() == 'cw':
            self.prepare_cw()
        if self.mode.lower() == 'pulsed':
            self.prepare_pulsed()

    def update_task(self):
        self.prepare_task()

    def start(self):
        self.pstreamer.Sequence(self.sequence, start=False)
        self.pstreamer.Run(n_runs=1)

    def stop(self):
        self.pstreamer.Light()

    def prepare_cw(self):
        T = self.period*self.sec/2.
        T_readout = T*self.duty_cycle
        T_init = T - T_readout
        
        N_samps = self.samps_per_chan
        self.N_per_samp = 1

        if self.laser_init < 1e-9:
            self.sequence = []
        else:
            self.sequence = [
                (['aom'], self.laser_init),
            ]
        self.sequence += [
            (['aom', 'mw'], T_init),
            (['aom', 'mw', 'detect'], T_readout),
            (['aom',], T_init),
            (['aom', 'detect', 'next'], T_readout),
        ]*N_samps

    def prepare_pulsed(self):
        T_readout = self.T_readout*self.sec
        T_init = self.T_init*self.sec
        T_pi = self.T_pi*self.sec
        wait = self.wait*self.sec
        T = T_pi + wait + T_readout + T_init

        N_per_samp = self.period*self.sec/2./T
        self.N_per_samp = N_per_samp
        N_samps = self.samps_per_chan

        if self.laser_init < 1e-9:
            self.sequence = []
        else:
            self.sequence = [
                (['aom'], self.laser_init),
            ]
        sig_unit = [
            (['mw'], T_pi),
            ([], wait),
            (['aom', 'detect'], T_readout),
            (['aom'], T_init),
        ]
        ref_unit = [
            ([], T_pi + wait),
            (['aom', 'detect'], T_readout),
            (['aom'], T_init),
        ]
        ref_next = [
            ([], T_pi + wait),
            (['aom', 'detect'], T_readout),
            (['aom', 'next'], T_init),
        ]
        point = (sig_unit + ref_unit)*(N_per_samp - 1) + sig_unit + ref_next
        
        self.sequence += point*N_samps

