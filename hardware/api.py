
"""
Hardware API is defined here.

Example of usage:

from hardware.api import PulseGenerator
PG = PulseGenerator

Default hardware api hooks are dummy classes.

Provide a file 'custom_api.py' to define actual hardware API hooks.
This can be imported names, modules, factory functions and factory functions
that emulate singleton behavior.

See 'custom_api_example.py' for examples.
"""

import numpy as np
import logging
import time
from tools.utility import singleton


tt_serial       = "1740000JEC"
ch_list_ticks   = [1, 2]
ch_detect       = 5
ch_sync         = 8

ch_marker_scanner = 6
ch_marker_counter = ch_detect

ps_ip = '169.254.8.2'
ps_channels = {
    'aom':0,
    'detect':1,
    'mw':4,
    'next':6,
    'sync':7,
}

scanner_params = dict(
	device_name = 'dev1',
    counter_name = 'ctr1',
    ao_channels = ['ao0', 'ao1', 'ao2', 'ao3'],
    voltage_range = [
        [0., 10.],       # ao0
        [0., 10.],       # ao1
        [0., 10.],       # ao2
        [-10., 10.],       # ao3
    ],
    sec_per_point = .01,
    duty_cycle = 0.9,
    x_range = (-100.0,100.0),
    y_range = (-100.0,100.0),
    z_range = (0.,100.0),
    aom_range = (-10.,10.),
    home_pos = [0., 0., 0., -10.],
    invert_x = False,
    invert_y = False,
    invert_z = False,
    swap_xy = False,
)

# counter_params = dict(
#     device_name = 'dev1',
#     ctr_list = ['ctr0','ctr1'],
#     sec_per_point = .01,
#     duty_cycle = 0.9,
# )

counter_params = dict(
    sec_per_point = .01,
    duty_cycle = 0.9,
    laser_init = 0.5
)

# Dummy TimeTagger
# from .time_tagger_dummy import TimeTaggerDummy
# time_tagger = TimeTaggerDummy()

from .time_tagger_swabian import time_tagger_control
time_tagger = time_tagger_control(
	tt_serial,	# serial
	ch_list_ticks,         # ch_list_ticks
	ch_detect,				# ch_detect
	ch_sync,				# ch_sync
)

@singleton
def PulseGenerator():
	#from .pulse_generator_dummy import PulseGeneratorDummy
	from .pulse_streamer import PulseStreamer
	return PulseStreamer(ps_ip, channel_map=ps_channels)

@singleton
def Scanner():
    from .finite_scanner import Stage_control
    return Stage_control(time_tagger, ch_marker_scanner, **scanner_params)
    

# Counter Initialization Used In ODMR
@singleton
def Counter():
    # from .finite_scanner import NIDAQ_Pulse_Train_Counter
    # return NIDAQ_Pulse_Train_Counter(time_tagger, **counter_params)
    from .finite_scanner import PS_Pulse_Train_Counter
    return PS_Pulse_Train_Counter(time_tagger, PulseGenerator(), ch_marker_counter, **counter_params)


# Microvave Source Initialization
@singleton
def Microwave():
    # from .microwave_dummy import MicrowaveDummy
    # return MicrowaveDummy(visa_address='GPIB0::00')
    from .microwave_smiq import SMIQ
    return SMIQ(visa_address='GPIB0::28::INSTR')

MicrowaveA = Microwave

@singleton
def RFSource():
    from .microwave_dummy import MicrowaveDummy
    return MicrowaveDummy(visa_address='GPIB0::01')


# @singleton
# def Scanner():
# 	from .nidaq_dummy import Scanner
# 	from .nidaq_dll import Scanner
# 	return Scanner( CounterIn='/Dev2/Ctr1',
# 					CounterOut='/Dev2/Ctr0',
# 					TickSource='/Dev2/PFI3',
# 					AOChannels='/Dev2/ao0:2',
# 					x_range=(0.0,344.0),
# 					y_range=(0.0,344.0),
# 					z_range=(0,100.0),
# 					v_range=(-1.00,10.00))