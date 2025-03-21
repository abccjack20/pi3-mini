
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

# Dummy TimeTagger
# from .time_tagger_dummy import TimeTaggerDummy
# time_tagger = TimeTaggerDummy()

from .time_tagger_swabian import time_tagger_control
time_tagger = time_tagger_control(
	"1740000JEC",	# serial
	[1, 2],         # ch_list_ticks
	5,				# ch_detect
	8,				# ch_sync
	ch_marker=6,
)

@singleton
def PulseGenerator():
	#from .pulse_generator_dummy import PulseGeneratorDummy
	from .pulse_streamer import PulseStreamer
	return PulseStreamer(
		'169.254.8.2',
		channel_map = {
			'aom':0,
			'detect':1,
			'mw':4,
			'sync':7,
		}
	)

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

counter_params = dict(
    device_name = 'dev1',
    ctr_list = ['ctr0','ctr1'],
    sec_per_point = .01,
    duty_cycle = 0.9,
)

@singleton
def Scanner():
    from .nidaq_finite_scanner import Stage_control
    return Stage_control(time_tagger, **scanner_params)
    

# Counter Initialization Used In ODMR
@singleton
def Counter():
    from .nidaq_finite_scanner import Pulse_Train_Counter
    return Pulse_Train_Counter(time_tagger, **counter_params)
	# from .nidaq_dummy import PulseTrainCounter
	# from .nidaq_dll import PulseTrainCounter
	# return PulseTrainCounter( CounterIn='/Dev1/Ctr3',
	# 						  CounterOut='/Dev1/Ctr2',
	# 						  TickSource='/Dev1/PFI0' )

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