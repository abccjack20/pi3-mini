
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

# Dummy TimeTagger
from .time_tagger_swabian import time_tagger_control
time_tagger = time_tagger_control("1740000JEJ", 1, 5, 8)

from tools.utility import singleton

@singleton
def Scanner():
	# from .nidaq_dummy import Scanner
	from .nidaq import Scanner
	
	return Scanner( CounterIn='/Dev2/Ctr1',
					CounterOut='/Dev2/Ctr0',
					TickSource='/Dev2/PFI3',
					AOChannels='/Dev2/ao0:2',
					x_range=(0.0,344.0),
					y_range=(0.0,344.0),
					z_range=(0,100.0),
					v_range=(-1.00,10.00))

# Counter Initialization Used In ODMR
@singleton
def Counter():
	from .nidaq_dummy import PulseTrainCounter
	return PulseTrainCounter( CounterIn='/Dev1/Ctr3',
							  CounterOut='/Dev1/Ctr2',
							  TickSource='/Dev1/PFI3' )

# Microvave Source Initialization
@singleton
def Microwave():
    from .microwave_dummy import MicrowaveDummy
    return MicrowaveDummy(visa_address='GPIB0::00')

MicrowaveA = Microwave

@singleton
def RFSource():
    from .microwave_dummy import MicrowaveDummy
    return MicrowaveDummy(visa_address='GPIB0::01')


@singleton
def PulseGenerator():
	#from .pulse_generator_dummy import PulseGeneratorDummy
	from .pulse_streamer import PulseStreamer
	return PulseStreamer(
		'169.254.8.2',
		channel_map = {
			'aom':0,
			'detect':1,
			'mw':2,
			'sync':7,
		}
	)
