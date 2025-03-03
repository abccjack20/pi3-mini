
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
# from .time_tagger_dummy import TimeTaggerDummy
# time_tagger = TimeTaggerDummy()

from .time_tagger_swabian import time_tagger_control
time_tagger = time_tagger_control("1740000JEC", 1, 3, 5)

from tools.utility import singleton

@singleton
def Scanner():
	# from .nidaq_dummy import Scanner
	from .nidaq import Scanner
	
	return Scanner( CounterIn='/Dev1/Ctr3',
					CounterOut='/Dev1/Ctr2',
					TickSource='/Dev1/PFI0',
					AOChannels='/Dev1/ao0:2',
					x_range=(0.0,100.0),
					y_range=(0.0,100.0),
					z_range=(0.0,100.0),
					v_range=(0.0,10.00))

# Counter Initialization Used In ODMR
@singleton
def Counter():
	from .nidaq_dummy import PulseTrainCounter
	# from .nidaq import PulseTrainCounter

	return PulseTrainCounter( CounterIn='/Dev1/Ctr3',
							  CounterOut='/Dev1/Ctr2',
							  TickSource='/Dev1/PFI0' )

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

from .pulse_generator_dummy import PulseGeneratorDummy

@singleton
def PulseGenerator():
    return PulseGeneratorDummy(
		'serial123',
        channel_map={
			'green':8,'aom':8, 
            'mw_x': 2, 'mw': 2, 'mw_A': 2, #2/2
            'laser': 7,
            'sequence':4, 
            'awg_dis': 5,
            'rf':5, 'rf1':5, 
            'mw_b':6, 'mw_y': 6, 'SmiqRf': 2,
            'awg':1,'awgA':3, 'rf_y': 3
		}
    )
