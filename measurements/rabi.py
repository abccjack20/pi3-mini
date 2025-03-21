import numpy as np

from traits.api import Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group, Label

import logging
import time

import hardware.api as ha

from .pulsed import Pulsed

class Rabi( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw','mw_b','mw_c',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2867493000.0, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-27,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=4000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveE().setOutput(self.power, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        """
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(None, self.frequency)
        elif self.switch=='mw_c':
            ha.MicrowaveD().setOutput(None, self.frequency)
        """


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser_time = self.laser
        wait = self.wait
        sync_time   = 100.
        sequence = []
        sequence += [ (['sync'], 100  )]
        sequence += [ ([  ], 1000 )  ]
        sequence = [(['sync'], sync_time)]
        for t in tau:
            sequence += [
                ([],wait),
                ([MW],t), ([],tau[-1] - t),
                (['aom','detect'], laser_time)
            ]

        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement',
                  )
    
class Rabi1( Pulsed ):
    
    """Defines a Rabi measurement."""

    switch      = Enum( 'mw_x','mw_b','mw_y',   desc='switch to use for different microwave source',     label='switch' )
    frequency   = Range(low=1,      high=20e9,  value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    power       = Range(low=-100.,  high=25.,   value=-24,      desc='microwave power',     label='power [dBm]',    mode='text', auto_set=False, enter_set=True)

    tau_begin   = Range(low=0., high=1e8,       value=1.5,     desc='tau begin [ns]',  label='tau begin [ns]',   mode='text', auto_set=False, enter_set=True)
    tau_end     = Range(low=1., high=1e8,       value=5000.,     desc='tau end [ns]',    label='tau end [ns]',     mode='text', auto_set=False, enter_set=True)
    tau_delta   = Range(low=1., high=1e6,       value=30.,      desc='delta tau [ns]',  label='delta tau [ns]',   mode='text', auto_set=False, enter_set=True)
    laser       = Range(low=1., high=100000.,   value=3000.,    desc='laser [ns]',      label='laser [ns]',       mode='text', auto_set=False, enter_set=True)
    wait        = Range(low=0., high=100000.,   value=1000.,    desc='wait [ns]',       label='wait [ns]',        mode='text', auto_set=False, enter_set=True)

    tau = Array( value=np.array((0.,1.)) )

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)
        
    def start_up(self):
        ha.PulseGenerator().Night()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(self.power, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(self.power, self.frequency)
        elif self.switch=='mw_f':
            ha.MicrowaveF().setOutput(self.power, self.frequency)
            
    def shut_down(self):
        ha.PulseGenerator().Light()
        if self.switch=='mw_a':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_x' or self.switch=='mw_y':
            ha.MicrowaveA().setOutput(None, self.frequency)
        elif self.switch=='mw_b':
            ha.MicrowaveD().setOutput(None, self.frequency)
        elif self.switch=='mw_f':
            ha.MicrowaveF().setOutput(None, self.frequency)


    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        sequence = []
        for t in tau:
            #sequence += [  ([MW,'mw'],t),  (['laser','aom'],laser),  ([],wait)  ]
            sequence += [  ([MW],t),  (['laser','aom'],laser),  ([],wait)  ]
        sequence += [ (['sequence'], 100  )  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['frequency','power','switch','tau_begin','tau_end','tau_delta','laser','wait','tau']
    get_set_order = ['tau','time_bins','count_data']

    traits_view = View(VGroup(HGroup(Item('submit_button',   show_label=False),
                                     Item('remove_button',   show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly',format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('switch', style='custom'),
                                                   Item('frequency',     width=-80, enabled_when='state == "idle"'),
                                                   Item('power',         width=-80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('tau_begin',     width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_end',       width=-80, enabled_when='state == "idle"'),
                                                   Item('tau_delta',     width=-80, enabled_when='state == "idle"'),
                                                   Item('stop_time'),),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser',         width=-80, enabled_when='state == "idle"'),
                                                   Item('wait',          width=-80, enabled_when='state == "idle"'),
                                                   Item('record_length', width=-80, enabled_when='state == "idle"'),
                                                   Item('bin_width',     width=-80, enabled_when='state == "idle"'),),
                                     label='settings'),
                              ),
                        ),
                       title='Rabi Measurement',
                  )