import numpy as np

from traits.api import Range, Int, Float, Bool, Array, Instance, Enum, on_trait_change, Button
from traitsui.api import View, Item, Tabbed, HGroup, VGroup, VSplit, EnumEditor, TextEditor

import logging
import time

from hardware.api import PulseGenerator, Microwave, RFSource
from hardware.api import time_tagger as TimeTagger


from tools.emod import ManagedJob

from tools.utility import GetSetItemsMixin

"""
Several options to decide when to start  and when to restart a job, i.e. when to clear data, etc.

1. set a 'new' flag on every submit button

pro: simple, need not to think about anything in subclass

con: continue of measurement only possible by hack (manual submit to JobManager without submit button)
     submit button does not do what it says
     
2. check at start time whether this is a new measurement.

pro: 

con: complicated checking needed
     checking has to be reimplemented on sub classes
     no explicit way to restart the same measurement

3. provide user settable clear / keep flag

pro: explicit

con: user can forget

4. provide two different submit buttons: submit, resubmit

pro: explicit

con: two buttons that user may not understand
     user may use wrong button
     wrong button can result in errors

"""

# utility functions
def find_detect_pulses(sequence):
    n = 0
    prev = []
    for channels, t in sequence:
        if 'detect' in channels and not 'detect' in prev:
            n += 1
        prev = channels
        if ('sync' in channels) and (n>0):
            break
    return n

def sequence_length(sequence):
    t = 0
    for c, ti in sequence:
        t += ti
    return t

def sequence_union(s1, s2):
    """
    Return the union of two pulse sequences s1 and s2.
    """
    # make sure that s1 is the longer sequence and s2 is merged into it
    if sequence_length(s1) < sequence_length(s2):
        sp = s2
        s2 = s1
        s1 = sp
    s = []
    c1, dt1 = s1.pop(0)
    c2, dt2 = s2.pop(0)
    while True:
        if dt1 < dt2:
            s.append((set(c1) | set(c2), dt1))
            dt2 -= dt1
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
        elif dt2 < dt1:
            s.append((set(c1) | set(c2), dt2))
            dt1 -= dt2
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf
        else:
            s.append((set(c1) | set(c2), dt1))
            try:
                c1, dt1 = s1.pop(0)
            except:
                break
            try:
                c2, dt2 = s2.pop(0)
            except:
                c2 = []
                dt2 = np.inf            
    return s

def sequence_remove_zeros(sequence):
    return filter(lambda x: x[1] != 0.0, sequence)

class Pulsed(ManagedJob, GetSetItemsMixin):
    
    """Defines a pulsed measurement."""
    
    keep_data = Bool(False) # helper variable to decide whether to keep existing data

    resubmit_button = Button(label='resubmit', desc='Submits the measurement to the job manager. Tries to keep previously acquired data. Behaves like a normal submit if sequence or time bins have changed since previous run.')

    sequence = Instance(list, factory=list)
    
    record_length = Range(low=100, high=1000000., value=3000, desc='length of acquisition record [ns]', label='record length [ns]', mode='text', auto_set=False, enter_set=True)
    bin_width = Range(low=0.1, high=10000., value=1.0, desc='bin width [ns]', label='bin width [ns]', mode='text', auto_set=False, enter_set=True)
    
    n_laser = Int(2)
    n_bins = Int(2)
    time_bins = Array(value=np.array((0, 1)))
    
    count_data = Array(value=np.zeros((2, 2)))
    
    run_time = Float(value=0.0, label='run time [s]', format_str='%.f')
    stop_time = Range(low=1., value=np.inf, desc='Time after which the experiment stops by itself [s]', label='Stop time [s]', mode='text', auto_set=False, enter_set=True)
    
    def submit(self):
        """Submit the job to the JobManager."""
        self.keep_data = False
        ManagedJob.submit(self)

    def resubmit(self):
        """Submit the job to the JobManager."""
        self.keep_data = True
        ManagedJob.submit(self)

    def _resubmit_button_fired(self):
        """React to start button. Submit the Job."""
        self.resubmit() 

    def generate_sequence(self):
        return []

    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()
        n_laser = find_detect_pulses(sequence)

        if self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins): # if the sequence and time_bins are the same as previous, keep existing data
            self.old_count_data = self.count_data.copy()
        else:
            self.old_count_data = np.zeros((n_laser, n_bins))
            self.run_time = 0.0
        
        self.sequence = sequence 
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.n_laser = n_laser
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.

    def start_up(self):
        """Put here additional stuff to be executed at startup."""
        pass

    def shut_down(self):
        """Put here additional stuff to be executed at shut_down."""
        pass

    def _run(self):
        """Acquire data."""

        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()

            if self.run_time >= self.stop_time:
                logging.getLogger().debug('Runtime larger than stop_time. Returning')
                self.state = 'done'
                return

            self.start_up()
            PulseGenerator().Night()
            tagger_0 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 0, 2, 3)
            # tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, 1, 2, 3)

            #tagger_0 = TimeTagger.Pulsed(int(self.n_bins), int(np.round(self.bin_width * 1000)), int(self.n_laser), Int(0), Int(2), Int(3))
            #tagger_1 = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), self.n_laser, Int(1), Int(2), Int(3))
            PulseGenerator().Sequence(self.sequence)
            if PulseGenerator().checkUnderflow():
                logging.getLogger().info('Underflow in pulse generator.')
                PulseGenerator().Night()
                PulseGenerator().Sequence(self.sequence)
                
            while self.run_time < self.stop_time:
                start_time = time.time()
                self.thread.stop_request.wait(1.0)
                if self.thread.stop_request.isSet():
                    logging.getLogger().debug('Caught stop signal. Exiting.')
                    break
                if PulseGenerator().checkUnderflow():
                    logging.getLogger().info('Underflow in pulse generator.')
                    PulseGenerator().Night()
                    PulseGenerator().Sequence(self.sequence)

                # print(tagger_0.getIndex())
                self.count_data = self.old_count_data + tagger_0.getData() # + tagger_1.getData()
                self.run_time += time.time() - start_time

            if self.run_time < self.stop_time:
                self.state = 'idle'
            else:
                self.state = 'done'
            del tagger_0
            self.shut_down()
            PulseGenerator().Light()

        except: # if anything fails, log the exception and set the state
            logging.getLogger().exception('Something went wrong in pulsed loop.')
            self.state = 'error'

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     #Item('run_time', style='readonly', format_str='%.f'),
                                     #Item('stop_time'),
                                     ),
                             HGroup(Item('run_time', style='readonly', format_str='%.f'),
                                    Item('stop_time'),
                                    ),
                              HGroup(Item('bin_width', width= -80, enabled_when='state != "run"'),
                                     Item('record_length', width= -80, enabled_when='state != "run"'),
                                     ),
                              ),
                       title='Pulsed Measurement',
                       )

    get_set_items = ['__doc__', 'record_length', 'bin_width', 'n_bins', 'time_bins', 'n_laser', 'sequence', 'count_data', 'run_time']


class PulsedTau(Pulsed):

    """Defines a Pulsed measurement with tau mesh."""

    tau_begin = Range(low=0., high=1e8, value=0., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=300., desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e8, value=3., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)

    get_set_items = Pulsed.get_set_items + ['tau_begin', 'tau_end', 'tau_delta', 'tau']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     #Item('run_time', style='readonly', format_str='%.f'),
                                     #Item('stop_time'),
                                     ),
                             HGroup(Item('run_time', style='readonly', format_str='%.f'),
                                    Item('stop_time'),
                                    ),
                              HGroup(Item('bin_width', width= -80, enabled_when='state != "run"'),
                                     Item('record_length', width= -80, enabled_when='state != "run"'),
                                     ),
                              ),
                       title='PulsedTau Measurement',
                       )

class Rabi(PulsedTau):
    
    """Defines a Rabi measurement."""

    frequency = Range(low=1, high=20e9, value=2.8705e9, desc='microwave frequency', label='frequency [Hz]', mode='text', auto_set=False, enter_set=True, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    power = Range(low= -100., high=25., value= -20, desc='microwave power', label='power [dBm]', mode='text', auto_set=False, enter_set=True)
    switch = Enum('mw', 'mw_y', desc='switch to use for microwave pulses', label='switch', editor=EnumEditor(cols=3, values={'mw':'1:X', 'mw_y':'2:Y'}))

    laser = Range(low=1., high=1000000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=0., high=100000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    wait2 = Range(low=0., high=100000., value=1000., desc='wait at the end', label='wait 2 [ns]', mode='text', auto_set=False, enter_set=True)


    def start_up(self):
        PulseGenerator().Night()
        Microwave().setOutput(self.power, self.frequency)

    def shut_down(self):
        PulseGenerator().Light()
        Microwave().setOutput(None, self.frequency)

    def generate_sequence(self):
        MW = self.switch
        tau = self.tau
        laser = self.laser
        wait = self.wait
        wait2 = self.wait2
        sequence = [ (['sync', 'aom'], laser)]
        for t in tau:
            sequence += [
                ([],wait),
                ([MW],t),
                # ([],tau[-1] - t),
                ([],wait2),
                (['aom', 'detect'], laser)
            ]
        
        return sequence

    get_set_items = PulsedTau.get_set_items + ['frequency', 'power', 'switch', 'laser', 'wait', 'wait2']

    traits_view = View(
        VGroup(
            HGroup(
                Item('submit_button', show_label=False),
                Item('remove_button', show_label=False),
                Item('resubmit_button', show_label=False),
                Item('priority'),
                Item('stop_time'),
                Item('run_time', style='readonly', format_str='%.f'),
                Item('state', style='readonly'),
            ),
            Tabbed(
                VGroup(
                    HGroup(
                        Item('frequency', width= -80, enabled_when='state != "run"'),
                        Item('power', width= -80, enabled_when='state != "run"'),
                        Item('switch', style='custom', enabled_when='state != "run"'),
                    ),
                    HGroup(
                        Item('tau_begin', width= -80, enabled_when='state != "run"'),
                        Item('tau_end', width= -80, enabled_when='state != "run"'),
                        Item('tau_delta', width= -80, enabled_when='state != "run"'),
                    ),
                    label='parameter'
                ),
                VGroup(
                    HGroup(
                        Item('laser', width= -80, enabled_when='state != "run"'),
                        Item('wait', width= -80, enabled_when='state != "run"'),
                        Item('wait2', width= -80, enabled_when='state != "run"'),
                    ),
                    HGroup(
                        Item('record_length', width= -80, enabled_when='state != "run"'),
                        Item('bin_width', width= -80, enabled_when='state != "run"'),
                    ),
                label='settings'
                ),
            ),
        ),
        title='Rabi Measurement',
    )

class T1(PulsedTau):
    
    """Defines a T1 measurement."""

    laser = Range(low=1., high=100000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
        
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        sequence = [ (['aom'], laser) ]
        for t in tau:
            sequence += [  ([], t), (['detect', 'aom'], laser)  ]
        sequence += [  ([], 1000)  ]
        sequence += [  (['sequence'], 100)  ]
        return sequence

    get_set_items = PulsedTau.get_set_items + ['laser']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     #Item('run_time', style='readonly', format_str='%.f'),
                                     #Item('stop_time'),
                                     ),
                              HGroup(Item('run_time', style='readonly', format_str='%.f'),
                                    Item('stop_time'),
                                    ),
                              Tabbed(VGroup(HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                              ),
                       ),
                       title='T1 Measurement',
                       )

class SingletDecay(Pulsed):
    
    """Singlet Decay."""

    laser = Range(low=1., high=100000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)

    tau_begin = Range(low=0., high=1e8, value=0., desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e9, value=300., desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e6, value=3., desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)

    tau = Array(value=np.array((0., 1.)))

    def apply_parameters(self):
        """Overwrites apply_parameters() from pulsed. Prior to generating sequence, etc., generate the tau mesh."""
        self.tau = np.arange(self.tau_begin, self.tau_end, self.tau_delta)
        Pulsed.apply_parameters(self)

    def start_up(self):
        PulseGenerator().Night()

    def shut_down(self):
        PulseGenerator().Light()

    def generate_sequence(self):

        tau = self.tau
        laser = self.laser

        sequence = [ (['sequence'], 100) ]
        for t in tau:
            sequence += [  ([], t), (['detect', 'aom'], laser)  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['tau_begin', 'tau_end', 'tau_delta', 'laser', 'tau']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   Item('stop_time'),),
                                            label='manipulation'),
                                     VGroup(HGroup(Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='detection'
                                            ),
                                     ),
                              ),
                              title='Singlet decay',
                              )
    
class SingletDecayDouble(PulsedTau):
    
    """Singlet Decay."""

    laser = Range(low=1., high=100000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)

    def start_up(self):
        PulseGenerator().Night()

    def shut_down(self):
        PulseGenerator().Light()

    def generate_sequence(self):

        tau = self.tau
        laser = self.laser

        sequence = [ (['sync'], 100) ]
        for t in tau:
            sequence += [  ([], t), (['detect', 'aom'], laser)  ]
        for t in tau[::-1]:
            sequence += [  ([], t), (['detect', 'aom'], laser)  ]
        return sequence

    get_set_items = Pulsed.get_set_items + ['tau_begin', 'tau_end', 'tau_delta', 'laser', 'tau']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='manipulation'),
                                     VGroup(HGroup(Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='detection'
                                            ),
                                     ),
                              ),
                              title='Singlet decay double',
                              )
    
class Hahn(Rabi):
    
    """Defines a Hahn-Echo measurement."""

    t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi = Range(low=1., high=100000., value=1000., desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2 = self.t_pi2
        t_pi = self.t_pi
        sequence = []
        for t in tau:
            sequence += [  (['mw','mw_x'], t_pi2), ([], 0.5 * t), (['mw','mw_x'], t_pi), ([], 0.5 * t), (['mw','mw_x'], t_pi2), (['detect', 'aom'], laser), ([], wait)  ]
        sequence += [  (['sync'], 100)  ]
        return sequence

    get_set_items = Rabi.get_set_items + ['t_pi2', 't_pi']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi2', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'
                                            ),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'
                                            ),
                                     ),
                              ),
                       title='Hahn-Echo Measurement',
                       )
    
class Hahn3pi2(Rabi):
    
    """Defines a Hahn-Echo measurement with both pi/2 and 3pi/2 readout pulse."""

    t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi = Range(low=1., high=100000., value=1000., desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2 = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2 = self.t_pi2
        t_pi = self.t_pi
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:
            sequence += [
                ([], self.tau_end - t),
                (['mw','mw_x'], t_pi2),
                ([], 0.5 * t),
                (['mw','mw_x'], t_pi),
                ([], 0.5 * t),
                (['mw','mw_x'], t_pi2),
                ([], wait),
                (['detect', 'aom'], laser),
            ]
        for t in tau:
            sequence += [
                ([], self.tau_end - t),
                (['mw','mw_x'], t_pi2),
                ([], 0.5 * t),
                (['mw','mw_x'], t_pi),
                ([], 0.5 * t),
                (['mw','mw_x'], t_3pi2),
                ([], wait),
                (['detect', 'aom'], laser),
            ]
        sequence += [  (['sync'], 100)  ]
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi2', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi', width= -80, enabled_when='state != "run"'),
                                                   Item('t_3pi2', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'
                                            ),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'
                                            ),
                                     ),
                              ),
                       title='Hahn-Echo Measurement with both pi/2 and 3pi/2 readout pulse',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi2', 't_pi', 't_3pi2']

class CPMG3pi2(Rabi):
    
    """
    Defines a CPMG measurement with both pi/2 and 3pi/2 readout pulse,
    using a second microwave switch for 90 degree phase shifted pi pulses.
    
    Includes also bright (no pulse) and dark (pi_y pulse) reference points.
    """

    t_pi2_x = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length (x)', label='pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_y = Range(low=1., high=100000., value=1000., desc='pi pulse length (y)', label='pi y [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2_x = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length (x)', label='3pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    n_pi = Range(low=1, high=100, value=5, desc='number of pi pulses', label='n pi', mode='text', auto_set=False, enter_set=True)
    n_ref = Range(low=0, high=100, value=0, desc='number of reference pulses', label='n ref', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2_x = self.t_pi2_x
        t_pi_y = self.t_pi_y
        t_3pi2_x = self.t_3pi2_x
        n_pi = self.n_pi
        n_ref = self.n_ref
        sequence = []
        for t in tau:
            sequence += [ (['mw_x'], t_pi2_x), ([], t / float(2 * n_pi))  ]
            sequence += (n_pi - 1) * [ (['mw_y'], t_pi_y), ([], t / float(n_pi))    ] 
            sequence += [ (['mw_y'], t_pi_y), ([], t / float(2 * n_pi))  ]
            sequence += [ (['mw_x'], t_pi2_x)                              ]
            sequence += [ (['detect', 'aom'], laser), ([], wait)             ]
        for t in tau:
            sequence += [ (['mw_x'], t_pi2_x), ([], t / float(2 * n_pi))  ]
            sequence += (n_pi - 1) * [ (['mw_y'], t_pi_y), ([], t / float(n_pi))    ] 
            sequence += [ (['mw_y'], t_pi_y), ([], t / float(2 * n_pi))  ]
            sequence += [ (['mw_x'], t_3pi2_x)                             ]
            sequence += [ (['detect', 'aom'], laser), ([], wait)             ]            
        sequence += n_ref * [ (['detect', 'aom'], laser), ([], wait)       ]
        sequence += n_ref * [ (['mw_y'], t_pi_y), (['laser', 'aom'], laser), ([], wait)      ]
        sequence += [ (['sync'], 100)                        ]
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('t_pi2_x', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi_y', width= -80, enabled_when='state != "run"'),
                                                   Item('t_3pi2_x', width= -80, enabled_when='state != "run"'),
                                                   Item('n_pi', width= -80, enabled_when='state != "run"'),
                                                   Item('n_ref', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'
                                            ),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'
                                            ),
                                     ),
                              ),
                       title='CPMG Measurement with both pi/2 and 3pi/2 readout pulse',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi2_x', 't_pi_y', 't_3pi2_x', 'n_pi', 'n_ref']


class XY83pi2(Rabi):
    
    """
    Defines a XY8 measurement with both pi/2 and 3pi/2 readout pulse,
    using a second microwave switch for 90 degree phase shifted pi pulses.
    
    Includes also bright (no pulse) and dark (pi_y pulse) reference points.
    """

    t_pi2_x = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length (x)', label='pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_x = Range(low=1., high=100000., value=1000., desc='pi pulse length (x)', label='pi x [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_y = Range(low=1., high=100000., value=1000., desc='pi pulse length (y)', label='pi y [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2_x = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length (x)', label='3pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    n_pi = Range(low=1, high=100, value=4, desc='number of pi pulses', label='n pi', mode='text', auto_set=False, enter_set=True)
    n_ref = Range(low=0, high=100, value=0, desc='number of reference pulses', label='n ref', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2_x = self.t_pi2_x
        t_pi_x = self.t_pi_x
        t_pi_y = self.t_pi_y
        t_3pi2_x = self.t_3pi2_x
        n_pi = self.n_pi
        n_ref = self.n_ref
        sequence = []
        for t in tau:
            sequence += [ (['mw_x'], t_pi2_x), ([], t / float(2 * n_pi))  ]
            sequence += int(n_pi / 4) * [ (['mw_x'], t_pi_x), ([], t / float(n_pi)), (['mw_y'], t_pi_y), ([], t / float(n_pi))] 
            sequence += int(n_pi / 4 - 1) * [ (['mw_y'], t_pi_y), ([], t / float(n_pi)), (['mw_x'], t_pi_x), ([], t / float(n_pi))] 
            sequence += [ (['mw_y'], t_pi_y), ([], t / float(n_pi)), (['mw_x'], t_pi_x), ([], t / float(2 * n_pi)), (['mw_x'], t_pi2_x)]
            sequence += [ (['detect', 'aom'], laser), ([], wait)             ]
        for t in tau:
            sequence += [ (['mw_x'], t_pi2_x), ([], t / float(2 * n_pi))  ]
            sequence += int(n_pi / 4) * [ (['mw_x'], t_pi_x), ([], t / float(n_pi)), (['mw_y'], t_pi_y), ([], t / float(n_pi))] 
            sequence += int(n_pi / 4 - 1) * [ (['mw_y'], t_pi_y), ([], t / float(n_pi)), (['mw_x'], t_pi_x), ([], t / float(n_pi))] 
            sequence += [ (['mw_y'], t_pi_y), ([], t / float(n_pi)), (['mw_x'], t_pi_x), ([], t / float(2 * n_pi)), (['mw_x'], t_3pi2_x)]
            sequence += [ (['detect', 'aom'], laser), ([], wait)             ]            
        #sequence += n_ref * [ (['laser', 'aom'], laser), ([], wait)       ]
        #sequence += n_ref * [ (['mw_y'], t_pi_y), (['laser', 'aom'], laser), ([], wait)      ]
        sequence += [ (['sync'], 100)                        ]
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('t_pi2_x', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi_x', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi_y', width= -80, enabled_when='state != "run"'),
                                                   Item('t_3pi2_x', width= -80, enabled_when='state != "run"'),
                                                   Item('n_pi', width= -80, enabled_when='state != "run"'),
                                                   Item('n_ref', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'
                                            ),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'
                                            ),
                                     ),
                              ),
                       title='XY8 Measurement with both pi/2 and 3pi/2 readout pulse',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi2_x', 't_pi_x','t_pi_y', 't_3pi2_x', 'n_pi', 'n_ref']


class CP(Rabi):
    
    """Defines a CP measurement."""

    t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 x pulse length', label='pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi = Range(low=1., high=100000., value=1000., desc='pi y pulse length', label='pi y [ns]', mode='text', auto_set=False, enter_set=True)
    n_pi = Range(low=1, high=100, value=5, desc='number of pi pulses', label='n pi', mode='text', auto_set=False, enter_set=True)
    
    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2 = self.t_pi2
        t_pi = self.t_pi
        n_pi = self.n_pi
        
        sequence = []
        for t in tau:
            sequence += [ (['mw_x','mw'], t_pi2), ([], t / float(2 * n_pi)) ]
            sequence += (n_pi - 1) * [ (['mw_x','mw'], t_pi), ([], t / float(n_pi))   ] 
            sequence += [ (['mw_x','mw'], t_pi), ([], t / float(2 * n_pi)) ]
            sequence += [ (['mw_x','mw'], t_pi2)                             ]
            sequence += [ (['detect', 'aom'], laser), ([], wait)            ]
        sequence += [ (['sync'], 100)                        ]
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('t_pi2', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi', width= -80, enabled_when='state != "run"'),
                                                   Item('n_pi', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                                     ),
                              ),
                       title='CP Measurement',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi2', 't_pi', 'n_pi']


class CPMG(Rabi):
    
    """Defines a basic CPMG measurement with a single sequence and bright / dark reference points."""

    t_pi2_x = Range(low=1., high=100000., value=1000., desc='pi/2 x pulse length', label='pi/2 x [ns]', mode='text', auto_set=False, enter_set=True)
    t_pi_y = Range(low=1., high=100000., value=1000., desc='pi y pulse length', label='pi y [ns]', mode='text', auto_set=False, enter_set=True)
    n_pi = Range(low=1, high=100, value=5, desc='number of pi pulses', label='n pi', mode='text', auto_set=False, enter_set=True)
    n_ref = Range(low=1, high=100, value=10, desc='number of reference pulses', label='n ref', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2_x = self.t_pi2_x
        t_pi_y = self.t_pi_y
        n_pi = self.n_pi
        n_ref = self.n_ref
        sequence = []
        for t in tau:
            sequence += [ (['mw_x','mw'], t_pi2_x), ([], t / float(2 * n_pi)) ]
            sequence += (n_pi - 1) * [ (['mw_y','mw'], t_pi_y), ([], t / float(n_pi))   ] 
            sequence += [ (['mw_y','mw'], t_pi_y), ([], t / float(2 * n_pi)) ]
            sequence += [ (['mw_x','mw'], t_pi2_x)                             ]
            sequence += [ (['detect', 'aom'], laser), ([], wait)            ]
        sequence += n_ref * [ (['detect', 'aom'], laser), ([], wait)       ]
        sequence += n_ref * [ (['mw_y'], t_pi_y), (['laser', 'aom'], laser), ([], wait)      ]
        sequence += [ (['sync'], 100)                        ]
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('t_pi2_x', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi_y', width= -80, enabled_when='state != "run"'),
                                                   Item('n_pi', width= -80, enabled_when='state != "run"'),
                                                   Item('n_ref', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                                     ),
                              ),
                       title='Basic CPMG Measurement with pi/2 pulse on B and pi pulses on A',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi2_x', 't_pi_y', 'n_pi', 'n_ref']

class T1pi(Rabi):
    
    """Defines a T1 measurement with pi pulse."""

    t_pi = Range(low=1., high=100000., value=1000., desc='pi pulse length', label='pi [ns]', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi = self.t_pi
        sequence = []
        for t in tau:
            sequence.append(([       ], t))
            sequence.append((['detect', 'aom'], laser))
            sequence.append(([       ], wait))
        for t in tau:
            sequence.append((['mw' ,'mw_x'  ], t_pi))
            sequence.append(([       ], t))
            sequence.append((['detect', 'aom'], laser))
            sequence.append(([       ], wait))
        sequence.append((['sync'], 100))
        return sequence

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                                     ),
                              ),
                       title='T1pi',
                       )
    
    get_set_items = Rabi.get_set_items + ['t_pi']


class FID3pi2(Rabi):
    
    """Defines a FID measurement with both pi/2 and 3pi/2 readout pulse."""

    t_pi2 = Range(low=1., high=100000., value=1000., desc='pi/2 pulse length', label='pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_3pi2 = Range(low=1., high=100000., value=1000., desc='3pi/2 pulse length', label='3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)

    def generate_sequence(self):
        tau = self.tau
        laser = self.laser
        wait = self.wait
        t_pi2 = self.t_pi2
        t_3pi2 = self.t_3pi2
        sequence = []
        for t in tau:
            sequence.append(([       ], 100))
            sequence.append((['mw','mw_x'], t_pi2))
            sequence.append(([       ], t))
            sequence.append((['mw','mw_x'   ], t_pi2))
            sequence.append(([       ], wait))
            sequence.append((['detect', 'aom'], laser))
        for t in tau:
            sequence.append(([       ], 100))
            sequence.append((['mw','mw_x'   ], t_pi2))
            sequence.append(([       ], t))
            sequence.append((['mw','mw_x'   ], t_3pi2))
            sequence.append(([       ], wait))
            sequence.append((['detect', 'aom'], laser))
        sequence.append((['sync'], 100))
        return sequence
    
    #items to be saved
    get_set_items = Rabi.get_set_items + ['t_pi2', 't_3pi2']

    # gui elements
    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority'),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly', format_str='%.f'),
                                     Item('stop_time'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('frequency', width= -80, enabled_when='state != "run"'),
                                                   Item('power', width= -80, enabled_when='state != "run"'),
                                                   Item('t_pi2', width= -80, enabled_when='state != "run"'),
                                                   Item('t_3pi2', width= -80, enabled_when='state != "run"'),),
                                            HGroup(Item('tau_begin', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_end', width= -80, enabled_when='state != "run"'),
                                                   Item('tau_delta', width= -80, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state != "run"'),
                                                   Item('wait', width= -80, enabled_when='state != "run"'),
                                                   Item('record_length', width= -80, enabled_when='state != "run"'),
                                                   Item('bin_width', width= -80, enabled_when='state != "run"'),),
                                            label='settings'),
                                     ),
                              ),
                       title='FID3pi2',
                       )
  
class DEER(Pulsed):
    
    mw1_power = Range(low= -100., high=25., value=5.0, desc='MW1 Power [dBm]', label='MW1 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    mw1_frequency = Range(low=1., high=20.e9, value=2.61e9, desc='MW1 frequency [Hz]', label='MW1 frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
    mw2_power = Range(low= -100., high=25., value=7.0, desc='MW2 Power [dBm]', label='MW2 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    rf_begin = Range(low=1, high=20e9, value=100.0e6, desc='Start Frequency [Hz]', label='Begin [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf_end = Range(low=1, high=20e9, value=400.0e6, desc='Stop Frequency [Hz]', label='End [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf_delta = Range(low=1e-3, high=20e9, value=2.0e6, desc='frequency step [Hz]', label='Delta [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    t_mw2_pi = Range(low=1., high=100000., value=90., desc='length of pi pulse of mw2 [ns]', label='mw2 pi [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=10000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    tau = Range(low=1., high=100000., value=300., desc='tau [ns]', label='tau [ns]', mode='text', auto_set=False, enter_set=True)
    seconds_per_point = Range(low=20e-3, high=1, value=0.2, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
 
    sweeps_per_point = Int()
    run_time = Float(value=0.0)
        
    frequencies = Array(value=np.array((0., 1.)))   
    
      
    def generate_sequence(self):
        if self.t_mw1_pi > self.t_mw2_pi:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw2_pi), (['mw_a'], (self.t_mw1_pi - self.t_mw2_pi)), ([], self.tau), (['mw_a'], self.t_mw1_pi2) , (['sync'], 10)]
        else:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw1_pi), (['mw_b'], (self.t_mw2_pi - self.t_mw1_pi)), ([], (self.tau - (self.t_mw2_pi - self.t_mw1_pi))), (['mw_a'], self.t_mw1_pi2) , (['sync'], 10)]
        
    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        frequencies = np.arange(self.rf_begin, self.rf_end + self.rf_delta, self.rf_delta)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()

        if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.all(frequencies == self.frequencies)): # if the sequence and time_bins are the same as previous, keep existing data
            self.count_data = np.zeros((len(frequencies), n_bins))
            self.run_time = 0.0
        
        self.frequencies = frequencies
        self.sequence = sequence 
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (self.laser + self.wait + 2 * (self.t_mw1_pi2 + self.tau) + self.t_mw1_pi)))))
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    

    def _run(self):
        """Acquire data."""
        
        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()
            PulseGenerator().Night()
            Microwave().setOutput(self.mw1_power, self.mw1_frequency)
            MicrowaveB().setPower(self.mw2_power)
            tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 1, 0, 2, 3)
            tagger.setMaxCounts(self.sweeps_per_point)
            PulseGenerator().Sequence(self.sequence)

            while True:

                if self.thread.stop_request.isSet():
                    break

                t_start = time.time()
                
                for i, fi in enumerate(self.frequencies):
                    
                    MicrowaveB().setOutput(self.mw2_power, fi)                    
                    tagger.clear()
                    while not tagger.ready():
                        time.sleep(1.1 * self.seconds_per_point)
                    self.count_data[i, :] += tagger.getData()[0]
                                                            
                self.trait_property_changed('count_data', self.count_data)
                self.run_time += time.time() - t_start
                
            del tagger
            PulseGenerator().Light()
            Microwave().setOutput(None, self.mw1_frequency)
            MicrowaveB().setOutput(None, self.rf_begin)
        finally: # if anything fails, recover
            self.state = 'idle'        
            PulseGenerator().Light()
        
    get_set_items = Pulsed.get_set_items + ['mw1_power', 'mw1_frequency',
                                                    'mw2_power', 'rf_begin', 'rf_end', 'rf_delta', 't_mw2_pi', 't_mw1_pi2', 't_mw1_pi',
                                                       'laser', 'wait', 'seconds_per_point', 'frequencies', 'count_data', 'sync']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', width= -40),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mw1_power', width= -40, enabled_when='state != "run"'),
                                                   Item('mw1_frequency', width= -120, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi2', width= -40, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi', width= -40, enabled_when='state != "run"'),
                                                   Item('tau', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('mw2_power', width= -40, enabled_when='state != "run"'),
                                                   Item('rf_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('t_mw2_pi', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state == "idle"'),
                                                   Item('wait', width= -80, enabled_when='state == "idle"'),
                                                   Item('record_length', width= -80, enabled_when='state == "idle"'),
                                                   Item('bin_width', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('seconds_per_point', width= -120, enabled_when='state == "idle"'),
                                                   Item('sweeps_per_point', width= -120, style='readonly'),
                                                   ),
                                            label='settings'
                                            ),
                                     ),
                              ),
                        title='DEER, use frequencies fitting',
                        )

    
    mw1_power = Range(low= -100., high=25., value=5.0, desc='MW1 Power [dBm]', label='MW1 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    mw1_frequency = Range(low=1., high=20.e9, value=2.61e9, desc='MW1 frequency [Hz]', label='MW1 frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
    mw2_power = Range(low= -100., high=25., value=7.0, desc='MW2 Power [dBm]', label='MW2 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    t_mw2_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_mw2_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
    tau_begin = Range(low=1., high=1e8, value=15.0, desc='tau begin [ns]', label='tau begin [ns]', mode='text', auto_set=False, enter_set=True)
    tau_end = Range(low=1., high=1e8, value=30.0e3, desc='tau end [ns]', label='tau end [ns]', mode='text', auto_set=False, enter_set=True)
    tau_delta = Range(low=1., high=1e6, value=3.0e2, desc='delta tau [ns]', label='delta tau [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=10000., value=4000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    tau = Range(low=1., high=100000., value=300., desc='tau [ns]', label='tau [ns]', mode='text', auto_set=False, enter_set=True)
    seconds_per_point = Range(low=20e-3, high=1, value=0.2, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
 
    sweeps_per_point = Int()
    run_time = Float(value=0.0)
        
    frequencies = Array(value=np.array((0., 1.)))   
    
      
    def generate_sequence(self):
        if self.t_mw1_pi > self.t_mw2_pi:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw2_pi), (['mw_a'], (self.t_mw1_pi - self.t_mw2_pi)), ([], self.tau), (['mw_a'], self.t_mw1_pi2) , (['sync'], 10)]
        else:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw1_pi), (['mw_b'], (self.t_mw2_pi - self.t_mw1_pi)), ([], (self.tau - (self.t_mw2_pi - self.t_mw1_pi))), (['mw_a'], self.t_mw1_pi2) , (['sync'], 10)]
        
    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        frequencies = np.arange(self.rf_begin, self.rf_end + self.rf_delta, self.rf_delta)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()

        if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.all(frequencies == self.frequencies)): # if the sequence and time_bins are the same as previous, keep existing data
            self.count_data = np.zeros((len(frequencies), n_bins))
            self.run_time = 0.0
        
        self.frequencies = frequencies
        self.sequence = sequence 
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (self.laser + self.wait + 2 * (self.t_mw1_pi2 + self.tau) + self.t_mw1_pi)))))
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    

    def _run(self):
        """Acquire data."""
        
        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()
            PulseGenerator().Night()
            Microwave().setOutput(self.mw1_power, self.mw1_frequency)
            MicrowaveB().setPower(self.mw2_power)
            tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 1, 0, 2, 3)
            tagger.setMaxCounts(self.sweeps_per_point)
            PulseGenerator().Sequence(self.sequence)

            while True:

                if self.thread.stop_request.isSet():
                    break

                t_start = time.time()
                
                for i, fi in enumerate(self.frequencies):
                    
                    MicrowaveB().setOutput(self.mw2_power, fi)                    
                    tagger.clear()
                    while not tagger.ready():
                        time.sleep(1.1 * self.seconds_per_point)
                    self.count_data[i, :] += tagger.getData()[0]
                                                            
                self.trait_property_changed('count_data', self.count_data)
                self.run_time += time.time() - t_start
                
            del tagger
            PulseGenerator().Light()
            Microwave().setOutput(None, self.mw1_frequency)
            MicrowaveB().setOutput(None, self.rf_begin)
        finally: # if anything fails, recover
            self.state = 'idle'        
            PulseGenerator().Light()
        
    get_set_items = Pulsed.get_set_items + ['mw1_power', 'mw1_frequency',
                                                    'mw2_power', 'rf_begin', 'rf_end', 'rf_delta', 't_mw2_pi', 't_mw1_pi2', 't_mw1_pi',
                                                       'laser', 'wait', 'seconds_per_point', 'frequencies', 'count_data', 'sync']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', width= -40),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mw1_power', width= -40, enabled_when='state != "run"'),
                                                   Item('mw1_frequency', width= -120, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi2', width= -40, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi', width= -40, enabled_when='state != "run"'),
                                                   Item('tau', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('mw2_power', width= -40, enabled_when='state != "run"'),
                                                   Item('rf_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('t_mw2_pi', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state == "idle"'),
                                                   Item('wait', width= -80, enabled_when='state == "idle"'),
                                                   Item('record_length', width= -80, enabled_when='state == "idle"'),
                                                   Item('bin_width', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('seconds_per_point', width= -120, enabled_when='state == "idle"'),
                                                   Item('sweeps_per_point', width= -120, style='readonly'),
                                                   ),
                                            label='settings'
                                            ),
                                     ),
                              ),
                        title='Ce DEER, use Rabi fitting',
                        )

class DEER3pi2(Pulsed):
    
    mw1_power = Range(low= -100., high=25., value=5.0, desc='MW1 Power [dBm]', label='MW1 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    mw1_frequency = Range(low=1., high=20.e9, value=2.61e9, desc='MW1 frequency [Hz]', label='MW1 frequency [Hz]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi2 = Range(low=1., high=100000., value=83., desc='length of pi/2 pulse of mw1 [ns]', label='mw1 pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    t_mw1_pi = Range(low=1., high=100000., value=166., desc='length of pi pulse of mw1 [ns]', label='mw1 pi [ns]', mode='text', auto_set=False, enter_set=True)
    t_mw1_3pi2 = Range(low=1., high=100000., value=250., desc='length of 3pi/2 pulse of mw1 [ns]', label='mw1 3pi/2 [ns]', mode='text', auto_set=False, enter_set=True)
    mw2_power = Range(low= -100., high=25., value=7.0, desc='MW2 Power [dBm]', label='MW2 Power [dBm]', mode='text', auto_set=False, enter_set=True)
    rf_begin = Range(low=1, high=20e9, value=100.0e6, desc='Start Frequency [Hz]', label='Begin [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf_end = Range(low=1, high=20e9, value=400.0e6, desc='Stop Frequency [Hz]', label='End [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    rf_delta = Range(low=1e-3, high=20e9, value=2.0e6, desc='frequency step [Hz]', label='Delta [Hz]', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e'))
    t_mw2_pi = Range(low=1., high=100000., value=90., desc='length of pi pulse of mw2 [ns]', label='mw2 pi [ns]', mode='text', auto_set=False, enter_set=True)
    laser = Range(low=1., high=10000., value=3000., desc='laser [ns]', label='laser [ns]', mode='text', auto_set=False, enter_set=True)
    wait = Range(low=1., high=10000., value=1000., desc='wait [ns]', label='wait [ns]', mode='text', auto_set=False, enter_set=True)
    tau = Range(low=1., high=100000., value=300., desc='tau [ns]', label='tau [ns]', mode='text', auto_set=False, enter_set=True)
    seconds_per_point = Range(low=20e-3, high=1, value=0.2, desc='Seconds per point', label='Seconds per point', mode='text', auto_set=False, enter_set=True)
 
    sweeps_per_point = Int()
    run_time = Float(value=0.0)
        
    frequencies = Array(value=np.array((0., 1.)))   
    
      
    def generate_sequence(self):
        if self.t_mw1_pi > self.t_mw2_pi:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw2_pi), (['mw_a'], (self.t_mw1_pi - self.t_mw2_pi)), ([], self.tau), (['mw_a'], self.t_mw1_pi2) , (['laser', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw2_pi), (['mw_a'], (self.t_mw1_pi - self.t_mw2_pi)), ([], self.tau), (['mw_a'], self.t_mw1_3pi2) , (['sync'], 10)]
        else:
            return 100 * [ (['detect', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw1_pi), (['mw_b'], (self.t_mw2_pi - self.t_mw1_pi)), ([], (self.tau - (self.t_mw2_pi - self.t_mw1_pi))), (['mw_a'], self.t_mw1_pi2) , (['laser', 'aom'], self.laser), ([], self.wait), (['mw_a'], self.t_mw1_pi2), ([], self.tau), (['mw_a', 'mw_b'], self.t_mw1_pi), (['mw_b'], (self.t_mw2_pi - self.t_mw1_pi)), ([], (self.tau - (self.t_mw2_pi - self.t_mw1_pi))), (['mw_a'], self.t_mw1_3pi2) , (['sync'], 10)]
        
    
    def apply_parameters(self):
        """Apply the current parameters and decide whether to keep previous data."""

        frequencies = np.arange(self.rf_begin, self.rf_end + self.rf_delta, self.rf_delta)
        n_bins = int(self.record_length / self.bin_width)
        time_bins = self.bin_width * np.arange(n_bins)
        sequence = self.generate_sequence()

        if not (self.keep_data and sequence == self.sequence and np.all(time_bins == self.time_bins) and np.all(frequencies == self.frequencies)): # if the sequence and time_bins are the same as previous, keep existing data
            self.count_data = np.zeros((2 * len(frequencies), n_bins))
            self.run_time = 0.0
        
        self.frequencies = frequencies
        self.sequence = sequence 
        self.time_bins = time_bins
        self.n_bins = n_bins
        self.sweeps_per_point = int(np.max((1, int(self.seconds_per_point * 1e9 / (self.laser + self.wait + 2 * (self.t_mw1_pi2 + self.tau) + self.t_mw1_pi)))))
        self.keep_data = True # when job manager stops and starts the job, data should be kept. Only new submission should clear data.
    

    def _run(self):
        """Acquire data."""
        
        try: # try to run the acquisition from start_up to shut_down
            self.state = 'run'
            self.apply_parameters()
            PulseGenerator().Night()
            Microwave().setOutput(self.mw1_power, self.mw1_frequency)
            MicrowaveB().setPower(self.mw2_power)
            tagger = TimeTagger.Pulsed(self.n_bins, int(np.round(self.bin_width * 1000)), 2, 0, 2, 3)
            tagger.setMaxCounts(self.sweeps_per_point)
            PulseGenerator().Sequence(self.sequence)

            while True:

                if self.thread.stop_request.isSet():
                    break

                t_start = time.time()
                
                for i, fi in enumerate(self.frequencies):
                    
                    MicrowaveB().setOutput(self.mw2_power, fi)                    
                    tagger.clear()
                    while not tagger.ready():
                        time.sleep(1.1 * self.seconds_per_point)
                    count = tagger.getData()
                    self.count_data[i, :] += count[0]
                    self.count_data[i + len(self.frequencies), :] += count[1]
                                                            
                self.trait_property_changed('count_data', self.count_data)
                self.run_time += time.time() - t_start
                
            del tagger
            PulseGenerator().Light()
            Microwave().setOutput(None, self.mw1_frequency)
            MicrowaveB().setOutput(None, self.rf_begin)
        finally: # if anything fails, recover
            self.state = 'idle'        
            PulseGenerator().Light()
        
    get_set_items = Pulsed.get_set_items + ['mw1_power', 'mw1_frequency',
                                                    'mw2_power', 'rf_begin', 'rf_end', 'rf_delta', 't_mw2_pi', 't_mw1_pi2', 't_mw1_pi',
                                                       'laser', 'wait', 'seconds_per_point', 'frequencies', 'count_data', 'sync']

    traits_view = View(VGroup(HGroup(Item('submit_button', show_label=False),
                                     Item('remove_button', show_label=False),
                                     Item('resubmit_button', show_label=False),
                                     Item('priority', width= -40),
                                     Item('state', style='readonly'),
                                     Item('run_time', style='readonly'),
                                     ),
                              Tabbed(VGroup(HGroup(Item('mw1_power', width= -40, enabled_when='state != "run"'),
                                                   Item('mw1_frequency', width= -120, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi2', width= -40, enabled_when='state != "run"'),
                                                   Item('t_mw1_pi', width= -40, enabled_when='state != "run"'),
                                                   Item('t_mw1_3pi2', width= -40, enabled_when='state != "run"'),
                                                   Item('tau', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            HGroup(Item('mw2_power', width= -40, enabled_when='state != "run"'),
                                                   Item('rf_begin', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_end', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('rf_delta', width= -80, editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_func=lambda x:'%e' % x)),
                                                   Item('t_mw2_pi', width= -40, enabled_when='state != "run"'),
                                                   ),
                                            label='parameter'),
                                     VGroup(HGroup(Item('laser', width= -80, enabled_when='state == "idle"'),
                                                   Item('wait', width= -80, enabled_when='state == "idle"'),
                                                   Item('record_length', width= -80, enabled_when='state == "idle"'),
                                                   Item('bin_width', width= -80, enabled_when='state == "idle"'),
                                                   ),
                                            HGroup(Item('seconds_per_point', width= -120, enabled_when='state == "idle"'),
                                                   Item('sweeps_per_point', width= -120, style='readonly'),
                                                   ),
                                            label='settings'
                                            ),
                                     ),
                              ),
                        title='DEER 3pi/2, use frequencies 3pi2 fitting',
                        )

        
                        
if __name__ == '__main__':
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')
    
    from tools.emod import JobManager
    
    JobManager().start()
    
    r1 = Rabi()
    r2 = Rabi()
    
    r1.edit_traits()
    r2.edit_traits()
    

