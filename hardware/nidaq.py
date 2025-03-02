import numpy as np
import nidaqmx as ni


def is_device_available(device_name):
    avail_device_names = ni.system.System().devices.device_names
    return device_name in set(dev.lower() for dev in avail_device_names)

def is_counter_available(device_handle, counter_name):
    avail_counter_names = [
        ctr.split('/')[-1] for ctr in device_handle.co_physical_chans.channel_names
    ]
    return counter_name in avail_counter_names

def is_ai_term_available(device_handle, term_name):
    avail_ai_term_names = [
        term.rsplit('/', 1)[-1].lower() for term in device_handle.ai_physical_chans.channel_names
    ]
    return term_name in avail_ai_term_names

def is_ao_term_available(device_handle, term_name):
    avail_ao_term_names = [
        term.rsplit('/', 1)[-1].lower() for term in device_handle.ao_physical_chans.channel_names
    ]
    return term_name in avail_ao_term_names

def is_pfi_term_available(device_handle, term_name):
    avail_pfi_term_names = [
        term.rsplit('/', 1)[-1].lower() for term in device_handle.terminals if 'PFI' in term
    ]
    return term_name in avail_pfi_term_names


class ni_tasks_manager:

    '''
    Static class to handle all the NI task globally.
    Use this class to create and remove NI tasks to prevent losing track of the task handle.
    
    We don't want to create an instance for this class, since we only need one manager to handle
    all NI task across all modules. Ideally, we will keep using the exact same class until we 
    restart the entire program.
    '''

    tasks_dict = dict()
    
    def create_task(task_name):
        TD = ni_tasks_manager.tasks_dict
        if task_name in TD.keys():
            raise Exception(f'{task_name} has already benn created!')
        
        try:
            task = ni.Task(task_name)
        except:
            raise Exception(f'Fail to create task named {task_name}.')
        TD[task_name] = task        
        return task

    def remove_task(name=None, task=None):
        TD = ni_tasks_manager.tasks_dict
        if task:
            if not task in TD.values():
                raise Exception('Fail to find {task} from our record!')
            for k, t in TD.items():
                if id(task) == id(t):
                    name = k
                    break
        if name:
            try:
                task = TD.pop(name)
                try:
                    task.close()
                    del task
                except ni.DaqError:
                    print('Fail to delete task {name}')
            except:
                raise Exception('Fail to remove task {name}')
        else:
            print('Specify which task to be removed.')

    def is_name_exist(task_name):
        return task_name in ni_tasks_manager.tasks_name_list.keys()
    
    def is_task_exist(task):
        return id(task) in [id(t) for t in ni_tasks_manager.values()]


class task_constructor:

    '''
    This class provides a framework on how you should design a class to handle tasks.

    Behavior:
        - If an instance of isn't being used, always clear the task and keep self.task = None
        - Preparing a task always involve three steps
            1. Acquire task handle by calling ni_tasks_manager.create_task()
                Avoid directly calling ni.Task() unless you are confident to keep track on all task handle.
                If a task reserved NIDAQ resources and you lose track of its task handle to clear it, it
                will result in crashing of the program.
            2. Create the desired output type
                It could be a list of analog output voltages, a list of analog/digital input bins for
                readout. It can also be a constant analog/digital output.
            3. Configure the timing
                NIDAQ needs to know when to output the values that you set at #2.
                It can be:
                - on_demand: 
    '''

    def __init__(self, device_name):
        self.device_name = device_name
        self.task = None
        self.task_name = ''

    # Overwrite it to create actual task
    def prepare_task(self):
        # 1. Acquire task handle
        if self.task:
            self.clear_task()
        self.task_name = f'task_{id(self)}'
        self.task = ni_tasks_manager.create_task(self.task_name)
        # 2. Configure output
        # 3. Configure Timing
        pass

    def clear_task(self):
        if self.task:
            # Clear task via ni_tasks_manager.remove_task() w/o calling task.close() directly.
            ni_tasks_manager.remove_task(task=self.task)
            del self.task
            self.task = None
            self.task_name = ''
        else:
            print('No task has been created.')

    def start(self):
        if not self.task:
            print('No task is created.')
            return
        if not self.task.is_task_done():
            print(f'{self.task_name}: The task has already started!')
            return
        try:
            self.task.start()
        except ni.DaqError:
            raise Exception(f'{self.task_name}: Failed to start task')

    def stop(self):
        if not self.task:
            print('No task is created.')
            return
        try:
            self.task.stop()
        except ni.DaqError:
            raise Exception(f'{self.task_name}: Failed to stop task')


class sample_clock(task_constructor):

    '''
    Task constructor for creating a sample clock signal from NIDAQ internal clock.
    Almost all scanning task (confocal or ODMR) needs this to sync triggers and readout.
    '''

    def __init__(self, device_name, counter_name, period=0.01, duty_cycle=0.9, samps_per_chan=None):
        super().__init__(device_name)
        self.counter_name = counter_name
        self.period = period
        self.duty_cycle = duty_cycle
        self.samps_per_chan = samps_per_chan
        
    @property
    def sample_rate(self):
        return 1/self.period
    
    @sample_rate.setter
    def sample_rate(self, rate):
        self.period = 1/rate

    @property
    def source(self):
        return f'/{self.device_name}/{self.counter_name}'

    def prepare_task(self):
        # 1. Acquire task handle
        if self.task:
            self.clear_task()
        self.task_name = f'clock_{id(self)}'
        self.task = ni_tasks_manager.create_task(self.task_name)

        # 2. Configure output
        self.output = self.task.co_channels.add_co_pulse_chan_freq(
            self.source,
            idle_state=ni.constants.Level.LOW,
            freq=self.sample_rate,
            duty_cycle=self.duty_cycle,
        )

        # 3. Configure Timing
        self.config_timing()

    def config_timing(self):
        samps_per_chan = self.samps_per_chan if self.samps_per_chan else 1000
        mode = ni.constants.AcquisitionType.FINITE if self.samps_per_chan else ni.constants.AcquisitionType.CONTINUOUS
        self.task.timing.cfg_implicit_timing(
            sample_mode=mode, samps_per_chan=samps_per_chan,
        )

    def update_task(self):
        self.output.co_pulse_freq = self.sample_rate
        self.config_timing()


class analog_output_constant(task_constructor):

    def __init__(self, device_name, ao_channels, voltage_range):
        super().__init__(device_name)
        self._ao_channels = ao_channels
        self.vrange = np.array(voltage_range)
        self.ao_list = []

    @property
    def ao_channels(self):
        return [f'/{self.device_name}/{ch}' for ch in self._ao_channels]

    def prepare_task(self):
        # 1. Acquire task handle
        if self.task:
            self.clear_task()
        self.task_name = f'ao_{id(self)}'
        self.task = ni_tasks_manager.create_task(self.task_name)

        # 2. Configure output
        self.ao_list = []
        for i, ch in enumerate(self.ao_channels):
            ao = self.task.ao_channels.add_ao_voltage_chan(
                physical_channel=ch,
                min_val=self.vrange[i,0],
                max_val=self.vrange[i,1]
            )
            self.ao_list.append(ao)

        # 3. Configure timing
        # Default timing is on-demand
    
    def write(self, values, auto_start=False):
        '''
        - Scalar:
            Single sample for 1 channel.
        - List/1D numpy.ndarray:
            Multiple samples for 1 channel or 1 sample for multiple channels.
        - List of lists/2D numpy.ndarray:
            Multiple samples for multiple channels.
        '''
        if not self.task:
            print('No task is created.')
            return
        try:
            self.task.write(values, auto_start=auto_start)
        except ni.DaqError:
            raise Exception(f'Failed to write values to {self.task}')

    def update_task(self):
        for i, ao in enumerate(self.ao_list):
            ao.ao_min = self.vrange[i,0]
            ao.ao_max = self.vrange[i,1]


class analog_output_sweeper(analog_output_constant):

    def __init__(self,
        device_name, ao_channels, voltage_range,
        samp_clk, use_falling=False
    ):
        super().__init__(device_name, ao_channels, voltage_range)
        self.samp_clk = samp_clk
        self.use_falling = use_falling
        self.on_demand = False
        self.samps_per_chan = 10

    @property
    def active_edge(self):
        if self.use_falling:
            return ni.constants.Edge.FALLING
        else:
            return ni.constants.Edge.RISING

    def prepare_task(self):

        # 1. Acquire task handle
        # 2. Configure output
        super().prepare_task()

        # 3. Configure timing
        self.config_timing()

    def config_timing(self):
        # Restore to on-demand mode
        if self.on_demand:
            self.task.timing.samp_timing_type = ni.constants.SampleTimingType.ON_DEMAND
            return
        
        if not self.samp_clk.samps_per_chan:
            print('Specify frame size of the sampling clock to use FINITE mode')
            return

        src = self.samp_clk.source + 'InternalOutput'
        rate = self.samp_clk.sample_rate
        self.task.timing.cfg_samp_clk_timing(
            rate,
            source=src,
            active_edge=self.active_edge,
            samps_per_chan=self.samps_per_chan
        )
    
    def update_task(self):
        super().update_task()
        self.config_timing()

