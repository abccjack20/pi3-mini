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
        - Initiating a task always involve three steps
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

    def __init__(self, device_name, counter_name):
        self.device_name = device_name
        self.counter_name = counter_name
        self.source = f'/{self.device_name}/{self.counter_name}'
        self.task = None

    # Overwrite it to create actual task
    def init_task(self):
        # 1. Acquire task handle
        self.task = ni_tasks_manager.create_task(f'task_{id(self)}')
        # 2. Configure output
        # 3. Configure Timing
        pass

    def clear_task(self):
        if self.task:
            # Clear task via ni_tasks_manager.remove_task() w/o calling task.close() directly.
            ni_tasks_manager.remove_task(task=self.task)
            del self.task
            self.task = None
        else:
            print('No task has been created.')


class sample_clock(task_constructor):

    '''
    Task constructor for creating a sample clock signal from NIDAQ internal clock.
    Almost all scanning task (confocal or ODMR) needs this to sync triggers and readout.
    '''

    def __init__(self, *args, period=0.01, duty_cycle=0.9, frame_size=None):
        super().__init__(*args)
        self.period = period
        self.duty_cycle = duty_cycle
        self.frame_size = frame_size
        self.mode = ni.constants.AcquisitionType.FINITE if self.frame_size else ni.constants.AcquisitionType.CONTINUOUS

    @property
    def sample_rate(self):
        return 1/self.period
    
    @sample_rate.setter
    def sample_rate(self, rate):
        self.period = 1/rate

    def init_task(self):
        # 1. Acquire task handle
        self.task = ni_tasks_manager.create_task(f'clock_{id(self)}')

        # 2. Configure output
        self.task.co_channels.add_co_pulse_chan_freq(
            self.source,
            idle_state=ni.constants.Level.LOW,
            freq=self.sample_rate,
            duty_cycle=self.duty_cycle,
        )

        # 3. Configure Timing
        self.task.timing.cfg_implicit_timing(
            sample_mode=self.mode, samps_per_chan=self.frame_size,
        )

    


# class 
