import time
import numpy as np
import nidaqmx as ni

from .nidaq import sample_clock, sample_DoubleClock, analog_output_sweeper


def Pulse_Train_Counter(
    time_tagger,
    device_name = 'dev1',
    ctr_list = ['ctr0','ctr1'],
    duty_cycle = 0.9,
    sec_per_point = 0.01, 
):
    frame_size = 10     # Just an initial value, it can be changed later.
    clk = sample_DoubleClock(
        device_name, ctr_list,
        samps_per_chan=frame_size, period=sec_per_point, duty_cycle=duty_cycle,
    )
    clk.prepare_task()
    
    counter = pulsetrain_counter(
        clk, time_tagger,
    )
    return counter

# Factory function of class piezostage_controller_aom
def Stage_control(
    time_tagger,
    device_name = 'dev1',
    counter_name = 'ctr0',
    ao_channels = ['ao0', 'ao1', 'ao2', 'ao3'],
    voltage_range = [
        [0., 1.],       # ao0
        [0., 1.],       # ao1
        [0., 1.],       # ao2
        [0., 1.],       # ao3
    ],
    sec_per_point = .01,
    duty_cycle = 0.9,
    x_range=(-100.0,100.0),
    y_range=(-100.0,100.0),
    z_range=(0,100.0),
    aom_range=(-10,10),
    home_pos=None,
    invert_x=False,
    invert_y=False,
    invert_z=False,
    swap_xy=False,
):
    frame_size = 10     # Just an initial value, it can be changed later.
    clk = sample_clock(
        device_name, counter_name,
        samps_per_chan=frame_size, period=sec_per_point, duty_cycle=duty_cycle,
    )
    clk.prepare_task()
    
    ao_sweep = analog_output_sweeper(
        device_name, ao_channels, voltage_range,
        clk, use_falling=True
    )
    ao_sweep.prepare_task()
    
    stage = piezostage_controller_aom(
        ao_sweep, clk, time_tagger,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        aom_range=aom_range,
        home_pos=home_pos,
        invert_x=invert_x,
        invert_y=invert_y,
        invert_z=invert_z,
        swap_xy=swap_xy,
    )
    return stage

class piezostage_controller_aom:
    
    def __init__(self,
        ao_task, sample_clk, tagger,
        x_range=(-100.0,100.0),
		y_range=(-100.0,100.0),
        z_range=(0,100.0),
        aom_range=(-10,10),
        home_pos=None,
        invert_x=False, invert_y=False, invert_z=False, swap_xy=False, 
    ):
        self.ao_task = ao_task          # A NI AO task controlling analog output to control x, y, z & aom.
        self.sample_clk = sample_clk    # A NI CO task providing clock signal to trigger the sweeping and readout.
        self.tagger = tagger            # A Timtagger object providing readout in sync with the NI sweeping.

        self.xRange = x_range
        self.yRange = y_range
        self.zRange = z_range
        self.aomRange = aom_range
        if home_pos:
            self.home_pos = np.array(home_pos)
        else:
            self.home_pos = np.array([
                np.mean(self.xRange),
                np.mean(self.yRange),
                np.mean(self.zRange),
                np.mean(self.aomRange),
            ])

        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self.swap_xy = swap_xy
        self.setPosToHome()

    def getXRange(self):
        return self.xRange

    def getYRange(self):
        return self.yRange

    def getZRange(self):
        return self.zRange
    
    def getAOMRange(self):
        return self.aomRange
    
    def PosToVolt(self, pos):
        posRange = np.vstack([self.xRange, self.yRange, self.zRange, self.aomRange])
        vRange = self.ao_task.vrange
        vLow = vRange[:,[0]]
        vHigh = vRange[:,[1]]
        vDiff = vHigh - vLow
        posLow = posRange[:,[0]]
        posHigh = posRange[:,[1]]
        posDiff = posHigh - posLow
        
        # print(vDiff.shape, posDiff.shape,pos.shape)

        vOutput = vLow + vDiff*(pos - posLow)/posDiff
        # print(vOutput)

        mask = [self.invert_x, self.invert_y, self.invert_z, False]
        vOutput[mask] = vLow[mask] + vDiff[mask]*(posHigh[mask] - pos[mask])/posDiff[mask]
        if self.swap_xy:
            return vOutput[[1,0,2,3]]
        else:
            return vOutput
    
    def setx(self, x):
        self.setPosition([x, self.y, self.z, self.aom])

    def sety(self, y):
        self.setPosition([self.x, y, self.z, self.aom])
    
    def setz(self, z):
        self.setPosition([self.x, self.y, z, self.aom])
    
    def setaom(self, aom):
        self.setPosition([self.x, self.y, self.z, aom])

    def setPosToHome(self):
        self.setPosition(*self.home_pos)

    def setPosition(self, x, y, z, aom):
        pos = np.array([x, y, z, aom])[:,None]
        if not self.ao_task.on_demand:
            self.ao_task.on_demand = True
            self.ao_task.update_task()
        self.ao_task.write(self.PosToVolt(pos), auto_start=True)
        self.x, self.y, self.z, self.aom = x, y, z, aom

    def scanLine(self, Line, SecondsPerPoint, timeout=None, add_aom=True):
        
        Line = np.array(Line)
        # print(Line)
        frame_size = Line.shape[1]
        if not timeout:
            timeout = max(SecondsPerPoint*frame_size*1.5, 10)

        if add_aom:
            Line_aom = np.vstack((Line, self.aom*np.ones(frame_size)))
        else:
            Line_aom = Line

        self.sample_clk.period = SecondsPerPoint
        self.sample_clk.samps_per_chan = frame_size + 1
        self.sample_clk.update_task()

        self.ao_task.samps_per_chan = frame_size
        self.ao_task.sample_rate = self.sample_clk.sample_rate
        self.ao_task.on_demand = False
        self.ao_task.update_task()
        self.ao_task.write(self.PosToVolt(Line_aom))

        cbm_task = self.tagger.Count_Between_Markers(frame_size + 1)

        cbm_task.start()
        self.ao_task.start()
        time.sleep(.1)
        self.sample_clk.start()

        t = 0
        while not cbm_task.ready():
            time.sleep(0.1)
            t += 0.1
            if t > timeout:
                print(f'Scanning timeout! after {t:.1f} sec')
                break
        # sccuess = self.cbm_task.waitUntilFinished(timeout=timeout*1.e3)

        self.ao_task.stop()
        self.sample_clk.stop()
        cbm_task.stop()
        
        if cbm_task.ready():
            scale = self.sample_clk.sample_rate*self.sample_clk.duty_cycle
            data = cbm_task.getData()
            return data[1:]*scale
        else:
            print(f'Fail to get data from Timetagger!')
            return np.zeros(frame_size)
    

class pulsetrain_counter:

    def __init__(self, sample_clk, tagger):
        self.sample_clk = sample_clk    # A NI CO task providing clock signal to trigger the sweeping and readout.
        self.tagger = tagger            # A Timtagger object providing readout in sync with the NI sweeping.
        self.cbm_task = None

    def configure(self, frame_size, SecondsPerPoint, DutyCycle=0.8):
        self.sample_clk.period = SecondsPerPoint

        '''
        From microwave_smiq.SMIQ:
            we switch frequency on negative edge. Thus, the first square pulse of the train
            is first used for gated count and then the frequency is increased. In this way
            the first frequency in the list will correspond exactly to the first acquired count.
        '''
        self.sample_clk.samps_per_chan = frame_size
        self.sample_clk.duty_cycle = DutyCycle
        self.sample_clk.update_task()
        self.cbm_task = self.tagger.Count_Between_Markers(frame_size)

    def run(self, timeout=None):
        frame_size = self.sample_clk.samps_per_chan
        period = self.sample_clk.period
        sample_rate = self.sample_clk.sample_rate
        duty_cycle = self.sample_clk.duty_cycle
        # print(frame_size, period, duty_cycle)

        if not timeout:
            time_per_line = period*frame_size
            timeout = max(time_per_line*1.5, 4)

        if not self.cbm_task:
            print('TimeTagger task has not been created!')
            print('Call configure() to initialize scanning.')

        self.cbm_task.start()
        time.sleep(.1)
        self.sample_clk.start()

        t = 0
        while not self.cbm_task.ready():
            time.sleep(0.1)
            t += 0.1
            if t > timeout:
                print(f'Scanning timeout! after {t:.1f} sec')
                break
        
        # sccuess = self.cbm_task.waitUntilFinished(timeout=timeout*1.e3)
        
        self.sample_clk.stop()
        self.cbm_task.stop()

        if self.cbm_task.ready():
            scale = sample_rate*duty_cycle
            bin_width = self.cbm_task.getBinWidths()
            data = self.cbm_task.getData()
            self.cbm_task.clear()
            return data/bin_width/1.e-12#*scale
        else:
            print(f'Fail to get data from Timetagger! Scanning timeout after {timeout:.1f} sec')
            return np.zeros(frame_size)
    
    def clear(self):
        self.cbm_task.clear()
