import time
import numpy as np
import nidaqmx as ni

from .nidaq import sample_clock, analog_output_sweeper


class stage_controller_aom:
    
    def __init__(self,
        ao_const, ao_sweep, sample_clk, tagger, ch_tt_marker,
        x_range=(-100.0,100.0),
		y_range=(-100.0,100.0),
        z_range=(0,100.0),
        aom_range=(-10,10),
        invert_x=False, invert_y=False, invert_z=False, swap_xy=False, 
    ):
        self.ao_const = ao_const        # A NI AO task providing constant analog output to control x, y, z & aom.
        self.ao_sweep = ao_sweep        # A NI AO task providing sweeping through different values of x, y and z.
        self.sample_clk = sample_clk    # A NI CO task providing clock signal to trigger the sweeping and readout.
        self.tagger = tagger            # A Timtagger object providing readout in sync with the NI sweeping.

        self.xRange = x_range
        self.yRange = y_range
        self.zRange = z_range
        self.aomRange = aom_range
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.aom = -10.
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self.swap_xy = swap_xy

    def getXRange(self):
        return self.xRange

    def getYRange(self):
        return self.yRange

    def getZRange(self):
        return self.zRange
    
    def PosToVolt(self, pos):
        posRange = np.vstack([self.xRange, self.yRange, self.zRange, self.aomRange])
        vRange = self.ao_const.vrange
        vLow = vRange[:,0]
        vHigh = vRange[:,1]
        vDiff = vLow - vHigh
        posLow = posRange[:,0]
        posHigh = posRange[:,1]

        vOutput = vLow + vDiff*(pos - posLow)/(posLow - posLow)

        mask = [self.invert_x, self.invert_y, self.invert_z]
        vOutput[mask] = vLow[mask] + vDiff[mask]*(posHigh[mask] - pos[mask])/(posLow[mask] - posLow[mask])
        if self.swap_xy:
            return vOutput[[1,0,2,3]]
        else:
            return vOutput
    
    def setx(self, x):
        pos = np.array([x, self.y, self.z, self.aom])
        self.ao_const.write(self.PosToVolt(pos), auto_start=True)
        self.x = x

    def sety(self, y):
        pos = np.array([self.x, y, self.z, self.aom])
        self.ao_const.write(self.PosToVolt(pos), auto_start=True)
        self.y = y
    
    def setz(self, z):
        pos = np.array([self.x, self.y, z, self.aom])
        self.ao_const.write(self.PosToVolt(pos), auto_start=True)
        self.z = z
    
    def setaom(self, aom):
        pos = np.array([self.x, self.y, self.z, aom])
        self.ao_const.write(self.PosToVolt(pos), auto_start=True)
        self.aom = aom

    def setPosition(self, x, y, z, aom):
        pos = np.array([x, y, z, aom])
        self.ao_const.write(self.PosToVolt(pos), auto_start=True)
        self.x, self.y, self.z, self.aom = x, y, z, aom


    def scanLine(self, Line, SecondsPerPoint, timeout=None):
        
        frame_size = Line.shape[1]
        if not timeout:
            timeout = max(SecondsPerPoint*frame_size*1.5, 10)

        self.sample_clk.period = SecondsPerPoint
        self.sample_clk.frame_size = frame_size + 1
        self.sample_clk.update_task()
        
        self.ao_sweep.samps_per_chan = frame_size
        self.ao_sweep.sampling_rate = self.sample_clk.sampling_rate
        self.ao_sweep.update_task()
        self.ao_sweep.write(self.PosToVolt(Line))

        cbm_task = self.tagger.Count_Between_Markers(frame_size + 1)

        self.ao_sweep.start()
        self.sample_clk.start()

        t = 0
        while not cbm_task.ready():
            time.sleep(0.1)
            t += 0.1
            if t > timeout:
                print(f'Scanning timeout! after {t} sec')
                self.ao_sweep.stop()
                self.sample_clk.stop()
                self.setPosition(self)
