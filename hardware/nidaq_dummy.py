import numpy as np
import time

class Scanner:

    def __init__(
        self, CounterIn, CounterOut, TickSource, AOChannels,
		x_range, y_range, z_range, v_range=(0.,10.),
	    invert_x=False, invert_y=False, invert_z=False, swap_xy=False, TriggerChannels=None
    ):
        self.xRange = x_range
        self.yRange = y_range
        self.zRange = z_range
        self.vRange = v_range
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
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
    
    def setx(self, x):
        self.x = x
    
    def sety(self, y):
        self.y = y
    
    def setz(self, z):
        self.z = z
    
    def setPosition(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def scanLine(self, Line, SecondsPerPoint, return_speed=None):
        N = Line.shape[1]
        time.sleep(N*SecondsPerPoint)
        return np.random.rand(N)

    def PosToVolt(self, r):
        x = self.xRange
        y = self.yRange
        z = self.zRange
        v = self.vRange
        v0 = v[0]
        dv = v[1]-v[0]
        if self.invert_x:
            vx = v0+(x[1]-r[0])/(x[1]-x[0])*dv         

        else:
            vx = v0+(r[0]-x[0])/(x[1]-x[0])*dv

        if self.invert_y:
            vy = v0+(y[1]-r[1])/(y[1]-y[0])*dv

        else:
            vy = v0+(r[1]-y[0])/(y[1]-y[0])*dv

        if self.invert_z:
            vz = v0+(z[1]-r[2])/(z[1]-z[0])*dv            
        else:
            vz = v0+(r[2]-z[0])/(z[1]-z[0])*dv

        if self.swap_xy:
            vt = vx
            vx = vy
            vy = vt            
        return np.vstack((vx,vy,vz)).T


class PulseTrainCounter:

    def __init__(self, CounterIn, CounterOut, TickSource):
		
        self._CounterIn = CounterIn
        self._CounterOut = CounterOut
        self._TickSource = TickSource

    def configure(self, SampleLength, SecondsPerPoint, DutyCycle=0.9, MaxCounts=1e7, RWTimeout=1.0):
        self._CIData = np.empty((SampleLength,), dtype=np.uint32)
        self.sleep_time = SecondsPerPoint*SampleLength

    def run(self):
        time.sleep(self.sleep_time)
        self._CIData = np.random.randint(0, 100, self._CIData.shape)
        return self._CIData
    
    def clear(self):
        pass
