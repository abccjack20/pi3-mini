import numpy as np
import time

class MicrowaveDummy:

    def __init__(self, visa_address='GPIB0::28'):
        self.visa_address = visa_address
        self._power = 0
        self._onStatus = 0
        self._frequency = 2.87e9
        self._output_threshold = -60
        self._freqlist = []

    def _write(self, string):
        pass

    def _ask(self, str):
        return 0
    
    def getPower(self):
        return self._power
    
    def setPower(self, power):
        if power is None or power < self._output_threshold:
            self._onStatus = 0
        else:
            self._onStatus = 1
        self._power = power

    def onStatus(self):
        return self._onStatus

    def getFrequency(self):
        return self._frequency

    def setFrequency(self, frequency):
        self._frequency = frequency
    
    def setOutput(self, power, frequency):
        self.setPower(power)
        self.setFrequency(frequency)
    
    def initSweep(self, frequency, power):
        self._freqlist = frequency

    def resetListPos(self):
        pass