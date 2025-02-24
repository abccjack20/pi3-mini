import numpy as np
import time

class PulseGeneratorDummy:

    def __init__(self, serial, channel_map):
        self.serial = serial
        self.channel_map = channel_map
    
    def Continuous(self, channels):
        pass
        
    def Sequence(self, sequence, loop=True):
        pass

    def Loadseq(self, c_pulses, length, loop0, seg_num, loop=True):
        pass

    def SaveSeq(self, sequence):
        return 0
        
    def Run(self, loop=None):
        pass
        
    def Night(self):
        pass

    def Light(self):
        pass

    def Open(self):
        pass

    def checkUnderflow(self):
        return False