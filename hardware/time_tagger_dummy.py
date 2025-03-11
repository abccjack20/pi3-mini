import numpy as np

class TimeTaggerDummy:
    
    def Counter(self, ch, SecondsPerPoint, TraceLength):
        class dummy_counter:
            def getData():
                return np.random.normal(scale=1e3, size=(len(ch),TraceLength)) + 10e3
        return dummy_counter
    
    def Pulsed(self, n_bins, binwidth, n_lasers, *args):
        print(n_bins, binwidth, n_lasers, )
        class dummy_pulsed:
            def getData():
                return np.random.normal(scale=1e3, size=(n_lasers, n_bins)) + 10e3
        return dummy_pulsed