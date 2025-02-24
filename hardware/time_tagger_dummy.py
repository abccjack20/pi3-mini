import numpy as np

class TimeTagger:
    
    def Counter(ch, SecondsPerPoint, TraceLength):
        class dummy_counter:
            def getData():
                return np.random.normal(scale=1e3, size=TraceLength) + 10e3
        return dummy_counter
    
    def Pulsed(n_bins, binwidth, n_lasers, *args):
        class dummy_pulsed:
            def getData():
                return np.random.normal(scale=1e3, size=(n_lasers, n_bins)) + 10e3
        return dummy_pulsed