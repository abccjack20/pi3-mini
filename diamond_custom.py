
from tools.utility import StoppableThread
from datetime import date
import os, time, threading




# start confocal including auto_focus tool and Toolbox
if __name__ == '__main__':
    
    print("Running Python3 ...")

    from hardware.api import PulseGenerator
    PulseGenerator().Night()

    # Photon Time Trace Startup
    from measurements.photon_time_trace import PhotonTimeTrace
    photon_time_trace = PhotonTimeTrace()
    photon_time_trace.c_enable1 = True
    photon_time_trace.c_enable2 = True
    photon_time_trace.sum_enable = True
    photon_time_trace.digi_channel = 'cha0+1'
    photon_time_trace.edit_traits()

    # Start confocal including auto_focus tool
    from measurements.confocal import Confocal
    confocal = Confocal()
    confocal.edit_traits()

    # Start autofocus tool
    from measurements.auto_focus import AutoFocus
    auto_focus = AutoFocus(confocal)
    auto_focus.edit_traits()

    # Start ODMR
    from measurements.odmr_ps import ODMR
    odmr = ODMR()
    odmr.edit_traits()

    # Start Rabi / PulsedAnalyser
    from measurements.rabi import Rabi
    rabi = Rabi()
    rabi.edit_traits()

    from analysis.pulsed import PulsedAnalyzer
    pa = PulsedAnalyzer()
    pa.edit_traits()

    from hardware.awg5000_test import AWG5014, AWGManager
    awg_control = AWGManager(
        gpib='GPIB0::1::INSTR',
        ftp='169.254.103.111',
        socket=('169.254.103.111',4001),
    )
