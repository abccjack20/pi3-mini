
"""pi3diamond startup script"""

import logging, logging.handlers
import os
import inspect

path = os.path.dirname(inspect.getfile(inspect.currentframe()))

# First thing we do is start the logger
logging_handler = logging.handlers.TimedRotatingFileHandler(path+'/log/diamond_log.txt', 'W6') # start new file every sunday, keeping all the old ones 
logging_handler.setFormatter(logging.Formatter("%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(logging_handler)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.INFO) # we don't want the console to be swamped with debug messages
logging.getLogger().addHandler(stream_handler) # also log to stderr
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().info('Starting logger.')


# start the JobManager
from tools.emod import JobManager
JobManager().start()

# start the CronDaemon
from tools.cron import CronDaemon
CronDaemon().start()

# define a shutdown function
from tools.utility import StoppableThread
import threading

"""    
def startExperiment:
    ha.AWG().set_vpp(2.0, 0b01)
    ha.AWG().set_vpp(2.0, 0b10)
    ha.AWG().set_sampling(1.e9)
"""

# import sys,os
# sys.path.append('C:\\Program Files (x86)\\Swabian Instruments\\Time Tagger\\driver\\Python3.6\\x64\\')
# sys.path.append('C:\\Program Files (x86)\\Swabian Instruments\\Time Tagger\\driver\\x64')
# sys.path.append('C:\\Program Files (x86)\\Swabian Instruments\\Time Tagger\\driver\\firmware')

from hardware.nidaq import ni_tasks_manager
def shutdown():
    """Terminate all threads."""
    JobManager().stop()
    for t in threading.enumerate():
        if isinstance(t, StoppableThread):
            t.stop()
    ni_tasks_manager.clear_all_tasks()
    exit()

# That's it for now! We pass over control to custom startup script if present. 
if os.access(path+'/diamond_custom.py', os.F_OK):
    # PYTHON3 EDIT execfile(path+'/diamond_custom.py')
    exec(open("./diamond_custom.py").read())
