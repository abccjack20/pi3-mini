# To install
	conda env create -f .\CondaEnvironment\environment.yml

# To run
	ipython --gui=qt -i diamond.py

# Check NIDAQ card status
	import nidaqmx as ni
	ni.system.System().devices.device_names
	
# Environment variables
	$env:pythonpath = 'C:\Program Files\Swabian Instruments\Time Tagger\driver\python'
	$env:timetagger_install_path = 'C:\Program Files\Swabian Instruments\Time Tagger\'
