# To install
	conda env create -f .\CondaEnvironment\environment.yml

# To run
	ipython --gui=qt -i diamond.py

# Check NIDAQ card status
	import nidaqmx as ni
	ni.system.System().devices.device_names
	
