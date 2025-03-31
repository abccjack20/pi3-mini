import os
from io import StringIO, BytesIO
import struct
import zipfile 
import xml.etree.ElementTree as ET
from copy import copy, deepcopy
import time
import datetime 
import numpy as np
from threading import Thread
import threading
# _____________________________________________________________________________
# some utility class:

class ZipStreamer(object):
    # copied from Pedro Werneck and dm2013 in stackoverflow
    # https://stackoverflow.com/questions/10405210/create-and-stream-a-large-archive-without-storing-it-in-memory-or-on-disk
    # modified a little bit

    def __init__(self):
        self.outStream = BytesIO()

        # write to the stringIO with no compression
        self.zipFile = zipfile.ZipFile(self.outStream, 'w', zipfile.ZIP_STORED)

        self.current_file = None

        self._last_streamed = 0

    def put_file(self, name, date_time=None):
        if date_time is None:
            date_time = time.localtime(time.time())[:6]

        zinfo = zipfile.ZipInfo(name, date_time)
        zinfo.compress_type = zipfile.ZIP_STORED
        zinfo.flag_bits = 0x08
        zinfo.external_attr = 0o600 << 16
        zinfo.header_offset = self.outStream.pos

        # write right values later
        zinfo.CRC = 0
        zinfo.file_size = 0
        zinfo.compress_size = 0

        self.zipFile._writecheck(zinfo)

        # write header to mega_streamer
        self.outStream.write(zinfo.FileHeader())

        self.current_file = zinfo

    def flush(self):
        zinfo = self.current_file
        self.outStream.write(
            struct.pack("<LLL", zinfo.CRC, zinfo.compress_size,
                        zinfo.file_size))
        self.zipFile.filelist.append(zinfo)
        self.zipFile.NameToInfo[zinfo.filename] = zinfo
        self.current_file = None

    def write(self, bytes):
        self.outStream.write(bytes)
        self.outStream.flush()
        zinfo = self.current_file
        # update these...
        zinfo.CRC = zipfile.crc32(bytes, zinfo.CRC) & 0xffffffff
        zinfo.file_size += len(bytes)
        zinfo.compress_size += len(bytes)

    def read(self, n=-1):
        pos = self._last_streamed
        self.outStream.seek(pos)
        bytes = self.outStream.read(n)
        self._last_streamed = pos + len(bytes)

        # # cleaning up memory in each iteration
        # self.outStream.seek(pos) 
        # self.outStream.truncate()
        # self.outStream.flush()

        return bytes
    def seek(self, byte):
        self._last_streamed = byte

    def close(self):
        self.zipFile.close()


# _____________________________________________________________________________
# PULSES:

class Pulse(object):
    
    def __init__(self, duration, amp=1.0, marker1=0b00):
        self.duration = int(duration)
        self.amp = amp
        self.marker = marker1 # 2 bit binary indicating (marker2)(marker1), e.g. 10 means marker 2 ON marker 1 OFF
    #    print(self.marker)
    # _________________
    # Operators on func
    
    def __add__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x : f(x) + pulse.func(x)
        return new
        
    def __iadd__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x : g(x) + f(x)
        return self
        
    def __sub__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x : f(x) - pulse.func(x)
        return new
        
    def __isub__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x : g(x) - f(x)
        return self
        
    def __mul__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x : f(x) * pulse.func(x)
        return new
        
    def __imul__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x : g(x) * f(x)
        return self
        
    def __div__(self, pulse):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x : f(x) / pulse.func(x)
        return new
        
    def __idiv__(self, pulse):
        f = pulse.func
        g = self.func
        self.func = lambda x : g(x) / f(x)
        return self
        
    def __neg__(self):
        new = deepcopy(self)
        f = new.func
        new.func = lambda x : -1 * f(x)
        return new
        
    # _____________________
    # Operators on duration
        
    def __lt__(self, pulse):
        """ Compare durations."""
        return self.duration < pulse.duration
        
    def __le__(self, pulse):
        """ Compare durations."""
        return self.duration <= pulse.duration
        
    def __gt__(self, pulse):
        """ Compare durations."""
        return self.duration > pulse.duration
        
    def __ge__(self, pulse):
        """ Compare durations."""
        return self.duration >= pulse.duration
        
    def __mod__(self, pulse):
        """ Compare durations."""
        return self.duration % pulse.duration
        
    # _________
    # Function
    
    def compile(self, t_0, datatype=np.single):
        """ This will be called by Waveform when compiling. """
        samples = np.arange(t_0, t_0 + self.duration, dtype=datatype) # use 8 single precision for compilation
        samples = self.func(samples)
        samples = self.norm(samples)
        return samples
        
    def func(self, samples):
        """ Override this to manipulate samples. """
        return samples
        
    def before(self, f):
        """ Make a composed function f(g(t)). """
        g = self.func # prevent recursion
        self.func = lambda t : f(g(t))
        
    def after(self, f):
        """ Make a composed function g(f(t)). """
        g = self.func # prevent recursion
        self.func = lambda t : g(f(t))
        
    def norm(self, samples):
        if len(samples) == 0: return samples
        max_sample = max(abs(samples))
        if max_sample > 1.0:
            samples = self.amp/max_sample * samples
        return samples
        
class Idle(Pulse):
    def __init__(self, duration, marker=0b00):
        Pulse.__init__(self, duration, marker1=marker)

    def func(self, samples):
        samples = 0.0 * samples
        return samples
        
class DC(Pulse):
    
    def func(self, samples):
        samples = 0.0 * samples + 1.0
        return samples
        
class Ramp_old(Pulse):
    
    def __init__(self, duration, start, stop, marker=0b00):
        Pulse.__init__(self, duration, marker1=marker)
        self.start = start
        self.slope = (1.0 * stop - start) / duration
        
    def func(self, samples):
        samples = self.start + self.slope * (samples - samples[0])
        return samples

class Ramp(Pulse):
    
    def __init__(self, duration, y0, y1, dx, lshift=0, offset=0, vshift=0, marker=0b00):
        Pulse.__init__(self, duration, marker1=marker)
        self.start = y0
        self.slope = 1.0*(y1 - y0) / dx
        if lshift > 0:
            self.lshift = lshift
        else:
            self.lshift = offset  # Trying to be compatible with the older version
        self.vshift = vshift

    def func(self, samples):
        samples = self.start + self.slope * (samples + self.lshift - samples[0])
        return np.clip(samples, 0.0, 1.0) + self.vshift

    
class Sin(Pulse):
    
    def __init__(self, duration, freq, phase=0.0, amp=1.0, lshift=0, vshift=0, marker=0b00):
        Pulse.__init__(self, duration, amp, marker1=marker)
        self.freq = freq
        self.phase = phase
        self.lshift = lshift
        self.vshift = vshift
        
    def func(self, samples):
        output = self.vshift + self.amp * np.sin(
            2*np.pi*self.freq*(samples + self.lshift) + self.phase
        )
        return output


class Saw(Pulse):

    def __init__(self, duration, period, amp=1.0, lshift=0, tilt=0.5, vshift=0, marker=0b00):
        Pulse.__init__(self, duration, amp, marker1=marker)
        self.period = period
        self.lshift = lshift
        self.vshift = vshift
        self.amp = amp
        self.tilt = tilt
    
    def H(self, x):
        return .5*(np.sign(x) + 1)

    def saw(self, x, t=0.5):
        x_p = np.mod(x, 1.)
        left = np.ones(x_p.size)
        left[x_p > t] *= 0
        right = 1 - left
        incline = (x_p/t)
        decline = (1 - x_p)/(1 - t)
        return left*incline + right*decline

    def func(self, samples):
        T = self.period
        x = samples + self.lshift
        Amp = self.amp
        
        return Amp*self.saw(x/T, t=self.tilt) + self.vshift


class SinDecay(Pulse):
    
    def __init__(self, duration, freq, tt, phase=0.0, amp=1.0, marker=0b00):
        Pulse.__init__(self, duration, amp, marker1=marker)
        self.freq = freq
        self.phase = phase
        self.tt = tt
        
    def func(self, samples):
        samples = self.amp * np.sin(2 * np.pi * self.freq * samples 
                                      + self.phase)
#       * np.exp(- samples / self.tt)
        return samples

class Cos(Sin):
    
    def func(self, samples):
        samples = self.amp * np.cos(2 * np.pi * self.freq * samples 
                                      + self.phase)
        return samples
    
class Gauss(Pulse):
    
    def __init__(self, duration, peak=None, sigma=None, amp=1.0, marker=0b00):
        Pulse.__init__(self, duration, amp, marker)
        if peak is None:
            self.peak = int(duration / 2)
        else:
            self.peak = peak
        if sigma is None:
            self.sigma = int(duration / 4)
        else:
            self.sigma = sigma
        
    def func(self, samples):
        samples -= samples[0] # make gauss pulse time-invariant
        samples = self.amp * np.exp( -( (samples-(self.peak))
                                        /2 / self.sigma) **2 )
        return samples

class Envelop(Pulse):
    
    def __init__(self, duration, amp_li, freq_li, phase_li, vshift=0.0, lshift=0.0, marker=0b00):
        Pulse.__init__(self, duration, marker1=marker)
        self.amp_li = amp_li
        self.freq_li = freq_li
        self.phase_li = phase_li
        self.lshift = lshift
        self.vshift = vshift
        
    def fourier(self, segment):
        ft = np.zeros(len(segment))
        Amp = self.amp_li[:,None]
        Freq = self.freq_li[:,None]*2*np.pi
        Phase = self.phase_li[:,None]*2*np.pi
        #ft = np.sum(Amp*np.cos(Freq*segment + self.lshift + Phase), axis=0)
        ft = np.sum(Amp*np.cos(Freq*(segment + self.lshift) + Phase), axis=0)
        return ft + self.vshift
    
    def func(self, samples):
        ft = self.fourier(samples)
        return ft
      
# _____________________________________________________________________________
# WAVEFORMS:
  
class WaveformX(BytesIO):

    def __init__(self, name, pulseSeq, IQenable=False, markerEnable=False, pointOffSet=0, samplingRate=25.0e9):
        super(WaveformX, self).__init__()
        if os.path.splitext(name)[-1] != ".wfmx":
            name = name + ".wfmx"
        self.name = name
        self.pulseSeq = deepcopy(pulseSeq)
        self.IQenable = IQenable
        self.markerEnable = markerEnable
        self.pointOffSet = pointOffSet
        self.samplingRate = samplingRate

        self.assetType = "Waveform"
        self.totalPulseLength = sum([int(p.duration) for p in self.pulseSeq])

        self.state = "compiling"
        self.compileWFMX()
        self.state = "ready"

    def writeHeader(self):
        # create XML header
        # require structure for WFMX header
        self.dataFile = ET.Element('DataFile')
        self.dataFile.attrib = {'offset': '000000000', 'version': '0.2'}
        dsc = ET.SubElement(self.dataFile, 'DataSetsCollection')
        # dsc.set("xmlns", "http://www.tektronix.com")
        # dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        # dsc.set("xsi:schemaLocation", (r"http://www.tektronix.com file:///" +
        #                                r"C:\Program%20Files\Tektronix\AWG70000" +
        #                                r"\AWG\Schemas\awgDataSets.xsd"))
        self.dataFile_dataSet = ET.SubElement(dsc, 'DataSets')
        self.dataFile_dataSet.attrib = {'version': '1'} # {'version': '1', "xmlns":"http://www.tektronix.com"}
        self.dataFile_dataDescrip = ET.SubElement(self.dataFile_dataSet, 'DataDescription')
        self.numberSamples = ET.SubElement(self.dataFile_dataDescrip, 'NumberSamples')
        self.numberSamples.text = str(self.totalPulseLength)
        ET.SubElement(self.dataFile, "Setup")

        self.addOptionalHeader()
        self.indent(self.dataFile)
        self.updateDataFileSize()
        self.write(ET.tostring(self.dataFile))
    def addOptionalHeader(self):
        # optional parts of WFMX header

        # data description part
        ET.SubElement(self.dataFile_dataDescrip, 'SamplesType').text = "AWGWaveformSample"
        mi = ET.SubElement(self.dataFile_dataDescrip, 'MarkersIncluded') # optional, default 'false'
        mi.text = "false"
        if self.markerEnable:
            mi.text = 'true' # has marker vector
        ET.SubElement(self.dataFile_dataDescrip, 'NumberFormat').text = "Single"
        ET.SubElement(self.dataFile_dataDescrip, 'Endian').text = "Little"
        ts = ET.SubElement(self.dataFile_dataDescrip, 'Timestamp')
        ts.text = datetime.datetime.now().isoformat()

        # product specific part
        ps = ET.SubElement(self.dataFile_dataSet, 'ProductSpecific') # optional 
        ps.attrib = {'name':'AWG70002A'} #specify the product name
        rsr = ET.SubElement(ps, 'RecSamplingRate') 
        rsr.attrib = {'units': 'Hz'}
        rsr.text = str(self.samplingRate)
        # ra = ET.SubElement(ps, 'RecAmplitude') 
        # ra.attrib = {'units': 'Volts'}
        # ra.text = '0.5'
        # ro = ET.SubElement(ps, 'RecOffset') 
        # ro.attrib = {'units': 'Volts'}
        # ro.text = '0'
        # recfreq = ET.SubElement(ps, 'RecFrequency') 
        # recfreq.attrib = {'units': 'Hz'}
        # recfreq.text = 'NaN'
        # sn = ET.SubElement(ps, 'SerialNumber')
        # sn.text = 'B020245'
        # swv = ET.SubElement(ps, 'SoftwareVersion') 
        # swv.text = "5.3.0128.0"
        # ET.SubElement(ps, 'UserNotes') 
        # obd = ET.SubElement(ps, 'OriginalBitDepth') 
        # obd.text = "Floating"
        # ET.SubElement(ps, '<Thumbnail ') 
        if self.IQenable:
            sf = ET.SubElement(ps, 'SignalFormat') # optional, default 'Real'
            sf.text = 'IQ' # 'IQ' for complex
    def indent(self, elem, level=0):
        # copy from Erick M. Sprengel in stackoverflow
        # https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
        i = '\n' + '  '*level
        if len(elem): 
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def updateDataFileSize(self):
        actual_size = len(ET.tostring(self.dataFile))
        if actual_size > 999999999:
            raise Exception("sequence files cannot exceed 999,999,999 bytes")
        self.dataFile.attrib['offset'] = '{:09d}'.format(actual_size)

    def generateWaveVector(self):

        wave = np.zeros(self.totalPulseLength, dtype=np.single) # use single precision for compilation
        marker_seq = np.zeros(self.totalPulseLength, dtype=np.int8) # unsigned 8-bit integer marker vector (optional)
        t_0 = self.pointOffSet
        i = 0
        for pulse in self.pulseSeq:
            pulse.duration = int(pulse.duration)
            wave[i:i+pulse.duration] = pulse.compile(t_0, datatype=np.single)
            if self.markerEnable:
                marker_seq[i:i+pulse.duration] = [pulse.marker,] * pulse.duration
            t_0 += pulse.duration #1/1 local phase
            i += pulse.duration
        # store data
        self.wave = wave
        self.markerSeq = marker_seq

    def writeWaveVector(self):

        #write into datafile
        self.seek(self.getvalue().find(b'</DataFile>')+len(b'</DataFile>'))
        self.write(memoryview(self.wave))
        if self.markerEnable:
            self.write(memoryview(self.markerSeq))

    def compileWFMX(self):
        self.writeHeader()
        self.generateWaveVector()
        self.writeWaveVector()

class SequenceDescriptor(BytesIO):
    DEFAULT_ENTRY = { 'wave' : [None, None],
                      'Repeat' : "Once",
                      'WaitInput' : "None",
                      'EventJumpInput' : "None",
                      'EventJumpTo': "Next",
                      'GoTo' : "Next",
                    }
    def __init__(self, name):
        super(SequenceDescriptor, self).__init__()
        nameSplit = os.path.splitext(name)
        if nameSplit[-1] != ".sml":
            name = nameSplit[0] + ".sml"
        self.name = name
        self.assetType = "Sequence"

        self.stepPointerPosition = 0

        self.writeHeader()

        self.state = "compiling"

    def writeHeader(self):
        # sml file
        self.dataFile = ET.Element('DataFile')
        self.dataFile.attrib = {'offset': '000000000', 'version': '0.1'}
        dsc = ET.SubElement(self.dataFile, 'DataSetsCollection')
        # dsc.set("xmlns", "http://www.tektronix.com")
        # dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        # dsc.set("xsi:schemaLocation", (r"http://www.tektronix.com file:///" +
        #                                r"C:\Program%20Files\Tektronix\AWG70000" +
        #                                r"\AWG\Schemas\awgSeqDataSets.xsd"))
        ET.SubElement(self.dataFile, "Setup")
        self.dataFile_dataSet = ET.SubElement(dsc, 'DataSets')
        self.dataFile_dataSet.attrib = {'version': '1'}
        self.dataFile_dataDescrip = ET.SubElement(self.dataFile_dataSet, 'DataDescription')
        ET.SubElement(self.dataFile_dataSet, "ProductSpecific").attrib = {"name": ""}
        ET.SubElement(self.dataFile_dataDescrip, 'SequenceName').text = os.path.splitext(self.name)[0]
        ET.SubElement(self.dataFile_dataDescrip, 'Timestamp').text = datetime.datetime.now().isoformat()
        ET.SubElement(self.dataFile_dataDescrip, 'JumpTiming').text = 'JumpImmed'  # What does this mean?
        ET.SubElement(self.dataFile_dataDescrip, 'RecSampleRate').text = "NaN" # '10000000000'
        ET.SubElement(self.dataFile_dataDescrip, "RepeatFlag").text = "false"
        ET.SubElement(self.dataFile_dataDescrip, "PatternJumpTable").attrib = {"Enabled": "false", "Count":"65536"} # the 65536 of count is default, I dont know why
        self.dataFile_steps = ET.SubElement(self.dataFile_dataDescrip, 'Steps')
        self.dataFile_steps.attrib = {'StepCount': '0', 'TrackCount': '2'}

        # self.updateDataFileSize()
        # self.seek(0)
        # self.write(ET.tostring(self.dataFile))

    def writeStep(self, *assetNameType, **kw):
        '''
        *assetNameType: 
                (name, type), eg. ("RABI", "Waveform")
                names and types of waveform/sequence on track 1 or/and track 2

        **kw : 
                wave : [waveObject, waveObject],
                Repeat : "Once", "Infinite", repeat number (int), 
                WaitInput : "None", "TrigA", "TrigB", "Internal"
                EventJumpInput : "None", "TrigA", "TrigB", "Internal"
                EventJumpTo: "First", "Next", "End", "Last", step number(int)
                GoTo : "First", "Next", "End", "Last", step number(int)

                compileAgain: False/True
        '''
        compileAgain = False # compile the element tree into bytes string?
        entry = deepcopy(self.DEFAULT_ENTRY)
        for key in kw:
            if key == 'compileAgain':
                compileAgain = kw[key]
            else:
                entry[key] = kw[key]

        stepNumber = len(self.dataFile_steps) + 1
        self.dataFile_steps.attrib['StepCount'] = str(stepNumber)
        stepElement = ET.SubElement(self.dataFile_steps, 'Step')
        ET.SubElement(stepElement, 'StepNumber').text = str(stepNumber)

        if type(entry['Repeat']) == int:
            ET.SubElement(stepElement, 'Repeat').text = "RepeatCount" #repeat several times
            ET.SubElement(stepElement, 'RepeatCount').text = str(entry['Repeat']) #repeatnumber if Repeat = 'RepeatCount'
        elif type(entry['Repeat']) == str:
            ET.SubElement(stepElement, 'Repeat').text = entry['Repeat'] #Once, Infinite
            ET.SubElement(stepElement, 'RepeatCount').text = '1' #repeat number if Repeat = 'RepeatCount'

        ET.SubElement(stepElement, 'WaitInput').text = entry['WaitInput']
        ET.SubElement(stepElement, 'EventJumpInput').text = entry['EventJumpInput']

        if entry['EventJumpInput'] != "None":
            if type(entry['EventJumpTo']) == int:
                ET.SubElement(stepElement, 'EventJumpTo').text = 'StepIndex' # First, Next, End, StepIndex
                ET.SubElement(stepElement, 'EventJumpToStep').text = str(entry['EventJumpTo']) # step number if GoTo = StepIndex, 1 otherwise
            elif type(entry['EventJumpTo']) == str:
                ET.SubElement(stepElement, 'EventJumpTo').text = entry['EventJumpTo'] # First, Next, End, StepIndex
                ET.SubElement(stepElement, 'EventJumpToStep').text = '1' # step number if GoTo = StepIndex, 1 otherwise

        if type(entry['GoTo']) == int:
            ET.SubElement(stepElement, 'GoTo').text = 'StepIndex' # First, Next, End, StepIndex
            ET.SubElement(stepElement, 'GoToStep').text = str(entry['GoTo']) # step number if GoTo = StepIndex, 1 otherwise
        elif type(entry['GoTo']) == str:
            ET.SubElement(stepElement, 'GoTo').text = entry['GoTo'] # First, Next, End, StepIndex
            ET.SubElement(stepElement, 'GoToStep').text = '1' # step number if GoTo = StepIndex, 1 otherwise

        # max 2 elements in assetNameType list, for track 1, track 2
        if len(assetNameType) > 2:
            raise str("More than 2 assets in a single step in the sequence.\nOnly first 2 assets were added")
            assetNameType = assetNameType[:2]
        elif len(assetNameType) == 1:
            assetNameType = list(assetNameType) + [("","None")]
        elif len(assetNameType) == 0: 
            assetNameType = [("","None"), ("","None")]
        else:
            pass
        assets = ET.SubElement(stepElement, 'Assets')
        for NameType in assetNameType:
            assetName, assetType = NameType
            if assetType not in ["Waveform", "Sequence", "None"]:
                raise ValueError(1)
                break
            a = ET.SubElement(assets, 'Asset')
            ET.SubElement(a, 'AssetName').text = os.path.splitext(assetName)[0] # name of WFMX file w/o extention
            ET.SubElement(a, 'AssetType').text = str(assetType) # "Waveform" or "Sequence"
        
        # add flags, this part may amend later
        flags = ET.SubElement(stepElement, 'Flags')
        for iii in range(2):
            flagset = ET.SubElement(flags, 'FlagSet')
            for ABCD in ["A", "B", "C", "D"]:
                flagflag = ET.SubElement(flagset, "Flag")
                flagflag.attrib = {"name": ABCD}
                flagflag.text = "NoChange"

        # write into bytes string
        if compileAgain:
            self.compileSML()

    def deleteStep(self, stepNumber, compileAgain=False):
        self.dataFile_steps.remove(self.dataFile_steps[stepNumber-1])
        for step in self.dataFile_steps[stepNumber-1:]:
            for stepNumTag in step.findall("StepNumber"):
                stepNumTag.text = str(int(stepNumTag.text) - 1)


        self.dataFile_steps.attrib['StepCount'] = str(len(self.dataFile_steps))
       
        if compileAgain:
            # write into bytes string
            self.compileSML()

    def compileSML(self):
        self.indent(self.dataFile)
        self.updateDataFileSize()
        # clear memory first
        self.seek(0)
        self.truncate()
        self.flush()
        self.write(ET.tostring(self.dataFile))
        self.state = "ready"

    def updateDataFileSize(self):
        actual_size = len(ET.tostring(self.dataFile))
        if actual_size > 999999999:
            raise Exception("sequence files cannot exceed 999,999,999 bytes")
        self.dataFile.attrib['offset'] = '{:09d}'.format(actual_size)

    def indent(self, elem, level=0):
        # copy from Erick M. Sprengel in stackoverflow
        # https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
        i = '\n' + '  '*level
        if len(elem): 
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

class SequenceX(ZipStreamer):
    def __init__(self, name):
        super(SequenceX, self).__init__()
        if os.path.splitext(name)[-1] != ".seqx":
            name = name + ".seqx"
        self.name = name

        self.wavBox = {}
        self.seqBox = {}

        self.addSequence(self.name) # add to seqBox since itself is a sequence which needs description
        self.mainSeqName = self.name # this is the main sequence which contains subsequences and waveforms
        
        self.writeSetupFile()
        self.state = "compiling"

    def addWaveform(self, *arg, **kw):
        # add waveform to the wavBox which is zipped into a zipstream later by zipAll() function
        # input:
        #     *arg:
        #         name (str)
        #         list of pulses (list)
        #     **kw:
        #         IQenable=True/False, 
        #         markerEnable=True/False, 
        #         pointOffSet= (int)
        #         samplingRate= number e.g 2.5e10

        waveform = WaveformX(*arg, **kw)
        waveformName = os.path.splitext(waveform.name)[0]
        self.wavBox[waveformName] = waveform
    def addSequence(self, *arg, **kw):
        # add sequence to the seqBox which is zipped into a zipstream later by zipAll() function
        # input:
        #     *arg:
        #         name (str)

        sequence = SequenceDescriptor(*arg, **kw)
        sequenceName = os.path.splitext(sequence.name)[0]
        self.seqBox[sequenceName] = sequence

    def addWaveformThreading(self, *arg, **kw):
        # TO DO: multithreading compiling 
        # unfinished function
        t = Thread(target = self.addWaveform, args=arg, kwargs=kw, name="createWav")
        t.start()

    def addSequenceThreading(self, *arg, **kw):
        # TO DO: multithreading compiling 
        # unfinished function
        t = Thread(target = self.addSequence, args=arg, kwargs=kw, name="createSeq")
        t.start()
    def writeSetupFile(self):
        # write a setup file to indicate the main sequence
        self.setupFile = ET.Element('RSAPersist')
        self.setupFile.attrib = {'version':'0.1'}
        ET.SubElement(self.setupFile, 'Application').text = 'Pascal'
        ET.SubElement(self.setupFile, 'MainSequence').text = os.path.splitext(self.mainSeqName)[0]

        productSpec = ET.SubElement(self.setupFile, 'ProductSpecific')
        productSpec.set('name', "AWG70002A")
        ET.SubElement(productSpec, 'SerialNumber').text = 'B020245' # need to check later, B020245, B020397
        ET.SubElement(productSpec, 'SoftwareVersion').text = '5.3.0128.0' # need to check later    
        ET.SubElement(productSpec, 'CreatorProperties').set('name', '')
        
    def zipAll(self):
        # for t in threading.enumerate():
        #     if t.name == "MainThread":
        #         continue
        #     t.join()
        self.zipFile.writestr('setup.xml', ET.tostring(self.setupFile))
        for key in self.wavBox:
            wav = self.wavBox[key]
            self.zipFile.writestr('Waveforms/%s'%(wav.name), wav.getvalue())
            wav.close()
            del wav
        for key in self.seqBox:
            seq = self.seqBox[key]
            self.zipFile.writestr('Sequences/%s'%(seq.name), seq.getvalue())
            seq.close()
            del seq
        self.zipFile.close() #this step is crucial, dont miss
        self.outStream.seek(0)
        self.state = "ready"

if __name__ == "__main__":

    #for code debugging and testing only

    # compile the sequence==============================================================
    tau = np.arange(0, 1.0e4, 1.0e3)
    freq = 0.01
    # create the sequence structure
    seqx = SequenceX("testSEQX")
    # write waveforms and sequences 
    for t in tau:
        if t < 2400:
            complement = 2400 - t
            name = "wave%i"%t
            assetType = "Waveform"
            # seqx.addWaveformThreading(name, [Idle(complement), Sin(t, freq)])
            seqx.addWaveform(name, [Idle(complement), Sin(t, freq)])
            seqx.seqBox["testSEQX"].writeStep((name, assetType), WaitInput="TrigA", Repeat="Once", EventJumpInput="None",EventJumpTo="Next", GoTo="Next")
        else:
            name = "wave%i"%t
            assetType = "Waveform"
            # seqx.addWaveformThreading("wave%i"%t, [Sin(t, freq)])
            seqx.addWaveform(name, [Sin(t, freq)])
            seqx.seqBox["testSEQX"].writeStep((name, assetType), WaitInput="TrigA", Repeat="Once", EventJumpInput="None",EventJumpTo="Next", GoTo="Next")
    # write header description to sequence(s)
    seqx.seqBox["testSEQX"].compileSML()

    # pack every waveform and sequence into a *.seqx memory
    seqx.zipAll()
    print("Finished Compiling SequenceX")

    # send to the AWG via FTP==============================================================
    from ftplib import FTP
    import socket
    ftp_server = '192.168.1.58'
    ftp_port = 4000
    ftp_user = 'ftpawg'
    ftp_pwd = 'ftpawg'
    ftp_dir = '/awg'
    timeout = 10
    # socket.setdefaulttimeout(60)
    ftp_main = FTP(ftp_server)
    # ftp_main.connect(ftp_server, ftp_port, timeout)
    ftp_main.login(ftp_user, ftp_pwd)
    # ftp_main.cwd(ftp_dir)
    print('FTP Connected')
    ftp_main.storbinary('STOR %s'%os.path.basename(seqx.name), seqx.outStream)
    seqx.close() #this step is crucial, dont miss
    ftp_main.quit()
    print("File sent via FTP")

    # send  command to the awg==============================================================
    import visa
    rm = visa.ResourceManager()
    awgcom = rm.open_resource('TCPIP0::192.168.1.58::INSTR')
    awgcom.timeout = 5.0
    def ask(query):
        """Send a query string to AWG and return the response."""
        awgcom.write(query)
        try:
            res = awgcom.read()
        except visa.VisaIOError as e:
            res = ''
            if 'Timeout' in e.message:
                res = 'No response from AWG for: "' + query + '"'
            else:
                raise e
        if "\n" in res:
            res = res[0:-1]
        return res
    def tell(command):
        """Send a command string to the AWG."""
        awgcom.write(command)
    def tellOverlapCommand(command, max_try=10):
        for i in range(max_try):
            try:
                if int(ask("*OPC?")) == 1:
                    tell(command)
                    break
                else:
                    time.sleep(1)
            except:
                print("AWG not Ready, Sending again")
                time.sleep(1)
    #load the seq ==============================================================
    command = 'SLISt:SEQuence:DELete ALL'
    print("Sending Command: %s"%command)
    tellOverlapCommand(command) 

    command = 'MMEMory:OPEN:SASSet:SEQuence "C:/%s/%s"' % ('/awg', seqx.name)
    print("Sending Command: %s"%command)
    tellOverlapCommand(command) 

    # command = 'MMEMory:OPEN "C:/%s/%s"' % ('/awg', seqx.name)
    # print("Sending Command: %s"%command)
    # tellOverlapCommand(command) 

    namename = os.path.splitext(seqx.name)[0]
    command = 'SOURCE1:CASSET:SEQUENCE "%s", 1'%namename
    print("Sending Command: %s"%command)
    tellOverlapCommand(command) 

    print("All Commands Sent")
    time.sleep(0)
    print("Current Seqeunce: %s"%(ask("SOURCE1:CASSET?")))
    # DONE