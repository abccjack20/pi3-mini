import numpy as np
import pickle as cPickle

# enthought library imports
from traits.api import HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, String, Str, Enum, Button, Tuple, List, on_trait_change,\
                                 cached_property, DelegatesTo
from traitsui.api import View, Item, UItem, HGroup, VGroup, Tabbed,\
    EnumEditor, TextEditor, CheckListEditor, ArrayEditor, CSVListEditor

from traitsui.file_dialog import save_file
from traitsui.menu import Action, Menu, MenuBar

from enable.api import ComponentEditor, Component
from chaco.api import Plot, ScatterPlot, CMapImagePlot, ArrayPlotData,\
    Spectral, ColorBar, LinearMapper, DataView,\
    LinePlot, ArrayDataSource, HPlotContainer,hot
#from chaco.tools.api import ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool

from tools.utility import GetSetItemsHandler, GetSetItemsMixin

import threading
import time
import logging

from tools.emod import FreeJob
from hardware.api import Microwave, PulseGenerator
from measurements.photon_time_trace import PhotonTimeTrace

mw = Microwave()
pg = PulseGenerator()

class MicrowavePanel(FreeJob):

    power = Float(
        value=mw.getPower(),
        desc='microwave power', label='Power [dBm]',
        mode='text', auto_set=False, enter_set=True
    )
    power_input = Range(
        low= -100., high=15., value=mw.getPower(),
        label='Set value [dBm]', desc='targeted microwave power',
        mode='text', auto_set=False, enter_set=True
    )
    power_sbutton = Button(label='Set')

    frequency = Float(
        value=mw.getFrequency(),
        label='frequency [Hz]', desc='microwave frequency',
        # mode='text', auto_set=False, enter_set=True,
        editor=TextEditor(auto_set=False, enter_set=True, evaluate=float, format_str='%e')
    )
    frequency_input = Range(
        low=1, high=20e9, value=mw.getFrequency(),
        label='Set value [Hz]', desc='targeted microwave frequency',
        mode='text', auto_set=False, enter_set=True
    )
    frequency_sbutton = Button(label='Set')

    on_status = Bool(False, label='on/off', desc='whether the output is on/off')
    toggle_onoff_button = Button(label='Power on/off')

    visa_address = Str(mw.visa_address, label='VISA address')
    RefreshRate = Range(
        low=0.1, high=100., value=0.5,
        desc='Refresh rate [s]', label='Refresh rate [s]',
        mode='text', auto_set=False, enter_set=True
    )
    keep_update = Bool(False, label='Update', desc='whether to keep updating the MW status')

    def __init__(self):
        super().__init__()

    @on_trait_change('keep_update')
    def start_monitor(self):
        if self.keep_update:
            self.start()
        else:
            self.stop()

    def _run(self):
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            
            self.on_status = bool(mw.onStatus())
            self.power = mw.getPower()
            self.frequency = mw.getFrequency()
            self.visa_address = mw.visa_address

    def _power_sbutton_fired(self):
        mw.setPower(self.power_input, output=False)
    
    def _frequency_sbutton_fired(self):
        mw.setFrequency(self.frequency_input)

    def _toggle_onoff_button_fired(self):
        if self.on_status:
            mw.setPower(None)
        else:
            mw.setPower(self.power)

    traits_view = View(
        VGroup(
            HGroup(
                Item('visa_address', style='readonly', width= -100),
                Item('RefreshRate', width= -60),
                Item('keep_update'),
            ),
            HGroup(
                Item('power', style='readonly', width= -100),
                Item('power_input', width= -100),
                UItem('power_sbutton'),
                Item('on_status', width= -100, enabled_when='0 > 1'),
                UItem('toggle_onoff_button'),
            ),
            HGroup(
                Item('frequency', style='readonly', width= -100),
                Item('frequency_input', width= -100),
                UItem('frequency_sbutton'),
            ),
            show_border=True,
            label='Microwave control',
        ),
    )

PG_CH_KEYS = list(pg.channel_map.keys())
PG_CH_VALUES = list(pg.channel_map.values())

class PulseGeneratorPanel(FreeJob):

    ip_address = Str(pg.ip, label='IP address')
    RefreshRate = Range(
        low=0.1, high=100., value=0.5,
        desc='Refresh rate [s]', label='Refresh rate [s]',
        mode='text', auto_set=False, enter_set=True,
    )
    keep_update = Bool(False, label='Update', desc='whether to keep updating the MW status')

    ch_high = List(
        value=pg.ch_high,
        label='Current ch high',
        editor=CheckListEditor(
            values=PG_CH_KEYS,
            cols=len(PG_CH_KEYS),
        )
    )
    ch_high_input = List(
        value=pg.ch_high,
        label='Current ch high',
        editor=CheckListEditor(
            values=PG_CH_KEYS,
            cols=len(PG_CH_KEYS),
        )
    )
    ch_high_sbutton = Button(label='Set')
    
    light_button = Button(label='Light')
    night_button = Button(label='Night')

    def __init__(self):
        super().__init__()
        self.start()

    def _run(self):
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            self.ch_high = pg.ch_high

    @on_trait_change('keep_update')
    def start_monitor(self):
        if self.keep_update:
            self.start()
        else:
            self.stop()

    def _ch_high_sbutton_fired(self):
        pg.Continuous(self.ch_high_input)

    def _light_button_fired(self):
        pg.Light()

    def _night_button_fired(self):
        pg.Night()  

    traits_view = View(
        VGroup(
            HGroup(
                Item('ip_address', style='readonly'),
                Item('RefreshRate', width= -60),
                Item('keep_update'),
            ),
            HGroup(
                VGroup(
                    HGroup(
                        UItem('ch_high', style='custom', enabled_when='0 > 1'),
                        UItem('ch_high_sbutton'),
                        UItem('light_button'),
                        UItem('night_button'),
                    ),
                    UItem('ch_high_input', style='custom'),
                ),
                springy=True,
                show_border=True,
                label='Constant output',
            ),
            show_border=True,
            label='PulseStreamer control',
        )
    )


class ControlPanel(HasTraits):

    mw_panel = Instance(MicrowavePanel, factory=MicrowavePanel, label='Microwave control')
    pg_panel = Instance(PulseGeneratorPanel, factory=PulseGeneratorPanel, label='PulseStreamer control')
    ptt_panel = Instance(PhotonTimeTrace, factory=PhotonTimeTrace, label='Photon time trace')

    def __init__(self):
        self.ptt_panel.c_enable1 = True
        self.ptt_panel.c_enable2 = True
        self.ptt_panel.sum_enable = True
        self.ptt_panel.digi_channel = 'cha0+1'
        pg.Night()
    
    traits_view = View(
        Tabbed(
            UItem('ptt_panel', style='custom'),
            VGroup(
                UItem('mw_panel', style='custom'),
                UItem('pg_panel', style='custom'),
                label='Hardware control',
            ),
        ),
        width=800, height=600,
        resizable=True,
    )
