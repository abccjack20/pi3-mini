import numpy
import pickle as cPickle

# enthought library imports
from traits.api import HasTraits, Trait, Instance, Property, Int, Float, Range,\
                                 Bool, Array, String, Str, Enum, Button, Tuple, List, on_trait_change,\
                                 cached_property, DelegatesTo
from traitsui.api import View, Item, Group, HGroup, VGroup, Tabbed, EnumEditor, UI

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

from hardware.api import time_tagger as tt

from tools.emod import FreeJob

class StartThreadHandler( GetSetItemsHandler ):

    def init(self, info):
        info.object.start()
        return True
        
class PhotonTimeTrace( FreeJob, GetSetItemsMixin ):

    TraceLength = Range(low=10, high=10000, value=100, desc='Length of Count Trace', label='Trace Length')
    SecondsPerPoint = Range(low=0.00001, high=1, value=0.1, desc='Seconds per point [s]', label='Seconds per point [s]')
    RefreshRate = Range(low=0.01, high=1, value=0.1, desc='Refresh rate [s]', label='Refresh rate [s]')
    digi_channel = Enum('cha0','cha1','cha2','cha3','cha4','cha5', 'cha6','cha7','cha0+1', desc='Digi channel', label='Digi channel')

    # trace data
    C0 = Array()
    C1 = Array()
    C2 = Array()
    C3 = Array()
    C4 = Array()
    C5 = Array()
    C6 = Array()
    C7 = Array()
    C0C1 = Array()
    T = Array()
    
    c_enable1  = Bool(True,  label='channel 1', desc='enable channel 1'  )
    c_enable2  = Bool(False, label='channel 2', desc='enable channel 2'  )
    c_enable3  = Bool(False, label='channel 3', desc='enable channel 3'  )
    c_enable4  = Bool(False, label='channel 4', desc='enable channel 4'  )
    c_enable5  = Bool(False, label='channel 5', desc='enable channel 5'  )
    c_enable6  = Bool(False, label='channel 6', desc='enable channel 6'  )
    c_enable7  = Bool(False, label='channel 7', desc='enable channel 7'  )
    sum_enable = Bool(False, label='c0 + c1',   desc='enable sum c0 + c1')
    
    TracePlot = Instance( Plot )
    TraceData = Instance( ArrayPlotData )
    
    digits_data = Instance( ArrayPlotData )
    digits_plot = Instance( Plot )
    
    def __init__(self):
        super(PhotonTimeTrace, self).__init__()
        self.channels_mask = numpy.ones(8, dtype=bool)
        self._update_channels_list()
        self.on_trait_change(
            self._update_channels_list,
            'c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,c_enable7',
            dispatch='ui'
        )
        self.on_trait_change(self._update_T, 'T', dispatch='ui')
        self.on_trait_change(self._update_C0, 'C0', dispatch='ui')
        self.on_trait_change(self._update_C1, 'C1', dispatch='ui')
        self.on_trait_change(self._update_C2, 'C2', dispatch='ui')
        self.on_trait_change(self._update_C3, 'C3', dispatch='ui')
        self.on_trait_change(self._update_C4, 'C4', dispatch='ui')
        self.on_trait_change(self._update_C5, 'C5', dispatch='ui')
        self.on_trait_change(self._update_C6, 'C6', dispatch='ui')
        self.on_trait_change(self._update_C0C1, 'C0C1', dispatch='ui')
        
        self._create_digits_plot()

    def _update_channels_list(self):
        self.channels_mask = numpy.array([
            self.c_enable1, self.c_enable2, self.c_enable3, self.c_enable4,
            self.c_enable5, self.c_enable6, self.c_enable7,
        ])
        self.channels_list = numpy.argwhere(self.channels_mask).flatten()
        print(self.channels_list)
        self._create_counter()

    def _create_counter(self):
        # self._counter0 = tt.Counter(0, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter1 = tt.Counter(1, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter2 = tt.Counter(2, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter3 = tt.Counter(3, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter4 = tt.Counter(4, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter5 = tt.Counter(5, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        # self._counter6 = tt.Counter(6, int(self.SecondsPerPoint*1e12), self.TraceLength) 
        self._counter = tt.Counter(self.channels_list + 1, int(self.SecondsPerPoint*1e12), self.TraceLength)

    def _create_digits_plot(self):
        data = ArrayPlotData(image=numpy.zeros((2,2)))
        plot = Plot(data, width=500, height=500, resizable='hv', aspect_ratio=37.0/9, padding=8, padding_left=48, padding_bottom=36)
        plot.img_plot(
            'image', xbounds=(0, 1), ybounds=(0, 1), colormap=hot
        )
        plot.plots['plot0'][0].value_range.high_setting = 1
        plot.plots['plot0'][0].value_range.low_setting = 0
        plot.x_axis = None
        plot.y_axis = None
        self.digits_data = data
        self.digits_plot = plot
        
    def _C0_default(self):
        return numpy.zeros((self.TraceLength,))   
         
    def _C1_default(self):
        return numpy.zeros((self.TraceLength,))
         
    def _C2_default(self):
        return numpy.zeros((self.TraceLength,))
         
    def _C3_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C4_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C5_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C6_default(self):
        return numpy.zeros((self.TraceLength,))

    def _C7_default(self):
        return numpy.zeros((self.TraceLength,))
    
    def _C0C1_default(self):
        return numpy.zeros((self.TraceLength,))

    def _T_default(self):
        return self.SecondsPerPoint*numpy.arange(self.TraceLength)

    def _update_T(self):
        self.TraceData.set_data('t', self.T)

    def _update_C0(self):
        self.TraceData.set_data('y0', self.C0)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha0':
            #self.update_digits_plot(self.C0[-1])
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C0[-5*averagelength:]))

    def _update_C1(self):
        self.TraceData.set_data('y1', self.C1)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha1':
            #self.update_digits_plot(self.C1[-1])
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C1[-5*averagelength:]))
            
    def _update_C2(self):
        self.TraceData.set_data('y2', self.C2)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha2':
            #self.update_digits_plot(self.C2[-1])
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C2[-5*averagelength:]))

    def _update_C3(self):
        self.TraceData.set_data('y3', self.C3)
        #self.TracePlot.request_redraw()

        if self.digi_channel=='cha3':
            #self.update_digits_plot(self.C3[-1])
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C3[-5*averagelength:]))
            
    def _update_C4(self):
        self.TraceData.set_data('y4', self.C4)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha4':
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C4[-5*averagelength:]))


    def _update_C5(self):
        self.TraceData.set_data('y5', self.C5)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha5':
            #self.update_digits_plot(self.C5[-1])
            averagelength=int(self.RefreshRate/self.SecondsPerPoint)
            self.update_digits_plot(numpy.average(self.C5[-5*averagelength:]))

    def _update_C6(self):
        self.TraceData.set_data('y6', self.C6)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha6':
            self.update_digits_plot(self.C6[-1])
            
    def _update_C7(self):
        self.TraceData.set_data('y7', self.C7)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha7':
            self.update_digits_plot(self.C7[-1])
            
    def _update_C0C1(self):
        self.TraceData.set_data('y8', self.C0C1)
        #self.TracePlot.request_redraw()
        if self.digi_channel=='cha0+1':
            self.update_digits_plot(self.C0C1[-1])

    def update_digits_plot(self, counts):
        if counts>1.0e5:
            string = ('%5.1f' % (counts/1000.0))[:5] + 'k'
            data = numpy.zeros((37,9))
            for i, char in enumerate(string):
                data[6*i+1:6*i+6,1:-1] = DIGIT[char].transpose()
            if (counts/1000.0) >= 2e3:
                data *= 0.4
        else:
            string = ('%6i' % counts)[:6]
            data = numpy.zeros((37,9))
            for i, char in enumerate(string):
                data[6*i+1:6*i+6,1:-1] = DIGIT[char].transpose()
            
        self.digits_data.set_data('image', data.transpose()[::-1])
           
    def _TraceLength_changed(self):
        self.C0 = self._C0_default()
        self.C1 = self._C1_default()
        self.C2 = self._C2_default()
        self.C3 = self._C3_default()
        self.C4 = self._C4_default()
        self.C5 = self._C5_default()
        self.C6 = self._C6_default()
        self.C7 = self._C7_default()
        self.C0C1 = self._C0C1_default()
        self.T = self._T_default()
        self._create_counter()
        
    def _SecondsPerPoint_changed(self):
        self.T = self._T_default()
        self._create_counter()

    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y0=self.C0, y1=self.C1, y2=self.C2, y3=self.C3, y4=self.C4, y5=self.C5, y6=self.C6, y7=self.C7, y8=self.C0C1)
    
    def _TracePlot_default(self):
        plot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        plot.plot(('t','y0'), type='line', color='black')
        return plot
    
    #@on_trait_change('c_enable0,c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,c_enable7,sum_enable') # ToDo: fix channel 7
    @on_trait_change('c_enable1,c_enable2,c_enable3,c_enable4,c_enable5,c_enable6,c_enable7,sum_enable')
    def _replot(self):
        
        self.TracePlot = Plot(self.TraceData, width=500, height=500, resizable='hv')
        self.TracePlot.legend.align = 'll'
        
        n=0
        if self.c_enable1:
            self.TracePlot.plot(('t','y0'), type='line', color='steelblue',  name='channel 0')
            n+=1
        if self.c_enable2:
            self.TracePlot.plot(('t','y1'), type='line', color='skyblue',   name='channel 1')
            n+=1
        if self.c_enable3:
            self.TracePlot.plot(('t','y2'), type='line', color='red', name='channel 2')
            n+=1
        if self.c_enable4:
            self.TracePlot.plot(('t','y3'), type='line', color='black', name='channel 3')
            n+=1
        if self.c_enable5:
            self.TracePlot.plot(('t','y4'), type='line', color='coral',  name='channel 4')
            n+=1
        if self.c_enable6:
            self.TracePlot.plot(('t','y5'), type='line', color='sandybrown',   name='channel 5')
            n+=1
        if self.c_enable7:
            self.TracePlot.plot(('t','y6'), type='line', color='green', name='channel 6')
            n+=1
        #if self.c_enable7:
        #    self.TracePlot.plot(('t','y7'), type='line', color='black', name='channel 7')
        if self.sum_enable:
            self.TracePlot.plot(('t','y8'), type='line', color='black', name='sum c0 + c1')
            n+=1

        if n > 1:
            self.TracePlot.legend.visible = True
        else:
            self.TracePlot.legend.visible = False

    def _run(self):
        """Acquire Count Trace"""
        while True:
            threading.current_thread().stop_request.wait(self.RefreshRate)
            if threading.current_thread().stop_request.isSet():
                break
            channel_data = self._counter.getData() / self.SecondsPerPoint
            data = numpy.zeros((self.channels_mask.shape[0], channel_data.shape[1]))
            data[self.channels_list] = channel_data
            self.C0 = data[0]
            self.C1 = data[1]
            self.C2 = data[2]
            self.C3 = data[3]
            self.C4 = data[4]
            self.C5 = data[5]
            self.C6 = data[6]
            #self.C7 = self._counter7.getData() / self.SecondsPerPoint
            self.C0C1 = self.C0 + self.C1

    # def configure_traits_non_blocking(self):
    #         # Run the configure_traits() method in a separate thread
    #         ui = self.edit_traits(kind='live')
    #         ui.handler = StartThreadHandler()
    #         UI.invoke_in_main_thread(ui.control.exec_)
    #         # t = threading.Thread(target=self.configure_traits)
    #         # t.start()

    traits_view = View(         
        HGroup(
            VGroup(
                Item('TracePlot', editor=ComponentEditor()),
                Item('digits_plot', editor=ComponentEditor())),
                VGroup(
                    Item('c_enable1'),
                    Item('c_enable2'),
                    Item('c_enable3'),
                    Item('c_enable4'),
                    Item('c_enable5'),
                    Item('c_enable6'),
                    Item('c_enable7'),
                    Item('sum_enable')
                )
        ),
        Item('digi_channel', style='custom'),
        Item('TraceLength'),
        Item ('SecondsPerPoint'),
        Item ('RefreshRate'),
        title='Counter',
        width=800, height=600,
        buttons=[], resizable=True,
        handler=StartThreadHandler
    )



DIGIT = {}
DIGIT['0'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,1,1,
                       1,0,1,0,1,
                       1,1,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['1'] = numpy.array([0,0,1,0,0,
                       0,1,1,0,0,
                       1,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       0,0,1,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['2'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,1,1,1,1]).reshape(7,5)
DIGIT['3'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['4'] = numpy.array([0,0,1,0,0,
                       0,1,0,0,0,
                       0,1,0,0,0,
                       1,0,0,1,0,
                       1,1,1,1,1,
                       0,0,0,1,0,
                       0,0,0,1,0]).reshape(7,5)
DIGIT['5'] = numpy.array([1,1,1,1,1,
                       1,0,0,0,0,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['6'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,0,
                       1,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['7'] = numpy.array([1,1,1,1,1,
                       0,0,0,0,1,
                       0,0,0,0,1,
                       0,0,0,1,0,
                       0,0,1,0,0,
                       0,1,0,0,0,
                       1,0,0,0,0]).reshape(7,5)
DIGIT['8'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['9'] = numpy.array([0,1,1,1,0,
                       1,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,1,
                       0,0,0,0,1,
                       1,0,0,0,1,
                       0,1,1,1,0]).reshape(7,5)
DIGIT['.'] = numpy.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,1,0,0]).reshape(7,5)
DIGIT['k'] = numpy.array([1,0,0,0,0,
                       1,0,0,0,0,
                       1,0,0,0,1,
                       1,0,0,1,0,
                       1,0,1,0,0,
                       1,1,0,1,0,
                       1,0,0,0,1]).reshape(7,5)
DIGIT[' '] = numpy.array([0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0,
                       0,0,0,0,0]).reshape(7,5)    
                       

if __name__=='__main__':

    import logging
    
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().info('Starting logger.')

    from tools.emod import JobManager
    JobManager().start()

    p = PhotonTimeTrace()
    p.edit_traits()
    
    