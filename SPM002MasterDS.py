'''
Created on Aug 25, 2014

@author: Filip Lindau
'''
import sys
import PyTango
import SPM002_control as spm
import threading
import time
import numpy as np
from socket import gethostname
import Queue


class SpectrometerCommand:
    def __init__(self, command, data=None):
        self.command = command
        self.data = data

class SpectrometerDataMessage:
    def __init__(self, serial, attribute, data=None):
        self.serial = serial
        self.attribute = attribute
        self.data = data
        
class SpectrometerData:
    """Container class for spectrometer data. 
    Contains:
        serial: serial number of the spectrometer
        index: index of the spectrometer in the device list. 
                Used by the .dll for opening the correct spectrometer 
        lock: lock to prevent concurrency errors when accessing data
        commandQueue: queue for issuing commands to the spectrometer hardware thread
        dataQueue: queue for receiving data from the spectrometer hardware thread
        wavelengths (numpy array): wavelength calibration array
        spectrum (numpy array): spectrum array
        exposureTime: exposure time in ms during acquisition
        updateTime: time between acquisitions in ms
        state (PyTango.DevState): spectrometer state
        status (string): spectrometer status
        hardwareThread: thread responsible for doing the actual hardware access
    """
    def __init__(self, serial, index, dataQueue):
        self.serial = serial
        self.index = index
        self.lock = threading.Lock()
        self.commandQueue = Queue.Queue(100)
        self.dataQueue = dataQueue
        self.wavelengths = None
        self.spectrum = None
        self.exposureTime = None
        self.updateTime = None
        self.state = None
        self.status = ''

        self.hardwareThread = SpectrometerThread(self, serial, index, self.commandQueue, self.dataQueue)
        
    def startThread(self):
        self.stopThread()
        self.hardwareThread.start()
        
    def stopThread(self):
        if self.hardwareThread.is_alive() == True:
            self.hardwareThread.stopThread()
            self.hardwareThread.join(3)

class SpectrometerThread(threading.Thread):
    def __init__(self, parent, serial, spectrometerIndex, commandQueue, dataQueue):
        """Init new SpectrometerThread.
        Args: 
            parent: parent self object
            serial: serial number of spectrometer
            commandQueue: queue for issuing commands to the spectrometer thread
            dataQueue: queue for receiving responses from the spectrometer thread 
            spectrometerIndex: list of the spectrometer as received by populateDeviceList
            
            No locks are needed since all access to hardware and attributes are 
            within a single thread.
        """
        threading.Thread.__init__(self)
        
        self.parent = parent
        self.serial = serial 
        self.spectrometerIndex = spectrometerIndex
        
        self.state = PyTango.DevState.UNKNOWN
        self.status = ''
        self.stopStateThreadFlag = False
        self.commandQueue = commandQueue
        self.dataQueue = dataQueue
        
        self.stateHandlerDict = {PyTango.DevState.ON: self.onHandler,
                                PyTango.DevState.STANDBY: self.standbyHandler,
                                PyTango.DevState.ALARM: self.onHandler,
                                PyTango.DevState.FAULT: self.faultHandler,
                                PyTango.DevState.INIT: self.initHandler,
                                PyTango.DevState.UNKNOWN: self.unknownHandler,
                                PyTango.DevState.OFF: self.offHandler}
        
        self.expTime = None
        self.wavelengths = None
        self.spectrumData = None

        
    def run(self):
        self.stopStateThreadFlag = False
        self.stateHandlerDispatcher()
    
    def stateHandlerDispatcher(self):
        """Handles switch of states in the state machine thread.
        Each state handled method should exit by setting the next state,
        going back to this method. The previous state is also included when
        calling the next state handler method.
        The thread is stopped by setting the stopStateThreadFlag.
        """
        prevState = self.state
        while self.stopStateThreadFlag == False:
            try:
                self.stateHandlerDict[self.state](prevState)
                prevState = self.state
            except KeyError:
                self.stateHandlerDict[PyTango.DevState.UNKNOWN](prevState)
                prevState = self.state

    def stopThread(self):
        """Stops the state handler thread by setting the stopStateThreadFlag
        """
        self.stopStateThreadFlag = True
        
    def checkCommands(self, blockTime=0):
        """Checks the commandQueue for new commands. Must be called regularly.
        If the queue is empty the method exits immediately.
        """
        try:
            if blockTime == 0:
                cmd = self.commandQueue.get(block=False)
            else:
                cmd = self.commandQueue.get(block=True, timeout=blockTime)
            self.info_stream(str(cmd.command))
            if cmd.command == 'writeExposureTime':
                self.expTime = cmd.data
                self.setExposure(True)
                self.getExposure()
            elif cmd.command == 'readExposureTime':
                self.getExposure()
                    
            elif cmd.command == 'writeUpdateTime':
                self.updateTime = cmd.data
                self.setExposure(True)
                msg = SpectrometerDataMessage(self.serial, 'updatetime', self.updateTime)
                self.dataQueue.put(msg, block=False)
                
            elif cmd.command == 'readUpdateTime':
                msg = SpectrometerDataMessage(self.serial, 'updatetime', self.updateTime)
                self.dataQueue.put(msg, block=False)            

            elif cmd.command == 'writeAutoExposure':
                self.autoExpose = cmd.data

            elif cmd.command == 'on' or cmd.command == 'start':           
                if self.state not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.ON)

            elif cmd.command == 'stop' or cmd.command == 'standby':
                if self.state not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.STANDBY)
                
            elif cmd.command == 'off':
                if self.state not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.OFF)       

            elif cmd.command == 'init':
                if self.state not in [PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.UNKNOWN)
            
            elif cmd.command == 'readWavelengths':
                msg = SpectrometerDataMessage(self.serial, 'readWavelengths', self.wavelengths)
                try:
                    self.dataQueue.put_nowait(msg)
                except:
                    pass


        except Queue.Empty:
            pass
        
    def unknownHandler(self, prevState):
        """Method for handling the UNKOWN state. This is when the spectrometer 
        hardware is not yet connected. Could be the initial state or a hardware 
        problem (such as the spectrometer is disconnected).
        """
        self.info_stream('Entering unknownHandler')
        connectionTimeout = 1.0
        
        self.expTime = 200
        self.updateTime = 500
        self.autoExpose = False

        while self.stopStateThreadFlag == False:
            try:
                self.spectrometer.close_device()
            except Exception, e:
                pass

            self.info_stream('Trying to connect...')
            try:                
                self.spectrometer = spm.SPM002Control()
                self.setState(PyTango.DevState.INIT)
                self.info_stream('... connected')
                break
            
            except Exception, e:
                self.error_stream(''.join(('Could not create spectrometer object.', str(e))))
                self.setState(PyTango.DevState.UNKNOWN)
                self.status = ''.join(('Could not create spectrometer object.', str(e)))
                

            self.checkCommands(blockTime=connectionTimeout)        

    def initHandler(self, prevState):
        """Handles the INIT state. This when the spectrometer hardware is found after
        the UNKNOWN handler and is used to check response and setup the wavelength table.
        """
        self.info_stream('Entering initHandler')
        self.setState(PyTango.DevState.INIT)
        s_status = 'Starting initialization\n'
        self.status = s_status
        self.info_stream(s_status)
        initTimeout = 0.5  # Retry time interval
                
        while self.stopStateThreadFlag == False:
            self.checkCommands(blockTime=initTimeout)
            try:
                s = ''.join(('Setting device ', str(self.serial), '\n'))
                s_status = ''.join((s_status, s))
                self.status = s_status
                self.openSpectrometer()
            except Exception, e:
                self.error_stream('Could not open spectrometer')
                continue
            try:
                s = 'Retrieving wavelength table\n'
                s_status = ''.join((s_status, s))
                self.status = s_status
                self.info_stream(s)
                self.spectrometer.construct_wavelengths()
                self.wavelengths = self.spectrometer.wavelengths
                # Immediately push wavelength table to device server:
                msg = SpectrometerDataMessage(self.serial, 'wavelengths', self.wavelengths)
                self.dataQueue.put(msg, block=False)
            except Exception, e:
                self.error_stream('Could not construct wavelengths')
                continue
    
            self.status = 'Connected to spectrometer, not acquiring'
            self.info_stream('Initialization finished.')
            self.setState(PyTango.DevState.STANDBY)
            break

    def faultHandler(self, prevState):
        """Handles the FAULT state. Called when there was an error when
        reading the hardware. Attempts to clear the fault condition a few 
        times before switching to the UNKNOWN state. 
        """
        responseAttempts = 0
        maxAttempts = 5
        responseTimeout = 0.5
        self.info_stream('Entering faultHandler.')
        self.status = 'Fault condition detected'
        handledStates = [PyTango.DevState.FAULT]
            
        while self.stopStateThreadFlag == False:
            try:
                self.spectrometer.close_device()
                self.openSpectrometer()
                
                self.setState(prevState)
                self.info_stream('Fault condition cleared.')
                break
            except Exception, e:
                self.error_stream(''.join(('In faultHandler: Testing controller response. Returned ', str(e))))
                responseAttempts += 1
            if responseAttempts >= maxAttempts:
                self.setState(PyTango.DevState.UNKNOWN)
                self.status = 'Could not connect to controller'
                self.error_stream('Giving up fault handling. Going to UNKNOWN state.')
                break
            self.checkCommands(blockTime=responseTimeout)
            
    def offHandler(self, prevState):
        """Handles the OFF state where the spectrometer hardware is disconnected.
        """
        self.info_stream('Entering offHandler')
        try:
            self.spectrometer.close_device()
        except Exception, e:
            self.error_stream(''.join(('Could not disconnect from spectrometer, ', str(e))))
                
        self.set_status('Disconnected from spectrometer')
        while self.stopStateThreadFlag == False:
            if self.state != PyTango.DevState.OFF:
                break
            # Check if any new commands arrived:
            self.checkCommands(blockTime=0.5)

    def standbyHandler(self, prevState):
        """Handles the STANDBY state where the spectrometer hardware is connected
        but the spectrum acquisition is not started. Runs a loop checking commands
        and checks the hardware connections every 0.5 s to see that it is alive.
        """
        self.info_stream('Entering standbyHandler')
        self.status = 'Connected to spectrometer, not acquiring spectra'
        self.openSpectrometer()
        while self.stopStateThreadFlag == False:
            if self.state != PyTango.DevState.STANDBY:
                break
            # Check if any new commands arrived:
            self.checkCommands(blockTime=2)
            if self.state != PyTango.DevState.STANDBY:
                break
                        
            try:
                expTime = self.spectrometer.get_exposure_time() * 1e-3
            except Exception, e:
                self.error_stream('Error reading device')
                self.setState(PyTango.DevState.FAULT)

    def onHandler(self, prevState):
        """Handles the ON state where the spectrometer is connected and acquiring
        spectra. Runs a loop checking for commands and reading a new spectrum from the
        hardware every updateTime ms. The new spectrum is posted to the dataQueue.
        
        """
        self.info_stream('Entering onHandler')
        self.status = 'Connected to spectrometer, acquiring spectra'
        handledStates = [PyTango.DevState.ON, PyTango.DevState.ALARM]
        self.openSpectrometer()
        if self.updateTime > self.expTime:
            self.sleepTime = (self.updateTime - self.expTime) * 1e-3
        else:
            self.sleepTime = self.expTime * 1e-3
        s = ''.join(('Sleeptime: ', str(self.sleepTime)))
        self.info_stream(s)
        newSpectrumTimestamp = time.time()
        oldSpectrumTimestamp = time.time()
        self.spectrumData = self.spectrometer.spectrum
        nextUpdateTime = time.time()
        while self.stopStateThreadFlag == False:
            if self.state not in handledStates:
                break
            # Check if any new commands arrived:
            self.checkCommands(blockTime=0.01)
            
            # Check if we should break this loop and go to a new state handler:
            if self.state not in handledStates:
                break

            try:
                t = time.time()
#                self.debug_stream(''.join(("In onHandler()... time ", str(t), ", next update ", str(nextUpdateTime))))
                if t > nextUpdateTime:
                    self.spectrometer.acquire_spectrum()
                    newSpectrum = self.spectrometer.spectrum
                    newSpectrumTimestamp = time.time()
                    d = np.abs(newSpectrum - self.spectrumData).sum()
                    if d == 0:                    
                        if newSpectrumTimestamp - oldSpectrumTimestamp > 5:                     
                            self.set_state(PyTango.DevState.FAULT)
                            self.set_status('Spectrum not updating. Reconnecting.')
                            self.error_stream('Spectrum not updating. Reconnecting.')
                    else:
                        oldSpectrumTimestamp = newSpectrumTimestamp
                    self.spectrumData = np.copy(newSpectrum)
                    msg = SpectrometerDataMessage(self.serial, 'spectrum', self.spectrumData)
                    self.dataQueue.put(msg, block=False)
                    if self.updateTime > self.expTime:
                        self.sleepTime = (self.updateTime - self.expTime) * 1e-3
                    else:
                        self.sleepTime = self.expTime * 1e-3
                    
                    nextUpdateTime = t + self.sleepTime
 
            except Exception, e:
                self.setState(PyTango.DevState.FAULT)
                self.status = 'Error reading hardware.'
            
    def info_stream(self, s):
        msg = SpectrometerDataMessage(self.serial, 'info', s)
        self.dataQueue.put(msg, block=False) 

    def debug_stream(self, s):
        msg = SpectrometerDataMessage(self.serial, 'debug', s)
        self.dataQueue.put(msg, block=False) 

    def error_stream(self, s):
        msg = SpectrometerDataMessage(self.serial, 'error', s)
        self.dataQueue.put(msg, block=False) 
        
    def setState(self, state):
        """Sets a new spectrometer state and posts it to the dataQueue
        
        """
        self.state = state
        msg = SpectrometerDataMessage(self.serial, 'state', self.state)
        self.dataQueue.put(msg, block=False)

    def setStatus(self, status):
        """Sets a new spectrometer status message and posts it to the dataQueue
        
        """
        self.status = status
        msg = SpectrometerDataMessage(self.serial, 'status', self.status)
        self.dataQueue.put(msg, block=False)
        
        
    def openSpectrometer(self):
        """Opens the communication with the spectrometer hardware. The index of
        the spectrometer in the deviceList retrieved from populateDeviceList is used.
        
        """
        # If the device was closed, we open it again
        self.debug_stream('Entering openSpectrometer')
        if self.spectrometer.device_index == None:
            try:
                self.spectrometer.open_device_index(self.spectrometerIndex)
                self.debug_stream(''.join(('openSpectrometer: device', str(self.serial), ' opened')))
            except Exception, e:
                self.error_stream(''.join(('Could not open device ', str(self.serial), str(e))))
                self.state = PyTango.DevState.UNKNOWN
                self.status = ''.join(('Could not open device ', str(self.serial)))

    def setExposure(self, forceSet=False):
        """Sets the exposure time of the acquisition according to the class
        member expTime. Also updates the sleepTime used in onHandler to time
        acquisition.
        
        """
        self.info_stream('In setExposure: ')
#         if self.autoExpose == True:
#             # We will try to keep the max reading at around nomI counts
#             nomI = 2500.0
#             maxI = np.max(self.spectrumData)
#             # Don't adjust if the intensity is within 10% of nominal    
#             if (nomI / maxI > 1.1) or (nomI / maxI < 0.9):         
#                 newExp = nomI / maxI * self.expTime
#                 # Don't adjust to over 500 ms, the update rate would be too slow
#                 if newExp > 500:
#                     newExp = 500            
#                 self.expTime = newExp
#                 forceSet = True 

        if forceSet == True:
            try:
                self.spectrometer.set_exposure_time(int(self.expTime * 1e3))  # expTime is in ms, the spectrometer excpects us
            except Exception, e:
                self.set_state(PyTango.DevState.FAULT)
                self.set_status(''.join(('Could not set exposure time', str(e))))
                self.error_stream(''.join(('Could not set exposure time', str(e))))
            if self.updateTime > self.expTime:
                self.sleepTime = (self.updateTime - self.expTime) * 1e-3
            else:
                self.sleepTime = self.expTime * 1e-3
                
                    
    def getExposure(self):
        """Reads the exposure time from the spectrometer hardware and posts
        exposure time to the dataQueue
        """
        if self.state not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
            try:
                self.expTime = self.spectrometer.get_exposure_time() * 1e-3
            except Exception, e:
                self.error_stream('Error reading device')
                self.setState(PyTango.DevState.FAULT)
                
        msg = SpectrometerDataMessage(self.serial, 'exposuretime', self.expTime)
        self.dataQueue.put(msg, block=False)


        
#==================================================================
#   SPM002MasterDS Class Description:
#
#         Control of a Photon control SPM002 spectrometer
#
#==================================================================
#     Device States Description:
#
#   DevState.ON :       Connected to one or more spectrometers
#   DevState.FAULT :    Error detected
#   DevState.UNKNOWN :  Communication problem
#   DevState.INIT :     Initializing spectrometers. Could take time.
#==================================================================


class SPM002MasterDS(PyTango.Device_4Impl):

#--------- Add you global variables here --------------------------

#------------------------------------------------------------------
#     Device constructor
#------------------------------------------------------------------
    def __init__(self, cl, name):
        PyTango.Device_4Impl.__init__(self, cl, name)
        SPM002MasterDS.init_device(self)

#------------------------------------------------------------------
#     Device destructor
#------------------------------------------------------------------
    def delete_device(self):
        self.debug_stream(''.join(("[Device delete_device method] for device", self.get_name())))
        self.stopSpectrometerThreads()
        self.dataReceiveThreadStopFlag = True
        self.dataReceiveThread.join(3)

#------------------------------------------------------------------
#     Device initialization
#------------------------------------------------------------------
    def init_device(self):
        self.debug_stream(''.join(("In ", self.get_name(), "::init_device()")))        
        self.set_state(PyTango.DevState.INIT)
        self.get_device_properties(self.get_device_class())
        
        try:
            self.stopSpectrometerThreads()
        except:
            pass
        
        try:
            self.dataReceiveThreadStopFlag = True
            self.dataReceiveThread.join(3)
        except:
            pass
        
        self.controlSpectrometer = spm.SPM002Control()
        
        try:
            self.spectrometerList
            self.spectrometerDict
            self.dataQueue
            self.spectrometerThreads
        except AttributeError:
            self.spectrometerThreads = []
            self.spectrometerList = []
            self.spectrometerDict = {}
            self.dataQueue = Queue.Queue(1000)
        self.dataReceiveThread = threading.Thread()
        threading.Thread.__init__(self.dataReceiveThread, target=self.dataReceiveThreadHandler)
        self.dataReceiveThreadStopFlag = False
#        self.enumerateSpectrometers()
        
        self.set_change_event('state', True)
        
        self.dataReceiveThread.start()
        
    def stopSpectrometerThreads(self):
        for spec in self.spectrometerList:
            self.info_stream(''.join(('Stopping thread ', str(spec))))
            self.spectrometerDict[spec].stopThread()
            
    def startSpectrometerThreads(self):
        self.stopSpectrometerThreads()
        for spec in self.spectrometerList:
            self.info_stream(''.join(('Starting thread ', str(spec))))
            self.spectrometerDict[spec].startThread()
            
    def enumerateSpectrometers(self):
        self.info_stream('In enumerateSpectrometers')
        self.set_state(PyTango.DevState.INIT)
        self.stopSpectrometerThreads()
        self.controlSpectrometer.populate_device_list()
        self.spectrometerList = self.controlSpectrometer.serial_list
        self.info_stream(''.join(('Found ', str(self.spectrometerList.__len__()), ' spectrometers.')))
        for ind, spec in enumerate(self.spectrometerList):
            if self.spectrometerDict.has_key(spec) == False:
                self.info_stream(''.join(('Adding spectrometer ', str(spec), ' to list.')))
                self.spectrometerDict[spec] = SpectrometerData(spec, ind, self.dataQueue)

                attrInfo = [[PyTango.DevString, PyTango.SCALAR, PyTango.READ],
                    {
                        'description':"Spectrometer state",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'State'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerState, is_allo_meth=self.is_SpectrometerState_allowed)
                self.set_change_event(attrName, True, False)

                attrInfo = [[PyTango.DevString, PyTango.SCALAR, PyTango.READ],
                    {
                        'description':"Spectrometer status",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'Status'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerStatus, is_allo_meth=self.is_SpectrometerStatus_allowed)
                self.set_change_event(attrName, True, False)

                attrInfo = [[PyTango.DevDouble, PyTango.SCALAR, PyTango.READ_WRITE],
                    {
                        'description':"Exposure time in ms",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'ExposureTime'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerExposureTime, w_meth=self.write_SpectrometerExposureTime, is_allo_meth=self.is_SpectrometerExposureTime_allowed)
                self.set_change_event(attrName, True, False)
                cmdMsg = SpectrometerCommand('readExposureTime')
                self.spectrometerDict[spec].commandQueue.put(cmdMsg)

                attrInfo = [[PyTango.DevDouble, PyTango.SPECTRUM, PyTango.READ, 3648],
                    {
                        'description':"spectrum",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'Spectrum'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerSpectrum, is_allo_meth=self.is_SpectrometerSpectrum_allowed)
                self.set_change_event(attrName, True, False)

                attrInfo = [[PyTango.DevDouble, PyTango.SPECTRUM, PyTango.READ, 3648],
                    {
                        'description':"wavelength table",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'Wavelengths'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerWavelengths, is_allo_meth=self.is_SpectrometerWavelengths_allowed)

                attrInfo = [[PyTango.DevDouble, PyTango.SCALAR, PyTango.READ_WRITE],
                    {
                        'description':"Time between spectrum updates in ms",
                        'Memorized':"false",
                    } ]
                attrName = ''.join(('Spectrometer', str(spec), 'UpdateTime'))
                attrData = PyTango.AttrData(attrName, self.get_name(), attrInfo)
                self.add_attribute(attrData, r_meth=self.read_SpectrometerUpdateTime, w_meth=self.write_SpectrometerUpdateTime, is_allo_meth=self.is_SpectrometerUpdateTime_allowed)                
                cmdMsg = SpectrometerCommand('readUpdateTime')
                self.spectrometerDict[spec].commandQueue.put(cmdMsg)
                

                
        self.startSpectrometerThreads()
        self.set_state(PyTango.DevState.ON)
        
    def dataReceiveThreadHandler(self):
        time.sleep(0.5)
        self.enumerateSpectrometers()
               
        while (self.dataReceiveThreadStopFlag == False):
            try:
                rcv = self.dataQueue.get(block=True, timeout=0.005)
                serial = rcv.serial
                if rcv.attribute == 'spectrum':
                    attrName = ''.join(('Spectrometer', str(serial), 'Spectrum'))
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].spectrum = rcv.data
#                         try:
#                             self.push_change_event(attrName, rcv.data)
#                         except Exception, e:
#                             self.error_stream(''.join(('Could not push spectrum event: ', str(e))))
                        self.debug_stream('Pushed spectrum change event')
                elif rcv.attribute == 'wavelengths':
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].wavelengths = rcv.data
                elif rcv.attribute == 'exposuretime':
                    attrName = ''.join(('Spectrometer', str(serial), 'ExposureTime'))
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].exposureTime = rcv.data
#                         try:
#                             self.push_change_event(attrName, rcv.data)
#                         except Exception, e:
#                             self.error_stream(''.join(('Could not push exposuretime event: ', str(e))))                            
                        self.debug_stream('Pushed exposuretime change event')                        
                elif rcv.attribute == 'updatetime':
                    attrName = ''.join(('Spectrometer', str(serial), 'UpdateTime'))
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].updateTime = rcv.data
#                         try:
#                             self.push_change_event(attrName, rcv.data)
#                         except Exception, e:
#                             self.error_stream(''.join(('Could not push updatetime event: ', str(e))))
                        self.debug_stream('Pushed updatetime change event')
                elif rcv.attribute == 'state':
                    attrName = ''.join(('Spectrometer', str(serial), 'State'))
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].state = rcv.data
                        try:
                            self.set_state(rcv.data)
                            self.push_change_event(attrName, str(rcv.data))
                        except Exception, e:
                            self.error_stream(''.join(('Could not push state event: ', str(e))))
                        self.debug_stream('Pushed state change event')
                elif rcv.attribute == 'status':
                    attrName = ''.join(('Spectrometer', str(serial), 'Status'))
                    with self.spectrometerDict[serial].lock:
                        self.spectrometerDict[serial].status = rcv.data
#                         try:
#                             self.push_change_event(attrName, str(rcv.data))
#                         except Exception, e:
#                             self.error_stream(''.join(('Could not push status event: ', str(e))))
                        self.debug_stream('Pushed status change event')
                elif rcv.attribute == 'info':
                    self.info_stream(''.join(('Spectrometer ', str(serial), ': ', rcv.data)))
                elif rcv.attribute == 'debug':
                    self.debug_stream(''.join(('Spectrometer ', str(serial), ': ', rcv.data)))
                elif rcv.attribute == 'error':
                    self.error_stream(''.join(('Spectrometer ', str(serial), ': ', rcv.data)))
                        
            except Queue.Empty:
                pass
            except KeyError:
                self.error_stream(''.join(("In dataReceiveThreadHandler: Serial ", str(serial), " not in spectrometer dictionary")))
        
#------------------------------------------------------------------
#     Always excuted hook method
#------------------------------------------------------------------
    def always_executed_hook(self):
        pass

#------------------------------------------------------------------
#     Read Attribute Hardware
#------------------------------------------------------------------
    def read_attr_hardware(self, data):
        pass


#------------------------------------------------------------------
#     Read DeviceList attribute
#------------------------------------------------------------------
    def read_DeviceList(self, attr):
        
        #     Add your own code here
        self.info_stream(''.join(('Reading DeviceList ')))
        attr_DeviceList_read = self.spectrometerList
        attr.set_value(attr_DeviceList_read, attr_DeviceList_read.__len__())


#---- DeviceList attribute State Machine -----------------
    def is_DeviceList_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.OFF,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     Read SpectrometerState attribute
#------------------------------------------------------------------
    def read_SpectrometerState(self, attr):
        
        #     Add your own code here
        self.info_stream(''.join(('Reading SpectrometerState for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('State')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = str(self.spectrometerDict[serial].state)
            attr.set_value(attr_read)


#---- SpectrometerState attribute State Machine -----------------
    def is_SpectrometerState_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     Read SpectrometerStatus attribute
#------------------------------------------------------------------
    def read_SpectrometerStatus(self, attr):
        
        #     Add your own code here
        self.info_stream(''.join(('Reading SpectrometerStatus for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('Status')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = str(self.spectrometerDict[serial].status)
            attr.set_value(attr_read)


#---- SpectrometerState attribute State Machine -----------------
    def is_SpectrometerStatus_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     SpectrometerExposureTime attribute
#------------------------------------------------------------------
    def read_SpectrometerExposureTime(self, attr):
        self.info_stream(''.join(('Reading SpectrometerExposureTime for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('ExposureTime')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = self.spectrometerDict[serial].exposureTime
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def write_SpectrometerExposureTime(self, attr):        
        self.info_stream(''.join(('Writing SpectrometerExposureTime for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('ExposureTime')[0].rsplit('Spectrometer')[1])
        data = attr.get_write_value()
        cmdMsg = SpectrometerCommand('writeExposureTime', data)
        self.spectrometerDict[serial].commandQueue.put(cmdMsg)

    def is_SpectrometerExposureTime_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     SpectrometerUpdateTime attribute
#------------------------------------------------------------------
    def read_SpectrometerUpdateTime(self, attr):
        self.info_stream(''.join(('Reading SpectrometerUpdateTime for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('UpdateTime')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = self.spectrometerDict[serial].updateTime
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def write_SpectrometerUpdateTime(self, attr):        
        self.info_stream(''.join(('Writing SpectrometerUpdateTime for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('UpdateTime')[0].rsplit('Spectrometer')[1])
        data = attr.get_write_value()
        cmdMsg = SpectrometerCommand('writeUpdateTime', data)
        self.spectrometerDict[serial].commandQueue.put(cmdMsg)

    def is_SpectrometerUpdateTime_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     SpectrometerSpectrum attribute
#------------------------------------------------------------------
    def read_SpectrometerSpectrum(self, attr):
        self.info_stream(''.join(('Reading SpectrometerSpectrum for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('Spectrum')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = self.spectrometerDict[serial].spectrum
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
            attr.set_value(attr_read, attr_read.shape[0])

    def is_SpectrometerSpectrum_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.STANDBY,
                                PyTango.DevState.FAULT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     SpectrometerWavelengths attribute
#------------------------------------------------------------------
    def read_SpectrometerWavelengths(self, attr):
        self.info_stream(''.join(('Reading SpectrometerWavelengths for ', attr.get_name())))
        serial = int(attr.get_name().rsplit('Wavelengths')[0].rsplit('Spectrometer')[1])
        with self.spectrometerDict[serial].lock:
            attr_read = self.spectrometerDict[serial].wavelengths
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = [0.0]
            attr.set_value(attr_read, attr_read.shape[0])

    def is_SpectrometerWavelengths_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True



#==================================================================
#
#     SPM002MasterDS command methods
#
#==================================================================

#------------------------------------------------------------------
#     StartSpectrometer command:
#
#     Description: Start acquiring on spectrometer with serial number 
#                
#------------------------------------------------------------------
    def StartSpectrometer(self, serialNumber):
        self.info_stream(''.join(("In ", self.get_name(), "::StartSpectrometer(", str(serialNumber), ")")))
        cmdMsg = SpectrometerCommand('start')
        self.spectrometerDict[serialNumber].commandQueue.put(cmdMsg)

#---- StartSpectrometer command State Machine -----------------
    def is_StartSpectrometer_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     StopSpectrometer command:
#
#     Description: Stop acquiring on spectrometer with serial number 
#                
#------------------------------------------------------------------
    def StopSpectrometer(self, serialNumber):
        self.info_stream(''.join(("In ", self.get_name(), "::StopSpectrometer(", str(serialNumber), ")")))
        cmdMsg = SpectrometerCommand('stop')
        self.spectrometerDict[serialNumber].commandQueue.put(cmdMsg)

#---- StopSpectrometer command State Machine -----------------
    def is_StopSpectrometer_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     InitSpectrometer command:
#
#     Description: Initialize spectrometer with serial number 
#                
#------------------------------------------------------------------
    def InitSpectrometer(self, serialNumber):
        self.info_stream(''.join(("In ", self.get_name(), "::InitSpectrometer(", str(serialNumber), ")")))
        cmdMsg = SpectrometerCommand('init')
        self.spectrometerDict[serialNumber].commandQueue.put(cmdMsg)

#---- InitSpectrometer command State Machine -----------------
    def is_InitSpectrometer_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True
        
#------------------------------------------------------------------
#     PopulateDeviceList command:
#
#     Description: Enumerate the spectrometers on the usb bus 
#                
#------------------------------------------------------------------
    def PopulateDeviceList(self):
        self.info_stream(''.join(("In ", self.get_name(), "::PopulateDeviceList()")))
        self.enumerateSpectrometers()

#---- InitSpectrometer command State Machine -----------------
    def is_PopulateDeviceList_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True
        
#==================================================================
#
#     SPM002MasterDSClass class definition
#
#==================================================================
class SPM002MasterDSClass(PyTango.DeviceClass):
    #     Class Properties
    class_property_list = {
        }


    #     Device Properties
    device_property_list = {        
        }
    
    #     Command definitions
    cmd_list = {
        'StartSpectrometer':
            [[PyTango.DevLong, "Serial number of spectrometer"],
            [PyTango.DevVoid, ""]],
        'StopSpectrometer':
            [[PyTango.DevLong, "Serial number of spectrometer"],
            [PyTango.DevVoid, ""]],
        'InitSpectrometer':
            [[PyTango.DevLong, "Serial number of spectrometer"],
            [PyTango.DevVoid, ""]],
        'PopulateDeviceList':
            [[PyTango.DevVoid, ""],
            [PyTango.DevVoid, ""]],
        }


    attr_list = { 'DeviceList':
            [[PyTango.DevLong,
            PyTango.SPECTRUM,
            PyTango.READ, 16]],

                 
    }
#------------------------------------------------------------------
#     SPM002MasterDSClass Constructor
#------------------------------------------------------------------
    def __init__(self, name):
        PyTango.DeviceClass.__init__(self, name)
        self.set_type(name);
        print "In SPM002MasterDSClass  constructor"
        
#==================================================================
#
#     SPM002MasterDS class main method
#
#==================================================================
if __name__ == '__main__':
    try:
        py = PyTango.Util(sys.argv)
        py.add_class(SPM002MasterDSClass, SPM002MasterDS, 'SPM002MasterDS')

        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

    except PyTango.DevFailed, e:
        print '-------> Received a DevFailed exception:', e
    except Exception, e:
        print '-------> An unforeseen exception occured....', e
