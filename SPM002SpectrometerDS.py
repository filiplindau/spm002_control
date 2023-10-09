'''
Created on Aug 27, 2014

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


#==================================================================
#   SPM002SpectrometerDS Class Description:
#
#         Control of a Photon control SPM002 spectrometer
#
#==================================================================
#     Device States Description:
#
#   DevState.ON :       Connected to spectrometer, acquiring spectra
#   DevState.OFF :      Disconnected from spectrometer
#   DevState.FAULT :    Error detected
#   DevState.UNKNOWN :  Communication problem
#   DevState.STANDBY :  Connected to spectrometer, not acquiring
#   DevState.INIT :     Initializing spectrometer. Could take time.
#==================================================================


class SPM002SpectrometerDS(PyTango.Device_4Impl):

#--------- Add you global variables here --------------------------

#------------------------------------------------------------------
#     Device constructor
#------------------------------------------------------------------
    def __init__(self, cl, name):
        PyTango.Device_4Impl.__init__(self, cl, name)
        SPM002SpectrometerDS.init_device(self)

#------------------------------------------------------------------
#     Device destructor
#------------------------------------------------------------------
    def delete_device(self):
        with self.streamLock:
            self.info_stream(''.join(("[Device delete_device method] for device", self.get_name())))
        self.stopThread()


#------------------------------------------------------------------
#     Device initialization
#------------------------------------------------------------------
    def init_device(self):
        self.streamLock = threading.Lock()
        with self.streamLock:
            self.info_stream(''.join(("In ", self.get_name(), "::init_device()")))        
        self.set_state(PyTango.DevState.UNKNOWN)
        self.get_device_properties(self.get_device_class())
        
        # Try stopping the stateThread if it was started before. We fail if this
        # is the initial start.
        try:
            self.stopStateThread()
            
        except Exception, e:
            pass

        self.attrLock = threading.Lock()         
        self.eventIdList = []          
        self.stateThread = threading.Thread()
        threading.Thread.__init__(self.stateThread, target=self.stateHandlerDispatcher)
        self.eventThread = threading.Thread()
        threading.Thread.__init__(self.eventThread, target=self.eventHandler)
        
        self.commandQueue = Queue.Queue(100)
        
        self.stateHandlerDict = {PyTango.DevState.ON: self.onHandler,
                                PyTango.DevState.STANDBY: self.standbyHandler,
                                PyTango.DevState.ALARM: self.onHandler,
                                PyTango.DevState.FAULT: self.faultHandler,
                                PyTango.DevState.INIT: self.initHandler,
                                PyTango.DevState.UNKNOWN: self.unknownHandler,
                                PyTango.DevState.OFF: self.offHandler}

        self.stopStateThreadFlag = False
        
        self.stateThread.start()
        

    def stateHandlerDispatcher(self):
        """Handles switch of states in the state machine thread.
        Each state handled method should exit by setting the next state,
        going back to this method. The previous state is also included when
        calling the next state handler method.
        The thread is stopped by setting the stopStateThreadFlag.
        """
        prevState = self.get_state()
        while self.stopStateThreadFlag == False:
            try:
                self.stateHandlerDict[self.get_state()](prevState)
                prevState = self.get_state()
            except KeyError:
                self.stateHandlerDict[PyTango.DevState.UNKNOWN](prevState)
                prevState = self.get_state()

    def stopThread(self):
        """Stops the state handler thread by setting the stopStateThreadFlag
        """
        self.stopStateThreadFlag = True
        self.stateThread.join(3)
        self.unsubscribeEvents()
        
    def unknownHandler(self, prevState):
        """Handles the UNKNOWN state, before communication with the master device
        has been established. Tries to create a deviceproxy object.
        """
        with self.streamLock:
            self.info_stream('Entering unknownHandler')
        connectionTimeout = 1.0
        
        self.wavelengths = None
        self.wavelengthsROI = None
        self.updateTime = None
        self.expTime = None
        self.spectrum = None
        self.spectrumROI = None
        self.peakROI = None
        self.peakROIIndex = np.array([0, 3647])
        self.peakEnergy = None
        self.peakWavelength = None
        self.peakWidth = None
        
        while self.stopStateThreadFlag == False:
            self.unsubscribeEvents()
            with self.streamLock:
                self.info_stream('Trying to connect...')        
            try:
                self.masterDevice = PyTango.DeviceProxy(self.Master)                
            except PyTango.DevFailed, e:
                with self.streamLock:
                    self.error_stream(''.join(('Could not create deviceproxy for ', self.Master)))
                with self.streamLock:
                    self.error_stream(str(e))
                self.checkCommands(blockTime=connectionTimeout)
                continue
            self.set_state(PyTango.DevState.INIT)
            break
        
    def initHandler(self, prevState):
        """Handles the INIT state. Tries to setup event subscription on the master
        device and retrieves the wavelength table.
        """
        with self.streamLock:
            self.info_stream('Entering initHandler')
        waitTime = 1.0
        
        while self.stopStateThreadFlag == False:
            try:
                # Wait until masterdevice is no longer UNKNOWN:
                if self.masterDevice.state() == PyTango.DevState.UNKNOWN:
                    self.checkCommands(blockTime=waitTime)
                    continue 
                attrName = ''.join(('Spectrometer', str(self.Serial), 'Wavelengths'))
                wavelengthsAttr = self.masterDevice.read_attribute(attrName)
                with self.attrLock:
                    self.wavelengths = wavelengthsAttr.value
                    self.wavelengthsROI = self.wavelengths[self.peakROIIndex[0] : self.peakROIIndex[1]]
                    self.peakROI = np.array([self.wavelengthsROI[0], self.wavelengthsROI[-1]])
                    
                self.subscribeEvents()
                self.masterDevice.command_inout('StopSpectrometer', self.Serial)

                
            except Exception, e:
                with self.streamLock:
                    self.error_stream(''.join(('Error when initializing device')))
                    self.error_stream(str(e))
                self.checkCommands(blockTime=waitTime)
                continue
                
            self.set_state(PyTango.DevState.STANDBY)
            cmdMsg = SpectrometerCommand('readExposureTime')
            self.commandQueue.put(cmdMsg)
            cmdMsg = SpectrometerCommand('readUpdateTime')
            self.commandQueue.put(cmdMsg)
            break

    def standbyHandler(self, prevState):
        """Handles the STANDBY state. Connected to the spectrometer but not
        acquiring spectra. Waits in a loop checking commands. 
        """
        with self.streamLock:
            self.info_stream('Entering standbyHandler')
        handledStates = [PyTango.DevState.STANDBY]
        waitTime = 0.1
        
        while self.stopStateThreadFlag == False:
            with self.attrLock:
                state = self.get_state()
            if state not in handledStates:
                break
            self.checkCommands(blockTime=waitTime)
            
    def onHandler(self, prevState):
        """Handles the ON state. Connected to the spectrometer and 
        acquiring spectra. Waits in a loop checking commands. Spectrometer 
        events are handled in a callback function spectrumEvent
        """
        with self.streamLock:
            self.info_stream('Entering onHandler')
        handledStates = [PyTango.DevState.ON, PyTango.DevState.ALARM]
        waitTime = 0.1
        
#        self.eventThread.start()
        
        while self.stopStateThreadFlag == False:
            with self.attrLock:
                state = self.get_state()
            if state not in handledStates:
                break
            self.checkCommands(blockTime=waitTime)
            attrName = ''.join(('Spectrometer', str(self.Serial), 'Spectrum'))
            with self.attrLock:
                attr = self.masterDevice.read_attribute(attrName)
                self.spectrum = attr.value
                with self.streamLock:
                    self.debug_stream('In spectrumEvent: spectrum retrieved')
                try:
                    self.spectrumROI = self.spectrum[self.peakROIIndex[0] : self.peakROIIndex[1]]
                    with self.streamLock:
                        self.debug_stream('In spectrumEvent: roi extracted')
                except Exception, e:
                    with self.streamLock:
                        self.error_stream(''.join(('In spectrumEvent: ', str(e))))
            self.calculateSpectrumParameters()
            with self.streamLock:
                self.debug_stream('In spectrumEvent: parameters calculated')
            
            

    def faultHandler(self, prevState):
        """Handles the FAULT state. A problem has been detected.
        """
        with self.streamLock:
            self.info_stream('Entering faultHandler')
        handledStates = [PyTango.DevState.FAULT]
        waitTime = 0.1
        
        while self.stopStateThreadFlag == False:
            if self.get_state() not in handledStates:
                break
            self.checkCommands(blockTime=waitTime)
            
    def offHandler(self, prevState):
        """Handles the OFF state. Does nothing, just goes back to STANDBY.
        """
        with self.streamLock:
            self.info_stream('Entering offHandler')
        self.set_state(PyTango.DevState.STANDBY)

    def eventHandler(self):
        """ Event handling is done in a separate thread to avoid concurrency problems
        """
        self.unsubscribeEvents()
        self.subscribeEvents()
        while self.stopEventThreadFlag == False:
            time.sleep(0.1)

    def subscribeEvents(self):
        try:
            attrName = ''.join(('Spectrometer', str(self.Serial), 'State'))
            eventId = self.masterDevice.subscribe_event(attrName, PyTango.EventType.CHANGE_EVENT, self.stateEvent)
            self.eventIdList.append(eventId)
        except PyTango.EventSystemFailed, e:
            with self.streamLock:
                self.error_stream(''.join(('Error subscribing to STATE event: ', str(e))))
            raise

#         try:
#             attrName = ''.join(('Spectrometer', str(self.Serial), 'ExposureTime'))
#             eventId = self.masterDevice.subscribe_event(attrName, PyTango.EventType.CHANGE_EVENT, self.exposureTimeEvent)
#             self.eventIdList.append(eventId)
#         except PyTango.EventSystemFailed, e:
#             with self.streamLock:
#                 self.error_stream(''.join(('Error subscribing to EXPOSURETIME event: ', str(e))))
#             raise
# 
#         try:
#             attrName = ''.join(('Spectrometer', str(self.Serial), 'Spectrum'))
#             eventId = self.masterDevice.subscribe_event(attrName, PyTango.EventType.CHANGE_EVENT, self.spectrumEvent)
#             self.eventIdList.append(eventId)
#         except PyTango.EventSystemFailed, e:
#             with self.streamLock:
#                 self.error_stream(''.join(('Error subscribing to SPECTRUM event: ', str(e))))
#             raise
        
    def unsubscribeEvents(self):
        try:
            self.masterDevice  # Check if the masterDevice is defined, if not AttributeError is thrown
            with self.streamLock:
                self.info_stream('Unsubscribing events...')
            for eventId in self.eventIdList:
                try:
                    self.masterDevice.unsubscribe_event(eventId)
                except PyTango.EventSystemFailed, e:
                    with self.streamLock:
                        self.error_stream(''.join(('Error event system failed unsubscribing event ', str(eventId))))
                        self.error_stream(str(e))
                    
                except Exception, e:
                    with self.streamLock:
                        self.error_stream(''.join(('General Error unsubscribing event ', str(eventId))))
                        self.error_stream(str(e))
        except AttributeError:
            pass

    def stateEvent(self, event):
        if event.err == True:
            with self.streamLock:
                self.info_stream(''.join(('Error for state event :', str(event.errors))))
        else:
            newMasterState = event.attr_value.value
            with self.streamLock:
                self.info_stream(''.join(('Master device state changed to ', str(newMasterState))))
            if newMasterState == 'ON':
                self.set_state(PyTango.DevState.ON)
            elif newMasterState == 'STANDBY':
                self.set_state(PyTango.DevState.STANDBY)
            elif newMasterState == 'ALARM':
                self.set_state(PyTango.DevState.ALARM)
            elif newMasterState == 'FAULT':
                self.set_state(PyTango.DevState.FAULT)
            elif newMasterState == 'UNKNOWN':
                self.set_state(PyTango.DevState.FAULT)
            elif newMasterState == 'OFF':
                self.set_state(PyTango.DevState.STANDBY)
            elif newMasterState == 'INIT':
                self.set_state(PyTango.DevState.INIT)
                
            
    def spectrumEvent(self, event):
        with self.streamLock:
            self.debug_stream('spectrumEvent received')
        if event.err == True:
            with self.streamLock:
                self.info_stream(''.join(('Error for spectrum event :', str(event.errors))))
        else:
            with self.attrLock:
                self.spectrum = event.attr_value.value
                with self.streamLock:
                    self.debug_stream('In spectrumEvent: spectrum retrieved')
                try:
                    self.spectrumROI = self.spectrum[self.peakROIIndex[0] : self.peakROIIndex[1]]
                    with self.streamLock:
                        self.debug_stream('In spectrumEvent: roi extracted')
                except Exception, e:
                    with self.streamLock:
                        self.error_stream(''.join(('In spectrumEvent: ', str(e))))
            self.calculateSpectrumParameters()
            with self.streamLock:
                self.debug_stream('In spectrumEvent: parameters calculated')
            
    def exposureTimeEvent(self, event):
        with self.streamLock:
            self.debug_stream('exposureTimeEvent received')
        if event.err == True:
            with self.streamLock:
                self.info_stream(''.join(('Error for exposureTime event :', str(event.errors))))
        else:
            with self.attrLock:
                self.expTime = event.attr_value.value
            
    def updateTimeEvent(self, event):
        with self.streamLock:
            self.debug_stream('updateTimeEvent received')
        if event.err == True:
            with self.streamLock:
                self.info_stream(''.join(('Error for updateTime event :', str(event.errors))))
        else:
            with self.attrLock:
                self.updateTime = event.attr_value.value
        
    def calculateSpectrumParameters(self):
        with self.streamLock:
            self.debug_stream('In calculateSpectrumParameters: entering')
        t0 = time.clock()
        with self.attrLock:
            sp = np.copy(self.spectrumROI)
            with self.streamLock:
                self.debug_stream('In calculateSpectrumParameters: copy')

        if sp.size != 1:
            # Start by median filtering to remove spikes
            m = np.median(np.vstack((sp[6:], sp[5:-1], sp[4:-2], sp[3:-3], sp[2:-4], sp[1:-5], sp[0:-6])), axis=0)
            noiseFloor = np.mean(m[0:10])
            peakCenterInd = m.argmax()
            halfMax = (m[peakCenterInd] + noiseFloor) / 2
            # Detect zero crossings to this half max to determine the FWHM
            halfInd = np.where(np.diff(np.sign(m - halfMax)))[0]
            halfIndReduced = halfInd[np.abs(halfInd - peakCenterInd).argsort()[0:2]]
            with self.streamLock:
                self.debug_stream('In calculateSpectrumParameters: halfInd done')
            # Check where the signal is below 1.2*noiseFloor:
            noiseInd = np.where(sp < 1.2 * noiseFloor)[0]
            if noiseInd.shape[0] < 3:
                noiseInd = np.array(1, sp.shape[0] - 1)
            # Index where the peak starts in the vector noiseInd:
            peakEdgeInd = abs(noiseInd - peakCenterInd).argmin()
            peakEdgeInd = max(peakEdgeInd, 1)
            peakEdgeInd = min(peakEdgeInd, noiseInd.shape[0] - 2)
            
            with self.streamLock:
                self.debug_stream('In calculateSpectrumParameters: peakInd done')
            # The peak is then located between [peakEdgeInd - 1] and [peakEdgeInd + 1]: 
            peakIndMin = max(noiseInd[peakEdgeInd - 1], 0)
            peakIndMax = min(noiseInd[peakEdgeInd + 1], noiseInd.shape[0] - 1)
            peakData = sp[peakIndMin : peakIndMax]
            
            with self.attrLock:
                with self.streamLock:
                    self.debug_stream('In calculateSpectrumParameters: peakData done')
                peakWavelengths = self.wavelengthsROI[peakIndMin : peakIndMax]
                try:
                    w = self.wavelengthsROI[3:-3]
                    spN = (m - noiseFloor * 1.1).clip(0)
                    peakEnergy = 1560 * 1e-6 * np.trapz(spN, w) / self.expTime
#                    peakEnergy = 1560 * 1e-6 * np.trapz(peakData, peakWavelengths) / self.expTime  # Integrate total intensity             
                    peakWidth = np.abs(np.diff(self.wavelengthsROI[halfIndReduced]))
#                    peakCenter = self.wavelengthsROI[peakCenterInd]
                    peakCenter = np.trapz(spN * w) / np.trapz(spN)
                except Exception, e:
                    with self.streamLock:
                        self.error_stream(''.join(('In calculateSpectrumParameters: Error calculating peak parameters: ', str(e))))
                    peakEnergy = 0.0
                    peakWidth = 0.0
                    peakCenter = 0.0
                with self.streamLock:
                    self.debug_stream('In calculateSpectrumParameters: peakCenter done')
                self.peakEnergy = peakEnergy
                self.peakWidth = peakWidth
                self.peakWavelength = peakCenter
            
                with self.streamLock:
                    self.info_stream(''.join(('In calculateSpectrumParameters: computations ', str(time.clock() - t0))))
    
        
    def checkCommands(self, blockTime=0):
        """Checks the commandQueue for new commands. Must be called regularly.
        If the queue is empty the method exits immediately.
        """
        with self.streamLock:
            self.debug_stream('Entering checkCommands')
        try:
            if blockTime == 0:
                with self.streamLock:
                    self.debug_stream('checkCommands: blockTime == 0')
                cmd = self.commandQueue.get(block=False)
            else:
                with self.streamLock:
                    self.debug_stream('checkCommands: blockTime != 0')
                cmd = self.commandQueue.get(block=True, timeout=blockTime)
            with self.streamLock:
                self.info_stream(str(cmd.command))
            if cmd.command == 'writeExposureTime':
                with self.attrLock:
                    self.expTime = cmd.data
                    attrName = ''.join(('Spectrometer', str(self.Serial), 'ExposureTime'))
                    self.masterDevice.write_attribute(attrName, cmd.data)
            elif cmd.command == 'readExposureTime':
                with self.attrLock:
                    attrName = ''.join(('Spectrometer', str(self.Serial), 'ExposureTime'))
                    attr = self.masterDevice.read_attribute(attrName)
                    self.expTime = attr.value                    
            elif cmd.command == 'writeUpdateTime':
                with self.attrLock:
                    self.updateTime = cmd.data
                    attrName = ''.join(('Spectrometer', str(self.Serial), 'UpdateTime'))
                    self.masterDevice.write_attribute(attrName, cmd.data)
            elif cmd.command == 'writePeakROI':
                with self.attrLock:
                    self.peakROI = cmd.data
                    roi1 = np.abs(self.wavelengths - self.peakROI[0]).argmin()
                    roi2 = np.abs(self.wavelengths - self.peakROI[1]).argmin()
                    self.peakROIIndex = np.array([min(roi1, roi2), max([roi1, roi2])])
                    self.wavelengthsROI = self.wavelengths[self.peakROIIndex[0] : self.peakROIIndex[1]]
                    self.spectrumROI = self.spectrum[self.peakROIIndex[0] : self.peakROIIndex[1]]                
            elif cmd.command == 'readUpdateTime':
                with self.attrLock:
                    attrName = ''.join(('Spectrometer', str(self.Serial), 'UpdateTime'))
                    attr = self.masterDevice.read_attribute(attrName)
                    self.updateTime = attr.value                    

            elif cmd.command == 'writeAutoExposure':
                self.autoExpose = cmd.data

            elif cmd.command == 'on' or cmd.command == 'start':           
                if self.get_state() not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.masterDevice.command_inout('StartSpectrometer', self.Serial)

            elif cmd.command == 'stop' or cmd.command == 'standby':
                if self.get_state() not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.masterDevice.command_inout('StopSpectrometer', self.Serial)
                
            elif cmd.command == 'off':
                if self.get_state() not in [PyTango.DevState.INIT, PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.OFF)       

            elif cmd.command == 'init':
                if self.get_state() not in [PyTango.DevState.UNKNOWN]:
                    self.setState(PyTango.DevState.UNKNOWN)
            
            elif cmd.command == 'readWavelengths':
                pass

        except Queue.Empty:
            with self.streamLock:
                self.debug_stream('checkCommands: queue empty')

            pass

#------------------------------------------------------------------
#     Always excuted hook method
#------------------------------------------------------------------
    def always_executed_hook(self):
        pass

#------------------------------------------------------------------
#     Wavelengths attribute
#------------------------------------------------------------------
    def read_Wavelengths(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading Wavelengths')))
        with self.attrLock:
            attr_read = self.wavelengths
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
            attr.set_value(attr_read, attr_read.shape[0])

    def is_Wavelengths_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     WavelengthsROI attribute
#------------------------------------------------------------------
    def read_WavelengthsROI(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading WavelengthsROI')))
        with self.attrLock:
            attr_read = self.wavelengthsROI
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
            attr.set_value(attr_read, attr_read.shape[0])

    def is_WavelengthsROI_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True


#------------------------------------------------------------------
#     Spectrum attribute
#------------------------------------------------------------------
    def read_Spectrum(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading Spectrum')))
        with self.attrLock:
            attr_read = np.copy(self.spectrum)
            with self.streamLock:
                self.debug_stream(''.join(('In read_Spectrum: attr_read')))
                self.debug_stream(''.join(('In read_Spectrum: attr_read type ', str(type(attr_read)))))
                self.debug_stream(''.join(('In read_Spectrum: attr_read shape ', str(attr_read.shape))))
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
                with self.streamLock:
                    self.debug_stream(''.join(('In read_Spectrum: attr_read==None')))
            attr.set_value(attr_read, attr_read.shape[0])
        with self.streamLock:
            self.info_stream(''.join(('exit read_Spectrum')))

    def is_Spectrum_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     SpectrumROI attribute
#------------------------------------------------------------------
    def read_SpectrumROI(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading SpectrumROI')))
        with self.attrLock:
            attr_read = self.spectrumROI
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
            attr.set_value(attr_read, attr_read.shape[0])

    def is_SpectrumROI_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     ExposureTime attribute
#------------------------------------------------------------------
    def read_ExposureTime(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading ExposureTime')))
        with self.attrLock:
            attr_read = self.expTime
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def write_ExposureTime(self, attr):        
        with self.streamLock:
            self.info_stream(''.join(('Writing ExposureTime')))
        data = attr.get_write_value()
        cmdMsg = SpectrometerCommand('writeExposureTime', data)
        self.commandQueue.put(cmdMsg)

    def is_ExposureTime_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     UpdateTime attribute
#------------------------------------------------------------------
    def read_UpdateTime(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading UpdateTime')))
        with self.attrLock:
            attr_read = self.updateTime
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def write_UpdateTime(self, attr):        
        with self.streamLock:
            self.info_stream(''.join(('Writing UpdateTime')))
        data = attr.get_write_value()
        cmdMsg = SpectrometerCommand('writeUpdateTime', data)
        self.commandQueue.put(cmdMsg)

    def is_UpdateTime_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     PeakROI attribute
#------------------------------------------------------------------
    def read_PeakROI(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading PeakROI')))
        with self.attrLock:
            attr_read = self.peakROI
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = np.array([0.0])
            attr.set_value(attr_read, attr_read.shape[0])

    def write_PeakROI(self, attr):        
        with self.streamLock:
            self.info_stream(''.join(('Writing PeakROI')))
        data = attr.get_write_value()
        cmdMsg = SpectrometerCommand('writePeakROI', data)
        self.commandQueue.put(cmdMsg)

    def is_PeakROI_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     PeakEnergy attribute
#------------------------------------------------------------------
    def read_PeakEnergy(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading PeakEnergy')))
        t0 = time.clock()
        with self.attrLock:
            attr_read = self.peakEnergy
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)
        with self.streamLock:
            self.debug_stream(''.join(('In read_PeakEnergy: response time ', str(time.clock() - t0))))

    def is_PeakEnergy_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     PeakWidth attribute
#------------------------------------------------------------------
    def read_PeakWidth(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading PeakWidth')))
        with self.attrLock:
            attr_read = self.peakWidth
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def is_PeakWidth_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     PeakWavelength attribute
#------------------------------------------------------------------
    def read_PeakWavelength(self, attr):
        with self.streamLock:
            self.info_stream(''.join(('Reading PeakWavelength')))
        with self.attrLock:
            attr_read = self.peakWavelength
            if attr_read == None:
                attr.set_quality(PyTango.AttrQuality.ATTR_INVALID)
                attr_read = 0.0
            attr.set_value(attr_read)

    def is_PeakWavelength_allowed(self, req_type):
        if self.get_state() in [PyTango.DevState.INIT,
                                PyTango.DevState.UNKNOWN]:
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
#     On command:
#
#     Description: Start acquiring on spectrometer  
#                
#------------------------------------------------------------------
    def On(self):
        with self.streamLock:
            self.info_stream(''.join(("In ", self.get_name(), "::On")))
        cmdMsg = SpectrometerCommand('on')
        self.commandQueue.put(cmdMsg)

#---- StartSpectrometer command State Machine -----------------
    def is_On_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True

#------------------------------------------------------------------
#     Stop command:
#
#     Description: Start acquiring on spectrometer  
#                
#------------------------------------------------------------------
    def Stop(self):
        with self.streamLock:
            self.info_stream(''.join(("In ", self.get_name(), "::Stop")))
        cmdMsg = SpectrometerCommand('stop')
        self.commandQueue.put(cmdMsg)

#---- StartSpectrometer command State Machine -----------------
    def is_Stop_allowed(self):
        if self.get_state() in [PyTango.DevState.UNKNOWN]:
            #     End of Generated Code
            #     Re-Start of Generated Code
            return False
        return True
    
#==================================================================
#
#     SPM002SpectrometerDSClass class definition
#
#==================================================================
class SPM002SpectrometerDSClass(PyTango.DeviceClass):

    #     Class Properties
    class_property_list = {
        }


    #     Device Properties
    device_property_list = {
        'Serial':
            [PyTango.DevLong,
            "Serial number of the spectrometer",
            [ 10107889 ] ],
        'Master':
            [PyTango.DevString,
            "Tango device name of the master device",
            [ 'gunlaser/devices/spm002' ] ],
        }


    #     Command definitions
    cmd_list = {
        'On':
            [[PyTango.DevVoid, ""],
            [PyTango.DevVoid, ""]],
        'Stop':
            [[PyTango.DevVoid, ""],
            [PyTango.DevVoid, ""]],
        }


    #     Attribute definitions
    attr_list = {   
        'ExposureTime':
            [[PyTango.DevDouble,
              PyTango.SCALAR,
              PyTango.READ_WRITE],
                    {
                        'description':"Exposure time in ms",
                        'Memorized':"false",
                        'unit': 'ms',
                    } ],
        'UpdateTime':
            [[PyTango.DevDouble,
              PyTango.SCALAR,
              PyTango.READ_WRITE],
                    {
                        'description':"Time between spectrum updates in ms",
                        'Memorized':"false",
                        'unit': 'ms',
                    } ],
        'Spectrum':
            [[PyTango.DevDouble,
            PyTango.SPECTRUM,
            PyTango.READ, 3648],
             {
                'description': "Spectrum trace",
                'unit': 'a.u.'
             }],
        'SpectrumROI':
            [[PyTango.DevDouble,
            PyTango.SPECTRUM,
            PyTango.READ, 3648],
             {
                'description': "Spectrum trace in the region of interest",
                'unit': 'a.u.'
             }],
        'Wavelengths':
            [[PyTango.DevDouble,
            PyTango.SPECTRUM,
            PyTango.READ, 3648],
             {
                'description': "Wavelength table",
                'unit': 'nm'
             }],
        'WavelengthsROI':
            [[PyTango.DevDouble,
            PyTango.SPECTRUM,
            PyTango.READ, 3648],
             {
                'description': "Wavelength table in the region of interest",
                'unit': 'nm'
             }],
        'PeakROI':
            [[PyTango.DevDouble,
            PyTango.SPECTRUM,
            PyTango.READ_WRITE, 2],
            {
                'unit':"nm",
                'description': "Region of interest for peak calculations [lambda_min, lambda_max]",
            } ],
        'PeakWavelength':
            [[PyTango.DevDouble,
            PyTango.SCALAR,
            PyTango.READ],
            {
                'unit':"nm",
                'description': "Center wavelength of the peak",
            } ],
        'PeakWidth':
            [[PyTango.DevDouble,
            PyTango.SCALAR,
            PyTango.READ],
            {
                'unit':"nm",
                'description':"FWHM width of the spectrum at the peak.",
            } ],
        'PeakEnergy':
            [[PyTango.DevDouble,
            PyTango.SCALAR,
            PyTango.READ],
            {
                'description':"Energy inside the main peak",
                'unit':'counts*m/s'
            } ],
        }


#------------------------------------------------------------------
#     SPM002SpectrometerDSClass Constructor
#------------------------------------------------------------------
    def __init__(self, name):
        PyTango.DeviceClass.__init__(self, name)
        self.set_type(name);
        print "In SPM002SpectrometerDSClass  constructor"

#==================================================================
#
#     SPM002SpectrometerDS class main method
#
#==================================================================
if __name__ == '__main__':
    try:
        py = PyTango.Util(sys.argv)
        py.add_class(SPM002SpectrometerDSClass, SPM002SpectrometerDS, 'SPM002SpectrometerDS')

        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

    except PyTango.DevFailed, e:
        print '-------> Received a DevFailed exception:', e
    except Exception, e:
        print '-------> An unforeseen exception occured....', e
