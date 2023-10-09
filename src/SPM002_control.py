# -*- coding:utf-8 -*-
"""
Created on Jun 11, 2012

@author: Laser
"""
from ctypes import *
from struct import *
import numpy as np
import time
# import atexit

spmlib = windll.LoadLibrary("SPM002.dll")

class SpectrometerError(Exception):
    pass

class SPM002control():
    def __init__(self):
        self.deviceList = []
        self.serialList = []
        self.deviceHandle = None
        self.deviceIndex = None
        self.CCD = np.zeros(3648)
        self.CCD = self.CCD.astype(np.uint16)
        self.CCD_ct = self.CCD.ctypes.data_as(POINTER(c_uint16))

        self.LUT = None
        self.wavelengths = np.zeros(3648)
        
    def populateDeviceList(self):
        indexTmp = None
        if self.deviceHandle != None:
            indexTmp = self.deviceIndex
            self.closeDevice()
        self.deviceList = []
        self.serialList = []
        for k in range(15):
            # It seems the first index is not valid... start with 1
            index = c_int(k + 1)
            handle = spmlib.PHO_Open(index)
            # All spectrometers seem to come in sequence, so we can break on the first
            # zero we detect
            if handle == 0:
                break
            serial = spmlib.PHO_Getsn(handle)
            self.deviceList.append(handle)
            self.serialList.append(serial)
            spmlib.PHO_Close(handle)
        if indexTmp != None:
            self.openDevice(indexTmp)
            
    def openDeviceSerial(self, serial):
        if self.deviceHandle != None:
            self.closeDevice()
        try:
            index = self.serialList.index(serial) + 1
#            print 'Opening device', index
        except ValueError:
            raise SpectrometerError(''.join(('No device ', str(serial), ' found in list of connected spectrometers.')))
        self.deviceHandle = spmlib.PHO_Open(c_int(index))
#        print 'Handle: ', self.deviceHandle
        if self.deviceHandle == 0:
            self.deviceHandle = None
            raise SpectrometerError('Error opening spectrometer')
        self.deviceIndex = index

    def openDeviceIndex(self, index):
        if self.deviceHandle != None:
            self.closeDevice()
        self.deviceHandle = spmlib.PHO_Open(c_int(index + 1))
        if self.deviceHandle == 0:
            raise SpectrometerError('Error opening spectrometer')
        self.deviceIndex = index
            
    def closeDevice(self):
        if self.deviceHandle != None:
            result = spmlib.PHO_Close(self.deviceHandle)
            if result != 0:
                raise SpectrometerError(''.join(('Could not close device, returned ', str(result))))
            self.deviceHandle = None
            self.deviceIndex = None
            
    def getSerial(self):
        if self.deviceIndex != None:
            serial = spmlib.PHO_Getsn(self.deviceHandle)
            return serial
        
    def getExposureTime(self):
        if self.deviceIndex != None:
            exposure = spmlib.PHO_Gettime(self.deviceHandle)
            return exposure
        
    def setExposureTime(self, exposure):
        if self.deviceIndex != None:
            time = c_int(exposure)
            result = spmlib.PHO_Settime(self.deviceHandle, time)
            if result != 0:
                raise SpectrometerError(''.join(('Could not set exposure time, returned ', str(result))))
            
    def getLUT(self):
        if self.deviceIndex != None:
            LUT = np.zeros(4)
            LUT = LUT.astype(np.float32)
            LUT_ct = LUT.ctypes.data_as(POINTER(c_float))
            result = spmlib.PHO_Getlut(self.deviceHandle, LUT_ct)
            # Unknown meaning of this result value...
#            if result != 0:
#                raise SpectrometerError(''.join(('Could not get LUT, returned ', str(result))))
            
            self.LUT = LUT
        
    def constructWavelengths(self):
        if self.LUT == None:
            self.getLUT()
        if self.LUT != None:
            x = np.arange(self.wavelengths.shape[0], dtype=np.float64)
            w = self.LUT[0] + self.LUT[1] * x + self.LUT[2] * x ** 2 + self.LUT[3] * x ** 3
            self.wavelengths = w
            
    def acquireSpectrum(self):
        if self.deviceIndex != None:
            result = spmlib.PHO_Acquire(self.deviceHandle, self.CCD_ct)
            # Unknown meaning of this result value
#            if result != 0:
#                raise SpectrometerError(''.join(('Could not acquire spectrum, returned ', str(result))))

        
        
  
if __name__ == '__main__':
    spm = SPM002control()    
    spm.populateDeviceList()
#    spm.openDevice(1)  
