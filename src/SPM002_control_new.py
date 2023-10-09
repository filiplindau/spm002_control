# -*- coding:utf-8 -*-
"""
Created on Jun 11, 2012

@author: Laser
"""
from ctypes import *
from struct import *
import numpy as np
import time
import atexit

spmlib = cdll.LoadLibrary("PhotonSpectr.dll")


class SpectrometerError(Exception):
    pass


class SPM002control:
    def __init__(self):
        self.deviceList = []
        self.serialList = []
        self.deviceIndex = None
        self.CCD = np.zeros(3648)
        self.CCD = self.CCD.astype(np.uint16)
        self.CCD_ct = self.CCD.ctypes.data_as(POINTER(c_uint16))
        self.startP = 0
        self.endP = 3647
        self.numPixels = 3648

        self.LUT = None
        self.wavelengths = np.zeros(3648)
        
    def populateDeviceList(self):
        indexTmp = None
        if self.deviceIndex is not None:
            indexTmp = self.deviceIndex
            self.closeDevice()
        self.deviceList = []
        self.serialList = []
        numDevices = spmlib.PHO_EnumerateDevices()
        sb = create_string_buffer(10)
        for k in range(numDevices):
            index = c_int(k)
            status = spmlib.PHO_Open(index)
            if status != 1:
                raise SpectrometerError(''.join(('Error spectrometer ', str(index), ': ', str(status))))
            status = spmlib.PHO_GetSn(index, sb, 9)
            if status != 1:
                raise SpectrometerError(''.join(('Error getting spectrometer ', str(index), ' serial number: ', str(status))))
            serial = int(sb.value)
            self.deviceList.append(index.value)
            self.serialList.append(serial)
            spmlib.PHO_Close(index)
        if indexTmp is not None:
            self.openDevice(indexTmp)
            
    def openDevice(self, index):
        if self.deviceIndex is not None:
            self.closeDevice()
        result = spmlib.PHO_Open(c_int(index))
        if result != 1:
            raise SpectrometerError(''.join(('Could not open device, returned ', str(result))))
        self.deviceIndex = index

    def openDeviceIndex(self, index):
        self.openDevice(index)
        
    def openDeviceSerial(self, serial):
        try:
            index = self.serialList.index(serial)
#            print 'Opening device', index
        except ValueError:
            raise SpectrometerError(''.join(('No device ', str(serial), ' found in list of connected spectrometers.')))
        self.openDevice(index)
            
    def closeDevice(self):
        if self.deviceIndex is not None:
            result = spmlib.PHO_Close(c_int(self.deviceIndex))
            if result != 1:
                raise SpectrometerError(''.join(('Could not close device, returned ', str(result))))
            self.deviceIndex = None

    def getStartEndPixels(self):
        if self.deviceIndex is not None:
            endP = c_int()
            startP = c_int()
            numPixels = c_int()
            result = spmlib.PHO_GetStartEnd(self.deviceIndex, byref(endP), byref(startP))
            if result != 1:
                raise SpectrometerError(''.join(('Could not read start and end pixels, returned ', str(result))))
            self.startP = startP.value
            self.endP = endP.value
            result = spmlib.PHO_GetPn(self.deviceIndex, byref(numPixels))
            if result != 1:
                raise SpectrometerError(''.join(('Could not read number of pixels, returned ', str(result))))
            self.numPixels = numPixels.value
            
    def getModel(self):
        if self.deviceIndex is not None:
            ml = create_string_buffer(13)
            result = spmlib.PHO_GetMl(self.deviceIndex, ml, 13)
            if result != 1:
                raise SpectrometerError(''.join(('Could not get model, returned ', str(result))))
            return ml.value
        
    def getMode(self):
        if self.deviceIndex is not None:
            mode = c_int()            
            result = spmlib.PHO_GetMode(self.deviceIndex, byref(mode))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
            return mode.value
    
    def setMode(self, mode):
        if self.deviceIndex is not None:
            result = spmlib.PHO_SetMode(self.deviceIndex, c_int(mode))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
    
    def getTemperature(self):
        if self.deviceIndex is not None:
            temp = c_float()            
            result = spmlib.PHO_GetMode(self.deviceIndex, byref(temp))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
            return temp.value
    
    def getLUT(self):
        if self.deviceIndex is not None:
            LUT = np.zeros(4)
            LUT = LUT.astype(np.float32)
            LUT_ct = LUT.ctypes.data_as(POINTER(c_float))
            result = spmlib.PHO_GetLut(self.deviceIndex, LUT_ct, 4)
            if result != 1:
                raise SpectrometerError(''.join(('Could not get LUT, returned ', str(result))))
            
            self.LUT = LUT
        
    def constructWavelengths(self):
        if self.LUT is None:
            self.getLUT()
        if self.LUT is not None:
            x = np.arange(self.wavelengths.shape[0], dtype=np.float64)
            w = self.LUT[0] + self.LUT[1] * x + self.LUT[2] * x ** 2 + self.LUT[3] * x ** 3
            self.wavelengths = w

    def acquireSpectrum(self):
        if self.deviceIndex is not None:
            result = spmlib.PHO_Acquire(self.deviceIndex, 0, self.numPixels, self.CCD_ct)
#            result = spmlib.PHO_Acquire(self.deviceIndex, self.startP, self.numPixels, self.CCD_ct)
            if result != 1:
                raise SpectrometerError(''.join(('Could not acquire spectrum, returned ', str(result))))

    def getExposureTime(self):
        if self.deviceIndex is not None:
            exposure = c_float()
            result = spmlib.PHO_GetTime(self.deviceIndex, byref(exposure))
            if result != 1:
                raise SpectrometerError(''.join(('Could not get exposure time, returned ', str(result))))
            return exposure.value
        
    def setExposureTime(self, exposure):
        if self.deviceIndex is not None:
            time = c_float(exposure)
            result = spmlib.PHO_SetTime(self.deviceIndex, time)
            if result != 1:
                raise SpectrometerError(''.join(('Could not set exposure time, returned ', str(result))))

            
if __name__ == '__main__':
    spm = SPM002control()    
#    spm.populateDeviceList()
#    spm.openDevice(1)
#     spm.constructWavelengths()
#     spm.acquireSpectrum()
#     import matplotlib.pyplot as mpl
#     mpl.plot(spm.wavelengths, spm.CCD)
