# -*- coding:utf-8 -*-
"""
Created on Jun 11, 2012

Updated on Oct 09, 2023

Code to control a photoncontrol SPM002 spectrometer using the dll photonspectr.dll.
The device is controlled via ctypes structures passed as reference to the dll methods.
The dll is 32-bit so a 32-bit python distribution is required.

The wavelength vector is constructed from the lookup table polynomial via the
construct_wavelengths method.

@author: Filip Lindau
"""
from ctypes import *
from struct import *
import numpy as np
import threading
import logging
import time
import atexit


log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"

# Set up the logging configuration using basicConfig
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger()

logger.debug("Loading dll")
spmlib = cdll.LoadLibrary("./PhotonSpectr.dll")


class SpectrometerError(Exception):
    pass


class SPM002Control:
    def __init__(self, populate_devices=True):
        """
        Create SPM002 control device.
        If populate_devices is True the usb bus is checked for spectrometers immediately.
        """
        self.device_list = []
        self.serial_list = []
        self.device_index = None
        self.spectrum = np.zeros(3648)
        self.spectrum = self.spectrum.astype(np.uint16)
        self.spectrum_pointer = self.spectrum.ctypes.data_as(POINTER(c_uint16))
        self.start_pixel = 0
        self.end_pixel = 3647
        self.num_pixels = 3648
        self.hw_lock = threading.Lock()
        self.data_lock = threading.Lock()

        self.lut = None
        self.wavelengths = np.zeros(self.num_pixels)
        if populate_devices:
            self.populate_device_list()
        
    def populate_device_list(self):
        logger.info("Populating device list")
        index_tmp = None
        if self.device_index is not None:
            index_tmp = self.device_index
            self.close_device()
        self.device_list = []
        self.serial_list = []
        with self.hw_lock:
            num_devices = spmlib.PHO_EnumerateDevices()
            logger.info(f"Found {num_devices} devices:")
            sb = create_string_buffer(10)
            for k in range(num_devices):
                index = c_int(k)
                status = spmlib.PHO_Open(index)
                if status != 1:
                    raise SpectrometerError(''.join(('Error spectrometer ', str(index), ': ', str(status))))
                status = spmlib.PHO_GetSn(index, sb, 9)
                if status != 1:
                    raise SpectrometerError(''.join(('Error getting spectrometer ', str(index), ' serial number: ', str(status))))
                serial = int(sb.value)
                self.device_list.append(index.value)
                self.serial_list.append(serial)
                spmlib.PHO_Close(index)
        dev_str = "\n".join([f"Index {self.device_list[ind]}: serial {serial}" for ind, serial in enumerate(self.serial_list)])
        logger.info(f"Device list:\n{dev_str}")
        if index_tmp is not None:
            self.open_device(index_tmp)
            
    def open_device(self, index):
        """
        Open a spectrometer device based on its index in the enumerated list of attached devices.
        """
        logger.info(f"Opening device with index {index}")
        if self.device_index is not None:
            self.close_device()
        with self.hw_lock:
            result = spmlib.PHO_Open(c_int(index))
            if result != 1:
                raise SpectrometerError(''.join(('Could not open device, returned ', str(result))))
            self.device_index = index

    def open_device_index(self, index):
        self.open_device(index)
        
    def open_device_serial(self, serial):
        """
        Open a spectrometer device based on its serial number. If the serial is not found in the list of attached
        devices an error is raised.
        """
        with self.hw_lock:
            try:
                index = self.serial_list.index(serial)
    #            print 'Opening device', index
            except ValueError:
                raise SpectrometerError(f"No device {serial} found in list of connected spectrometers.")
        self.open_device(index)
            
    def close_device(self):
        logger.info(f"Closing device with index {self.device_index}")
        with self.hw_lock:
            if self.device_index is not None:
                result = spmlib.PHO_Close(c_int(self.device_index))
                if result != 1:
                    raise SpectrometerError(''.join(('Could not close device, returned ', str(result))))
                self.device_index = None

    def get_start_end_pixels(self):
        logger.debug("Retrieving start and end pixel values from device")
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            end_p = c_int()
            start_p = c_int()
            num_pixels = c_int()
            result = spmlib.PHO_GetStartEnd(self.device_index, byref(end_p), byref(start_p))
            if result != 1:
                raise SpectrometerError(''.join(('Could not read start and end pixels, returned ', str(result))))
            self.start_pixel = start_p.value
            self.end_pixel = end_p.value
            result = spmlib.PHO_GetPn(self.device_index, byref(num_pixels))
            if result != 1:
                raise SpectrometerError(''.join(('Could not read number of pixels, returned ', str(result))))
            self.num_pixels = num_pixels.value
            
    def get_model(self):
        logger.debug("Retrieving device model")
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            ml = create_string_buffer(13)
            result = spmlib.PHO_GetMl(self.device_index, ml, 13)
            if result != 1:
                raise SpectrometerError(''.join(('Could not get model, returned ', str(result))))
        return ml.value
        
    def get_mode(self):
        logger.debug("Retrieving device capture mode")
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            mode = c_int()
            result = spmlib.PHO_GetMode(self.device_index, byref(mode))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
        return mode.value
    
    def set_mode(self, mode):
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            result = spmlib.PHO_SetMode(self.device_index, c_int(mode))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
    
    def get_temperature(self):
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            temp = c_float()
            result = spmlib.PHO_GetMode(self.device_index, byref(temp))
            if result != 1:
                raise SpectrometerError(''.join(('Could not set mode, returned ', str(result))))
        return temp.value
    
    def get_lut(self):
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            lut = np.zeros(4)
            lut = lut.astype(np.float32)
            lut_ct = lut.ctypes.data_as(POINTER(c_float))
            result = spmlib.PHO_GetLut(self.device_index, lut_ct, 4)
            if result != 1:
                raise SpectrometerError(''.join(('Could not get LUT, returned ', str(result))))
        with self.data_lock:
            self.lut = lut
        return self.lut
        
    def construct_wavelengths(self):
        if self.lut is None:
            self.get_lut()
        if self.lut is not None:
            x = np.arange(self.wavelengths.shape[0], dtype=np.float64)
            w = self.lut[0] + self.lut[1] * x + self.lut[2] * x ** 2 + self.lut[3] * x ** 3
            with self.data_lock:
                self.wavelengths = w

    def acquire_spectrum(self):
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        t0 = time.time()
        with self.hw_lock:
            spectrum = np.zeros(self.num_pixels, dtype=np.uint16)
            spectrum_pointer = spectrum.ctypes.data_as(POINTER(c_uint16))
            result = spmlib.PHO_Acquire(self.device_index, 0, self.num_pixels, spectrum_pointer)
#            result = spmlib.PHO_Acquire(self.deviceIndex, self.startP, self.numPixels, self.CCD_ct)
            if result != 1:
                raise SpectrometerError(''.join(('Could not acquire spectrum, returned ', str(result))))
            logger.debug(f"Spectrum acquired in {(time.time() - t0)*1000:.1f} ms")
        with self.data_lock:
            self.spectrum = spectrum

    def get_exposure_time(self):
        """
        Get spectrometer exposure time in ms."
        """
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            exposure = c_float()
            result = spmlib.PHO_GetTime(self.device_index, byref(exposure))
            if result != 1:
                raise SpectrometerError(''.join(('Could not get exposure time, returned ', str(result))))
        return exposure.value
        
    def set_exposure_time(self, exposure):
        """
        Set spectrometer exposure time in ms.
        """
        logger.info(f"Setting exposure time {exposure:.1e} ms")
        if self.device_index is None:
            raise SpectrometerError("Not connected to device")
        with self.hw_lock:
            exposure_time_ct = c_float(exposure)
            result = spmlib.PHO_SetTime(self.device_index, exposure_time_ct)
            if result != 1:
                raise SpectrometerError(''.join(('Could not set exposure time, returned ', str(result))))

    def get_wavelengths(self):
        with self.data_lock:
            w = np.copy(self.wavelengths)
        return w

    def get_spectrum(self):
        with self.data_lock:
            s = np.copy(self.spectrum)
        return s

            
if __name__ == '__main__':
    spm = SPM002Control()
    # spm.open_device(0)
    # spm.construct_wavelengths()
    # spm.acquire_spectrum()
    # import matplotlib.pyplot as mpl
    # mpl.plot(spm.get_wavelengths(), spm.get_spectrum())
