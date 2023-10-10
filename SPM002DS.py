"""
Tango device server to control an SPM002 spectrometer via the SPM002_control python code.
The control code use the PhotonSpectr.dll which is 32-bit, as such the device server must
also be run in a 32-bit python environment.

:created: 2023-10-10
:author: Filip Lindau <filip.lindau@maxiv.lu.se>
"""

import threading
import time

import PyTango as pt
import numpy as np
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import device_property
# from scipy.interpolate import interp1d
import SPM002_control as spm


class SPM002DS(Device):
    __metaclass__ = DeviceMeta

    ExposureTime = attribute(label="Exposure time",
                             dtype=float,
                             access=pt.AttrWriteType.READ_WRITE,
                             unit="ms",
                             format="%6.3f",
                             min_value=0.0,
                             max_value=10000.0,
                             fget="get_exposuretime",
                             fset="set_exposuretime",
                             doc="Exposure time in ms",
                             memorized=False,
                             hw_memorized=False)

    UpdateTime = attribute(label="Update time",
                           dtype=float,
                           access=pt.AttrWriteType.READ_WRITE,
                           unit="ms",
                           format="%6.3f",
                           min_value=0.0,
                           # max_value=10000.0,
                           fget="get_updatetime",
                           fset="set_updatetime",
                           doc="Time between spectrum updates in ms",
                           memorized=True,
                           hw_memorized=False)

    Spectrum = attribute(label="Spectrum",
                         dtype=(int,),
                         access=pt.AttrWriteType.READ,
                         min_value=0,
                         max_dim_x=3648,
                         fget="get_spectrum",
                         doc="Spectrum trace")

    Wavelengths = attribute(label="Wavelengths",
                            dtype=(float,),
                            access=pt.AttrWriteType.READ,
                            max_dim_x=3648,
                            fget="get_wavelengths",
                            unit="nm",
                            doc="Wavelength array")

    Model = attribute(label="Model",
                            dtype=str,
                            access=pt.AttrWriteType.READ,
                            fget="get_model",
                            doc="Spectrometer model")

    Serial = device_property(dtype=int,
                             doc="Spectrometer serial number")

    def __init__(self, klass, name):
        self.spectrum_data = None
        self.wavelengths_data = None
        self.exposuretime_data = None
        self.updatetime_data = None
        self.model_data = None

        self.spectrometer_dev: spm.SPM002Control = None

        self.thread = None
        self.stop_thread_flag = False
        self.attr_lock = threading.Lock()

        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)

        self.set_state(pt.DevState.UNKNOWN)

        self.spectrum_data = None
        self.wavelengths_data = None
        self.exposuretime_data = None
        self.updatetime_data = 100.0

        # Close acquire thread if running
        self.stop_thread()

        # Close device
        if self.spectrometer_dev is not None:
            try:
                self.spectrometer_dev.close_device()
            except spm.SpectrometerError:
                pass

        # Create device instance
        try:
            self.info_stream("Creating device instance")
            self.spectrometer_dev = spm.SPM002Control()
        except spm.SpectrometerError:
            return

        # Connect to device
        try:
            self.info_stream(f"Found {len(self.spectrometer_dev.serial_list)} devices. Connecting to {self.Serial}")
            if self.Serial in self.spectrometer_dev.serial_list:
                self.spectrometer_dev.open_device_serial(self.Serial)
            else:
                self.error_stream(f"Serial {self.Serial} not found in {self.spectrometer_dev.serial_list}")
                self.set_status(f"Serial {self.Serial} not found in {self.spectrometer_dev.serial_list}")
                return
        except spm.SpectrometerError as e:
            self.error_stream(f"Could not connect to spectrometer. Error {e}")
            self.set_status(f"Could not connect to spectrometer. Error {e}")
            return

        self.set_state(pt.DevState.INIT)
        try:
            self.spectrometer_dev.construct_wavelengths()
            self.wavelengths_data = self.spectrometer_dev.get_wavelengths()
        except spm.SpectrometerError as e:
            self.error_stream(f"Could not construct wavelengths. Error {e}")
            self.set_status(f"Could not construct wavelengths. Error {e}")
            return

        try:
            self.exposuretime_data = self.spectrometer_dev.get_exposure_time()
        except spm.SpectrometerError as e:
            self.error_stream(f"Could not get exposure time. Error {e}")
            self.set_status(f"Could not get exposure time. Error {e}")
            return

        try:
            self.model_data = self.spectrometer_dev.get_model()
        except spm.SpectrometerError as e:
            self.error_stream(f"Could not get model. Error {e}")
            self.set_status(f"Could not get model. Error {e}")
            return

        self.set_state(pt.DevState.ON)
        self.set_status("ON\n\nNot grabbing spectra")

    def get_spectrum(self):
        with self.attr_lock:
            spectrum_data = np.copy(self.spectrum_data)
        return spectrum_data

    def get_wavelengths(self):
        return self.wavelengths_data

    def set_exposuretime(self, value):
        self.spectrometer_dev.set_exposure_time(value)
        self.exposuretime_data = value

    def get_exposuretime(self):
        return self.exposuretime_data

    def set_updatetime(self, value):
        self.updatetime_data = value

    def get_updatetime(self):
        return self.updatetime_data

    def get_model(self):
        return self.model_data

    def update_spectrum(self):
        last_update_time = time.time()
        self.spectrometer_dev.acquire_spectrum()
        while not self.stop_thread_flag:
            t = time.time()
            if t - last_update_time > self.updatetime_data * 0.001:
                with self.attr_lock:
                    self.spectrum_data = self.spectrometer_dev.get_spectrum()
                    self.spectrometer_dev.acquire_spectrum()
                last_update_time = time.time()
            time.sleep(0.02)

    def stop_thread(self):
        if self.thread is not None:
            self.stop_thread_flag = True
            self.thread.join(3)
        self.thread = None
        self.stop_thread_flag = False

    @command
    def On(self):
        self.info_stream("Starting aqcuisition thread")
        self.stop_thread()
        self.thread = threading.Thread(target=self.update_spectrum)
        self.thread.start()
        self.set_state(pt.DevState.RUNNING)
        self.set_status("RUNNING\n\nGrabbing spectra")

    @command
    def Stop(self):
        self.info_stream("Stopping acquistition thread")
        self.stop_thread()
        self.set_state(pt.DevState.ON)
        self.set_status("ON\n\nNot grabbing spectra")


if __name__ == "__main__":
    pt.server.server_run((SPM002DS,))