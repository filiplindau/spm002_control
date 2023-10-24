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

    # PeakROI = attribute(label="Peak ROI",
    #                     dtype=(float,),
    #                     access=pt.AttrWriteType.READ_WRITE,
    #                     max_dim_x=2,
    #                     fget="get_peakroi",
    #                     fset="set_peakroi",
    #                     unit="nm",
    #                     doc="Region of interest for peak calculations [lambda_min, lambda_max]")

    SpectrumROI = attribute(label="Spectrum ROI",
                            dtype=(int,),
                            access=pt.AttrWriteType.READ,
                            min_value=0,
                            max_dim_x=3648,
                            fget="get_spectrum_roi",
                            doc="Spectrum trace in ROI")

    WavelengthsROI = attribute(label="Wavelengths ROI",
                               dtype=(float,),
                               access=pt.AttrWriteType.READ,
                               max_dim_x=3648,
                               fget="get_wavelengths_roi",
                               unit="nm",
                               doc="Wavelength array in ROI")

    PeakWavelength = attribute(label="Peak wavelength",
                               dtype=float,
                               access=pt.AttrWriteType.READ,
                               unit="nm",
                               fget="get_peakwavelength",
                               doc="Wavelength of the most prominent peak found in the ROI")

    PeakWidth = attribute(label="Peak width",
                          dtype=float,
                          access=pt.AttrWriteType.READ,
                          unit="nm",
                          fget="get_peakwidth",
                          doc="FWHM width of the most prominent peak found in the ROI")

    PeakEnergy = attribute(label="Peak energy",
                           dtype=float,
                           access=pt.AttrWriteType.READ,
                           unit="a.u.",
                           fget="get_peakenergy",
                           doc="Energy inside the most prominent peak found in the ROI")

    Model = attribute(label="Model",
                      dtype=str,
                      access=pt.AttrWriteType.READ,
                      fget="get_model",
                      doc="Spectrometer model")

    Serial = device_property(dtype=int,
                             doc="Spectrometer serial number")

    PeakROI_min = device_property(dtype=float,
                                  doc="Region of interest for peak calculations lambda_min",
                                  default_value=700)

    PeakROI_max = device_property(dtype=float,
                                  doc="Region of interest for peak calculations lambda_max",
                                  default_value=900)

    def __init__(self, klass, name):
        self.spectrum_data = None
        self.wavelengths_data = None
        self.exposuretime_data = None
        self.updatetime_data = None
        self.model_data = None
        self.peak_width_data = None
        self.peak_energy_data = None
        self.peak_wavelength_data = None
        self.spectrum_roi_data = None
        self.wavelengths_roi_data = None
        self.peak_roi_data = None
        self.peak_roi_index = np.array([0, 3647])

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
        self.set_peakroi((self.PeakROI_min, self.PeakROI_max))

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

    def get_spectrum_roi(self):
        with self.attr_lock:
            spectrum_data = np.copy(self.spectrum_roi_data)
        return spectrum_data

    def get_wavelengths_roi(self):
        return self.wavelengths_roi_data

    def set_exposuretime(self, value):
        self.spectrometer_dev.set_exposure_time(value)
        self.exposuretime_data = value

    def get_exposuretime(self):
        return self.exposuretime_data

    def set_updatetime(self, value):
        self.updatetime_data = value

    def get_updatetime(self):
        return self.updatetime_data

    def set_peakroi(self, value):
        self.info_stream(f"Setting PeakROI to {value}")
        with self.attr_lock:
            self.peak_roi_data = value
            if self.wavelengths_data is None:
                return
            roi1 = np.abs(self.wavelengths_data - self.peak_roi_data[0]).argmin()
            roi2 = np.abs(self.wavelengths_data - self.peak_roi_data[1]).argmin()
            self.peak_roi_index = np.array([min(roi1, roi2), max([roi1, roi2])])
            try:
                self.wavelengths_roi_data = self.wavelengths_data[self.peak_roi_index[0]: self.peak_roi_index[1]]
                self.spectrum_roi_data = self.spectrum_data[self.peak_roi_index[0]: self.peak_roi_index[1]]
            except TypeError as e:
                self.error_stream(f"Could not set roi data: {e}")

    def get_peakroi(self):
        return self.peak_roi_data

    def get_model(self):
        return self.model_data

    def get_peakwavelength(self):
        return self.peak_wavelength_data

    def get_peakwidth(self):
        return self.peak_width_data

    def get_peakenergy(self):
        return self.peak_energy_data

    def update_spectrum(self):
        last_update_time = time.time()
        self.spectrometer_dev.acquire_spectrum()
        while not self.stop_thread_flag:
            t = time.time()
            if t - last_update_time > self.updatetime_data * 0.001:
                with self.attr_lock:
                    self.spectrum_data = self.spectrometer_dev.get_spectrum()
                    self.spectrometer_dev.acquire_spectrum()
                    self.spectrum_roi_data = self.spectrum_data[self.peak_roi_index[0]: self.peak_roi_index[1]]
                self.calculate_spectrum_parameters()
                last_update_time = time.time()
            time.sleep(0.02)

    def calculate_spectrum_parameters(self):
        self.debug_stream('Entering calculate_spectrum_parameters')
        t0 = time.time()
        if self.spectrum_roi_data is None:
            return


        with self.attr_lock:
            sp = np.copy(self.spectrum_roi_data)

        if sp.size != 1:
            # Start by median filtering to remove spikes
            m = np.median(np.vstack((sp[6:], sp[5:-1], sp[4:-2], sp[3:-3], sp[2:-4], sp[1:-5], sp[0:-6])), axis=0)
            noise_floor = np.mean(m[0:10])
            peak_center_ind = m.argmax()
            half_max = (m[peak_center_ind] + noise_floor) / 2
            # Detect zero crossings to this half max to determine the FWHM
            half_ind = np.where(np.diff(np.sign(m - half_max)))[0]
            half_ind_reduced = half_ind[np.abs(half_ind - peak_center_ind).argsort()[0:2]]
            self.debug_stream('In calculate_spectrum_parameters: halfInd done')
            # Check where the signal is below 1.2*noiseFloor:
            noise_ind = np.where(sp < 1.2 * noise_floor)[0]
            if noise_ind.shape[0] < 3:
                noise_ind = np.array(1, sp.shape[0] - 1)
            # Index where the peak starts in the vector noiseInd:
            peak_edge_ind = abs(noise_ind - peak_center_ind).argmin()
            peak_edge_ind = max(peak_edge_ind, 1)
            peak_edge_ind = min(peak_edge_ind, noise_ind.shape[0] - 2)

            self.debug_stream('In calculate_spectrum_parameters: peakInd done')
            # The peak is then located between [peakEdgeInd - 1] and [peakEdgeInd + 1]:
            peak_ind_min = max(noise_ind[peak_edge_ind - 1], 0)
            peak_ind_max = min(noise_ind[peak_edge_ind + 1], noise_ind.shape[0] - 1)
            peak_data = sp[peak_ind_min: peak_ind_max]

            with self.attr_lock:
                self.debug_stream('In calculate_spectrum_parameters: peakData done')
                peak_wavelengths = self.wavelengths_roi_data[peak_ind_min: peak_ind_max]
                try:
                    w = self.wavelengths_roi_data[3:-3]
                    spN = (m - noise_floor * 1.1).clip(0)
                    peak_energy = 1560 * 1e-6 * np.trapz(spN, w) / self.exposuretime_data
                    peak_width = np.abs(np.diff(self.wavelengths_roi_data[half_ind_reduced]))
                    peak_center = np.trapz(spN * w) / np.trapz(spN)
                except Exception as e:
                    self.error_stream(f"In calculate_spectrum_parameters: Error calculating peak parameters: {e}")
                    peak_energy = 0.0
                    peak_width = 0.0
                    peak_center = 0.0
                self.debug_stream('In calculate_spectrum_parameters: peakCenter done')
                self.peak_energy_data = peak_energy
                self.peak_width_data = peak_width
                self.peak_wavelength_data = peak_center

                self.debug_stream(f"In calculate_spectrum_parameters: computation time {time.time() - t0:.2} s")

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