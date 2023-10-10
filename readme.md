# SPM002 spectrometer control
This repository provides control of Photoncontrol SPM002 spectrometers 
in a python class. There is also a Tango control system device server.

### Prerequisites
- Windows 32-bit python3 distribution (the dll file is 32-bit)

- Numpy

- PyTango if using the Tango device server  

### Usage
Simple example of connecting to spectrometer and plotting a spectrum using matplotlib:
```
from SPM002_control import SPM002Control
import matplotlib.pyplot as mpl

spm = SPM002Control()
spm.open_device(0)
spm.construct_wavelengths()
spm.acquire_spectrum()
mpl.plot(spm.get_wavelengths(), spm.get_spectrum())
mpl.show()
```