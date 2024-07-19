# This example focuses on creating simple irradiance time series.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from NEUVAC import neuvac

f107 = np.random.uniform(low=60, high=200, size=(100,))
f107a = f107

neuvac_tableFile = '../NEUVAC/neuvac_table.txt'
neuvacIrradiance, _, _, _ = neuvac.neuvacEUV(f107, f107a, tableFile=neuvac_tableFile)

# Just plot the irradiance in the 10th wavelength band:
plt.figure()
plt.plot(neuvacIrradiance[:, 9])
plt.xlabel('Sample')
plt.ylabel('Irradiance W/m$^2$')
plt.show()

