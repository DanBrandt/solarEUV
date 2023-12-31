# Code for computing solar irradiance according to the EUVAC model.
# Reference: Richard, P. G., Fennelly, J. A., and Torr, D. G., EUVAC: A solar EUV flux model for aeronomic calculations,
# Journal of Geophysical Research, 99, A5, 8981-8991, 1994

#-----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
from random import randrange
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from tools.spectralAnalysis import spectralIrradiance
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Constants
h = 6.62607015e-34 # Planck's constant in SI units of J s
c = 299792458 # Speed of light in m s^-1
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Global variables:
euvacTable = np.array([
    [1, 50, 100, 1.200, 1.0017e-2],
    [2, 100, 150, 0.450, 7.1250e-3],
    [3, 150, 200, 4.800, 1.3375e-2],
    [4, 200, 250, 3.100, 1.9450e-2],
    [5, 256.32, 256.32, 0.460, 2.7750e-3],
    [6, 284.15, 284.15, 0.210, 1.3768e-1],
    [7, 250, 300, 1.679, 2.6467e-2],
    [8, 303.31, 303.31, 0.800, 2.5000e-2],
    [9, 303.78, 303.78, 6.900, 3.3333e-3],
    [10, 300, 350, 0.965, 2.2450e-2],
    [11, 368.07, 368.07, 0.650, 6.5917e-3],
    [12, 350, 400, 0.314, 3.6542e-2],
    [13, 400, 450, 0.383, 7.4083e-3],
    [14, 465.22, 465.22, 0.290, 7.4917e-3],
    [15, 450, 500, 0.285, 2.0225e-2],
    [16, 500, 550, 0.452, 8.7583e-3],
    [17, 554.37, 554.37, 0.720, 3.2667e-3],
    [18, 584.33, 584.33, .270, 5.1583e-3],
    [19, 550, 600, 0.357, 3.6583e-3],
    [20, 609.76, 609.76, 0.530, 1.6175e-2],
    [21, 629.73, 629.73, 1.590, 3.3250e-3],
    [22, 600, 650, 0.342, 1.1800e-2],
    [23, 650, 700, 0.230, 4.2667e-3],
    [24, 703.36, 703.36, 0.360, 3.0417e-3],
    [25, 700, 750, 0.141, 4.7500e-3],
    [26, 765.15, 765.15, 0.170, 3.8500e-3],
    [27, 770.41, 770.41, 0.260, 1.2808e-2],
    [28, 789.36, 789.36, 0.702, 3.2750e-3],
    [29, 750, 800, 0.758, 4.7667e-3],
    [30, 800, 850, 1.625, 4.8167e-3],
    [31, 850, 900, 3.537, 5.6750e-3],
    [32, 900, 950, 3.000, 4.9833e-3],
    [33, 977.02, 977.02, 4.400, 3.9417e-3],
    [34, 950, 1000, 1.475, 4.4167e-3],
    [35, 1025.72, 1025.72, 3.500, 5.1833e-3],
    [36, 1031.91, 1031.91, 2.100, 5.2833e-3],
    [37, 1000, 1050, 2.467, 4.3750e-3]
    ])
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Functions:
def refSpec(i):
    """
    Return the standard solar flux in 37 bands from the F74113 Spectrum (pp. 584-585 in Schunk and Nagy).
    Source 2: Richard, P. G., Fennelly, J. A., and Torr, D. G., EUVAC: A solar EUV flux model for aeronomic
    calculations, Journal of Geophysical Research, 99, A5, 8981-8992, 1994.
    Source 3: Heroux, L. and Hinteregger, H. E., Aeronomical Reference Spectrum for Solar UV Below 2000 A, Journal of
    Geophysical Research, 83, A11, 1978.
    :param: i: int
        The index for the wavelength. Must be between 0 and 37.
    :return: F74113_i: float
        The reference solar flux in units of photons m^-2 s^-1.
    :return: A_i: float
        The scaling factor for the wavelength interval.
    """
    lookUpIdx = np.where(euvacTable == i)[0]
    F74113_i = euvacTable[lookUpIdx, 3][0]*(1e13) # Multiply by 1e13 to obtain photons m^-2 s^-1.
    A_i = euvacTable[lookUpIdx, 4][0]
    return F74113_i, A_i

def euvac(F107, F107A):
    """
    Compute the solar flux from F10.7, according to the EUVAC model. Return the solar flux across 37 wavelength
    bands in units of photons m^-2 s^-1.
    :param F107: ndarray
        Values of the F10.7 solar flux.
    :param F107A: ndarray
        Values of the 81-day averaged solar flux, centered on the present day.
    :return: euvacFlux: ndarray
        Values of the solar radiant flux in 37 distinct wavelength bands.
    :return euvacIrr: ndarray
        Values of the solar spectral irradiance in 37 distinct wavelength bands.
    """
    P = (F107A + F107)/2.0
    if type(F107) == np.ndarray:
        euvacFlux = np.zeros((len(F107), 37)) # Columns represent each wavelength band 37 (59).
        euvacIrr = np.zeros((len(F107), 37))
    else:
        euvacFlux = np.zeros((1, 37))
        euvacIrr = np.zeros((1, 37))
    for i in range(37):
        wav = 0.5*(euvacTable[i, 2] + euvacTable[i, 1])
        dWav = euvacTable[i, 2] - euvacTable[i, 1]
        if dWav == 0:
            dWav = None
        F74113_i, A_i = refSpec(i+1)
        fluxFactor = (1. + A_i*(P-80))
        # if fluxFactor < 0.8:
            # fluxFactor = 0.8
        photonFlux = (F74113_i)*fluxFactor
        # If P-80 is negative, set the flux to ZERO.
        try:
            photonFlux[photonFlux < 0] = 0
        except:
            if photonFlux < 0:
                photonFlux = 0
        euvacFlux[:, i] = photonFlux
        euvacIrr[:, i] = spectralIrradiance(photonFlux, wavelength=wav, dWavelength=dWav)
    return euvacFlux, euvacIrr
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution:
if __name__ == '__main__':
    F107 = np.array([20, 25, 40, 70, 85, 84, 72, 58, 59, 49, 37, 21])
    # F107A = averageF107(F107)
    myFlux, myIrr = euvac(F107, F107)
    # myIrr = spectralIrradiance(myFlux, 400, 5)
    # flux = euvac(200, 200)
    # F107 = np.array([randrange(200) for element in range(6000)])

    # EUVAC wavelengths (ranges):
    # euvacShort = np.array([50., 100., 150., 200., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750.,
    #                        800., 850., 900., 950., 1000.])
    # euvacLong = np.array([ 100.,  150.,  200.,  250.,  300.,  350.,  400.,  450.,  500., 550.,  600.,  650.,  700.,
    #                        750.,  800.,  850.,  900.,  950., 1000., 1050.])
    # middleWavelengths = 0.5*(euvacLong + euvacShort)
    # differences = euvacLong - euvacShort

    # Sanity check between fluxes/irradiances from EUVAC and those from Ridley's method:
    # validInds = np.array([0, 1, 2, 3, 6, 9, 11, 12, 14, 15, 18, 21, 22, 24, 28, 29, 30, 31, 33, 36])
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Qt5Agg')  # Install Pyqt5
    # from neuvac import ridleyEUV
    # rFlux = ridleyEUV(F107, F107A, bandLim=True)
    # eFlux = euvac(F107, F107A)
    # # Extract the bands for the singular wavelengths:
    # rFlux = rFlux[:, validInds]
    # eFlux = eFlux[:, validInds]
    # band = 5 # 36
    # for i in range(band+1):
    #     fig, axs = plt.subplots(2, 1)
    #     par1 = axs[0].twinx()
    #     par2 = axs[1].twinx()
    #
    #     axs[0].set_xlabel('Time (hours)')
    #     axs[0].set_ylabel('Flux')
    #     par1.set_ylabel('sfu')
    #
    #     # Fluxes
    #     axs[0].plot(eFlux[:, i], 'b-', alpha=0.5, label='EUVAC')
    #     axs[0].plot(rFlux[:, i], 'c-', label='Ridley')
    #
    #     # Irradiances
    #     eIrr = spectralIrradiance(eFlux[:, i], middleWavelengths[i], differences[i])
    #     rIrr = spectralIrradiance(rFlux[:, i], middleWavelengths[i], differences[i])
    #     axs[1].set_xlabel('Time (hours)')
    #     axs[1].set_ylabel('Irradiance')
    #     axs[1].plot(eIrr, 'b-', alpha=0.5, label='EUVAC')
    #     axs[1].plot(rIrr, 'c-', label='Ridley')
    #     par2.set_ylabel('sfu')
    #
    #     #par1.plot(F107, 'g-', label='F10.7')
    #     par1.plot(F107A, 'r-', label='F10.7A')
    #     par2.plot(F107A, 'r-', label='F10.7A')
    #
    #     axs[0].legend(loc='best')
    #     axs[1].legend(loc='best')
    #     plt.show()

