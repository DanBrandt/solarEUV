# Code for computing converting flux into irradiance and the reverse, along with miscelleneous helper functions.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Constants
h = 6.62607015e-34 # Planck's constant in SI units of J s
c = 299792458 # Speed of light in m s^-1
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Functions:
def spectralIrradiance(photonFlux, wavelength, dWavelength=None):
    """
    Convert the photon flux to the corresponding spectral irradiance, given a specific wavelength.
    :param: photonFlux: ndarray, float, or int
        Photon flux in units of photons s^-1 m^-2. For a singular wavelength, units are photons m^-2
    :param: wavelength: float
        A specific wavelength in Angstroms.
    :param: dWavelength: float or int
        Wavelength bin width in Angstroms. Default is None.
    :return: irradiance: ndarray or float
        The corresponding spectral irradiance in units of W/m^2/nm.
    """
    photonEnergy = (h*c) / (wavelength*1e-10) # Convert the wavelength in the denominator to meters.
    if dWavelength != None:
        irradiance = photonFlux * photonEnergy * (1./(dWavelength*0.1)) # Multiply the denominator by 0.1 in order to convert from an Angstrom interval to a nanometer interval.
    else:
        irradiance = photonFlux * photonEnergy / wavelength
    return irradiance

def spectralFlux(irradiance, wavelength, dWavelength=10):
    """
    Convert the spectral irradiance to spectral flux, given a specific wavelength.
    :param: irradiance: ndarray, float, or int
        [Solar] spectral irradiance for a specific wavelength, in W/m^2/nm.
    :param: wavelength: float
        A specific wavelength in Angstroms.
    :param: dWavelength: float or int
        Wavelength bin width in Angstroms. Default is 1.
    :return: photonFlux: ndarray or float
        The corresponding spectral flux in units of Watts.
    """
    photonEnergy = (h * c) / (wavelength * 1e-10)
    photonFlux = (irradiance * dWavelength * 0.1) / photonEnergy
    return photonFlux