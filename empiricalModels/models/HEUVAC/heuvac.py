# Run HEUVAC

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import os
from tqdm import tqdm
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Unchangeable filenames (lines 167-169 in HEUVAC-Driver.for), where HEUVAC outputs are stored:
topDir = os.getcwd()
directory = '../empiricalModels/models/HEUVAC/'
torrFluxFile = 'Torr-37-bins.txt'
userFluxFile = 'flux-User-bins-10A.txt'
userIonizationFile = 'XS-User-bins-10A.txt'
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from tools.spectralAnalysis import spectralIrradiance
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Helper Functions:
def writeInputFile(F107, F107A):
    """
    Write the formatted input file for running HEUVAC.
    :param F107: float
        A single value of F10.7.
    :param F107A: float
        A single value for the 81-day averaged F10.7, centered on the current day.
    :return:
        Returns nothing.
    """
    filename = 'HEUVAC-scratch.TXT'
    with open(filename, 'w') as heuvacFile:
        heuvacFile.write(str(F107)+'\n')
        heuvacFile.write(str(F107A)+'\n')
        heuvacFile.write(str(10)) # We manually keep the 10 Angstrom bin width (the highest resolution of HEUVAC)
    print(filename)

def getTorr(fluxFile):
    """
    Read the output file from HEUVAC in the Torr bins.
    :param fluxFile: str
        The filename where the HEUVAC fluxes in the Torr bins have been output.
    :return wav: ndarray
        The wavelength bin centers for the Torr bins (Angstroms).
    :return flux: ndarray
        The HEUVAC flux in the Torr bins (W/m2/nm).
    """
    wavs = np.zeros(37)
    fluxes = np.zeros(37)
    irrs = np.zeros(37)
    with open(fluxFile) as myFile:
        fileData = myFile.readlines()
        i = 0
        j = 0
        for line in fileData:
            if i > 0:
                wavs[j] = float(line.split()[1])
                fluxes[j] = float(line.split()[-1])
                irrs[j] = spectralIrradiance(fluxes[j], wavs[j])
                j += 1
            i += 1
    # The arrays will have values from largest to smallest - they should be flipped before being returned:
    wav = np.flip(wavs)
    flux = np.flip(fluxes)
    irr = np.flip(irrs)
    return wav, flux, irr

def getFlux(userFile):
    """
    Read the output file from HEUVAC in the Torr bins.
    :param userFile: str
        The filename where the HEUVAC fluxes in the user-defined bins have been output.
    :return wav: ndarray
        The wavelength bin centers for the Torr bins (Angstroms).
    :return flux: ndarray
        The HEUVAC flux in the user-defined bins (ph/cm2/s).
    :return: irr
        The HEUVAC irradiance in the user-defined bins (W/m2/nm).
    """
    wav = np.zeros(106)
    flux = np.zeros(106)
    irr = np.zeros(106)
    with open(userFile) as myFile:
        fileData = myFile.readlines()
        i = 0
        j = 0
        for line in fileData:
            if i >= 2:
                wav[j] = float(line.split()[3])
                flux[j] = float(line.split()[-2])
                irr[j] = float(line.split()[-1])
                j += 1
            i += 1
    return wav, flux, irr

def heuvac(F107, F107A, torr=True):
    """
    Call the HEUVAC Fortran code for each F10.7, F10.7A pair.
    :param F107: arraylike
        The values of [daily] F10.7.
    :param F107A: arraylike
        The values of 81-day averaged F10.7, centered on the current day.
    :param torr: bool
        Controls whether or not the binned data returned is in the 37 standard Torr et al bins or in the high-resolution
        10 Angstrom-wide bins (the standard high resolution of HEUVAC). Default is True.
    :return heuvacWav: ndarray
        The bin center wavelengths for the HEUVAC data.
    :return heuvacFlux: ndarray
        The solar EUV flux in different wavelength bins returned from HEUVAC.
    :return heuvacIrr: ndarray
        The solar EUV irradiance in different wavelength bins returend from HEUVAC.
    """
    if torr==True:
        heuvacFlux = np.zeros((len(F107), 37))
        heuvacIrr = np.zeros((len(F107), 37))
    else:
        heuvacFlux = np.zeros((len(F107), 106))
        heuvacIrr = np.zeros((len(F107), 106))
    for i in tqdm(range(len(F107))):
        # Write the input file and run HEUVAC:
        os.chdir(directory)
        writeInputFile(F107[i], F107A[i])
        os.system('./HEUVAC.exe')
        os.chdir(topDir)

        # Read in the fluxes in the Torr bins (37 bins) and the user-specified bins:
        torrWav, torrFlux, torrIrr = getTorr(torrFluxFile)
        userWav, userFlux, userIrr = getFlux(userFluxFile)

        # Collect the flux and irradiance into their respective arrays:
        if torr==True:
            heuvacFlux[i, :] = torrFlux
            heuvacIrr[i, :] = torrIrr
        else:
            heuvacFlux[i, :] = userFlux
            heuvacIrr[i, :] = userIrr

    if torr==True:
        heuvacWav = torrWav
    else:
        heuvacWav = userWav

    return heuvacWav, heuvacFlux, heuvacIrr
#-----------------------------------------------------------------------------------------------------------------------


