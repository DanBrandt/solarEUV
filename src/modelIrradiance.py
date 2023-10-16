# MASTER MODULE
# This module contains functions for achieving the following:
# 1. Automatically generating irradiances given the following:
#    a. Bins within which to divide up the irradiances.
#    b. A source for the irradiances.
# 2. Generating plots of the requested irradiances.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from empiricalModels.models.EUVAC import euvac
from empiricalModels.models.HEUVAC import heuvac
from empiricalModels.models.SOLOMON import solomon
from NEUVAC.src import neuvac
from tools import processIndices
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Directory Management:
outputDir = '../experiments/Queries/'
FISM2DataDir = '../empiricalModels/FISM2/'
TIMEDDataDir = '../measurements/TIMED_SEE_Level_3/'
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Global variables:
euvacTable = euvac.euvacTable
neuvacTable = np.flipud(neuvac.waveTable) # Only use this table for its wavelength boundaries in the first two columns.
neuvac_tableFile = '../NEUVAC/src/neuvac_table.txt' # For actually executing NEUVAC.
solomonTable = solomon.solomonTable
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Functions:
def getIrradiance(dateStart, dateEnd, bins, source=None):
    """
    Return the solar EUV irradiance between two dates in a variety of wavelength bins, from a specified source, if
    applicable. The resulting time series data is saved to a formatted file. Additionally, make figures of the time
    series data (in each bin) in a specific location. Note that all irradiances are in W/m^2. When the irradiance is for
    a singular wavelength band, the units are W/m^2/nm. Additionally, all wavelength are in nanometers.
    :param dateStart: str
        The starting date, in YYYY-MM-DD format.
    :param dateEnd: str
        The ending date, in YYYY-MM-DD format.
    :param bins: str
        The scheme with which to bin the data. The following arguments are valid:
        - EUVAC, HEUVAC or None: The 37 wavelength bins used by EUVAC and HEUVAC.
        - NEUVAC: The 59 wavelength bins used by Aether and GITM.
        - SOLOMON: The 23 bands from Solomon and Qian 2005.
        - SEE: The bins corresponding to TIMED/SEE Level 3 daily-averaged spectra.
    :param source: str
        Arguments are either FISM2 or SEE. If no argument is given, then the default is to use the same data source
        as specified by 'bins'. Otherwise, this argument allows one to bin up irradiance data from one source according
        to the bins of another source. Please note the following:
        For EUVAC/HEUVAC/NEUVAC/SOLOMON, no other source can be applied. These models are specific to specific
        wavelength bins. One is only free to divide FISM2 and SEE data into different bin structures.
    :return times: ndarray
        A 1d array of datetimes for each irradiance measurement (spectrum).
    :return binLow: ndarray
        The lower edges of the wavelength bins for the binned irradiance data.
    :return binHigh: ndarray
        The upper edges of the wavelength bins for the binned irradiance data.
    :return binCenters: ndarray
        The centers of wavelength bins for the binned irradiance data.
    :return irradiance: ndarray
        A two-dimensional array of spectra, where each row is an individual spectrum at a specific time.
    """
    # Obtain F10.7 data:
    times, F107, F107A = processIndices.getF107(dateStart, dateEnd)

    # Obtain the desired bins:
    if bins == 'HEUVAC':
        binLow = euvacTable[:, 1]
        binHigh = euvacTable[:, 2]
        binCenters = 0.5 * (binLow + binHigh)
    elif bins == 'NEUVAC':
        binLow = neuvacTable[:, 0]
        binHigh = neuvacTable[:, 1]
        binCenters = 0.5 * (binLow + binHigh)
    elif bins == 'SOLOMON':
        # TODO: COMPLETE THE SOLOMON MODEL!
        binLow = solomonTable  # []!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        binHigh = solomonTable  # []!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        binCenters = 0.5 * (solomonTable + solomonTable)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    else:
        # Default to using EUVAC:
        binLow = euvacTable[:, 1]
        binHigh = euvacTable[:, 2]
        binCenters = 0.5 * (binLow + binHigh)

    # Obtain irradiances in the desired manner:
    if source == None:
        if bins == 'HEUVAC':
            flux, irradiance = heuvac.heuvac(F107, F107A)
            return times, binLow, binHigh, binCenters, irradiance
        elif bins == 'NEUVAC':
            flux, irradiance = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile)
            return times, binLow, binHigh, binCenters, irradiance
        elif bins == 'SOLOMON':
            # TODO: COMPLETE THE SOLOMON MODEL!
            flux, irradiance = solomon.solomon(F107, F107A)
            return times, binLow, binHigh, binCenters, irradiance
        else:
            # Default to using EUVAC:
            flux, irradiance = euvac.euvac(F107, F107A)
            return times, binLow, binHigh, binCenters, irradiance
    else:
        # FISM2:
        if source == 'FISM2':
            print()
        # TIMED/SEE:
        elif source == 'SEE':
            print()






#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    myDateStart = '2020-07-20' # Date on when St. Alphonsus Ligouri was consecrated a bishop.
    myDateEnd = '2020-08-01' # Date on when St. Alphonsus Ligouri died (and when he was canonized).
    myBins = 'HEUVAC'
    mySource = None
    getIrradiance(myDateStart, myDateEnd, myBins, source=mySource)
# -----------------------------------------------------------------------------------------------------------------------
