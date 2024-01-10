# MASTER MODULE
# This module contains functions for achieving the following:
# 1. Automatically generating irradiances given the following:
#    a. Bins within which to divide up the irradiances.
#    b. A source for the irradiances.
# 2. Generating plots of the requested irradiances.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from empiricalModels.models.EUVAC import euvac
from empiricalModels.models.HEUVAC import heuvac
from empiricalModels.models.SOLOMON import solomon
from NEUVAC.src import neuvac
from tools import processIndices
from tools import processIrradiances
from tools import spectralAnalysis
from tools import toolbox
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
SEETable = processIrradiances.SEEBands
figuresDirectory = '../experiments/Figures/'
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
        - 'EUVAC', 'HEUVAC': The 37 wavelength bins used by EUVAC and HEUVAC.
        - 'NEUVAC' or None: The 59 wavelength bins used by Aether and GITM.
        - 'SOLOMON': The 23 bands from Solomon and Qian 2005.
        - 'SEE': The bins corresponding to TIMED/SEE Level 3 daily-averaged spectra.
    :param source: str
        Arguments are either 'FISM2' or 'SEE'. If no argument is given, then the default is to use the same data source
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
    if bins == 'EUVAC' or bins == 'HEUVAC':
        binLow = euvacTable[:, 1]
        binHigh = euvacTable[:, 2]
        binCenters = 0.5 * (binLow + binHigh)
    elif bins == 'SOLOMON':
        binLow = solomonTable[:, 1]
        binHigh = solomonTable[:, 2]
        binCenters = 0.5 * (binLow + binHigh)
    elif bins == 'SEE':
        # Bins for SEE are the center wavelength of each band. They are uniformly spaced by 10 Angstroms.
        binLow = SEETable - 50
        binHigh = SEETable + 50
        binCenters = SEETable
    else:
        # Default to using NEUVAC:
        binLow = neuvacTable[:, 0]
        binHigh = neuvacTable[:, 1]
        binCenters = 0.5 * (binLow + binHigh)
    wavelengths = binCenters

    # Establish a dictionary for wavelength resolution:
    euv_data = {'short': binLow,
                'long': binHigh}

    # Obtain irradiances in the desired manner:
    if source == None:
        if bins == 'HEUVAC':
            flux, irradiance = heuvac.heuvac(F107, F107A)
        elif bins == 'NEUVAC':
            irradiance, irradiance_perturbed, _, _ = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile)
        elif bins == 'SOLOMON':
            # TODO: FIX ISSUES WITH THE SOLOMON MODEL!
            flux, irradiance = solomon.solomon(F107, F107A)
        else:
            # Default to using EUVAC:
            flux, irradiance = euvac.euvac(F107, F107A)
    else:
        # Obtain the irradiance data:
        if source == 'SEE':
            binFactor = 5
            # Download/load in TIMED/SEE data:
            times, rawWavelengths, rawIrr = processIrradiances.getIrr(dateStart, dateEnd, source='SEE')
            # SEE Data must be cleaned before being returned:
            rawIrr[rawIrr <= 0] = np.nan
        else:
            binFactor = None
            if source == 'FISM2':
                times, rawWavelengths, rawIrr = processIrradiances.getIrr(dateStart, dateEnd, source='FISM2')
            if source == 'FISM2S':
                times, rawWavelengths, rawFlux = processIrradiances.getIrr(dateStart, dateEnd, source='FISM2S')
                # The FISM2 Standard Bands are in units of photons/cm^2/s and need to be converted to irradiances in
                # units of W/m^2:
                rawIrr = np.zeros_like(rawFlux)
                for i in range(rawFlux.shape[1]):
                    wav = rawWavelengths[i]
                    dWav = solomon.solomonBandWidths[i]
                    rawIrr[:, i] = spectralAnalysis.spectralIrradiance(rawFlux[:, i], wav, dWavelength=dWav)
                if bins != 'SOLOMON':
                    raise ValueError("'FISM2S' can only be supplied as a source if 'bins' is set to 'SOLOMON'.")

        # Rebin the obtained irradiance data as desired:
        if bins == 'SOLOMON' and source == 'FISM2S':
            # Do nothing:
            wavelengths = rawWavelengths
            irradiance = rawIrr
            print('No rebinning will take place for FISM2 Daily Standard Bands Data.')
        else:
            wavelengths, irradiance = toolbox.rebin(wavelengths=rawWavelengths, data=rawIrr, resolution=euv_data,
                                        factor=binFactor, zero=False)

    # Plotting irradiance data:
    # Make a directory to save things into, if it doens't already exist:
    if source:
        titleStr = source+' ('+bins+' bins)'
        figName = source+'_'+bins+'_bins_'+dateStart+'-'+dateEnd+'.png'
        saveDir = figuresDirectory+source+'_'+bins+'_'+dateStart+'-'+dateEnd+'/'
    else:
        titleStr = bins
        figName =  bins+'_' + dateStart + '-' + dateEnd + '.png'
        saveDir = figuresDirectory + bins + dateStart + '-' + dateEnd+'/'
    toolbox.openDir(saveDir)

    # 1: Image of the entire spectrum:
    y_lims = mdates.date2num(times)
    myExtent = [wavelengths[0], wavelengths[-1], y_lims[0], y_lims[-1]]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(irradiance, extent=myExtent, aspect='auto')
    ax.yaxis_date()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.yaxis.set_major_formatter(date_format)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Date')
    ax.set_title('Spectra for '+titleStr)
    plt.savefig(saveDir+figName, dpi=300)

    # 2: Time-series images of each wavelength band:
    for i in range(irradiance.shape[1]):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(times, irradiance[:, i])
        plt.xlabel('Time')
        plt.ylabel('Irradiance W/m$^2$(/nm)')
        plt.title('Irradiance Time Series '+str(np.round(wavelengths[i], 2))+' nm')
        plt.savefig(saveDir+figName[:-4]+'_'+str(np.round(wavelengths[i], 2)).replace('.','_')+'_nm.png')
        plt.close(fig)

    return times, binLow, binHigh, binCenters, irradiance
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    myDateStart = '2020-07-20' # Date on when St. Alphonsus Ligouri was consecrated a bishop.
    myDateEnd = '2020-08-01' # Date on when St. Alphonsus Ligouri died (and when he was canonized).
    # times, binLow, binHigh, binCenters, irradiance = getIrradiance(myDateStart, myDateEnd, 'SOLOMON', source='SEE')
    # times, binLowSOLOMON, binHighSOLOMON , binCentersSOLOMON , irradianceSOLOMON = getIrradiance(myDateStart, myDateEnd, 'SOLOMON', source='FISM2S')

    # NEUVAC, FISM2, SEE:
    times, binLow, binHigh, binCenters, irradianceNEUVAC = getIrradiance(myDateStart, myDateEnd, 'NEUVAC')
    times, binLow, binHigh, binCenters, irradianceFISM2 = getIrradiance(myDateStart, myDateEnd, 'NEUVAC', source='FISM2')
    times, binLow, binHigh, binCenters, irradianceFISM2 = getIrradiance(myDateStart, myDateEnd, 'NEUVAC', source='SEE')
# -----------------------------------------------------------------------------------------------------------------------
