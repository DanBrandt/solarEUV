# This script prepares compares solar irradiance generated by NEUVAC to other models of solar EUV, including EUVAC,
# HEUVAC, and SOLOMON.
# The figures generated in this script correspond to those in the main NEUVAC publication.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from tools import toolbox
from tools.EUV.fism2_process import read_euv_csv_file
from tools.processIrradiances import obtainFism2
from NEUVAC.src import neuvac
from empiricalModels.models.EUVAC import euvac
from empiricalModels.models.HEUVAC import heuvac
from empiricalModels.models.SOLOMON import solomon
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Directory Management
euv_folder = '../tools/EUV/'
neuvac_tableFile = '../NEUVAC/src/neuvac_table.txt'
results_dir = 'Results/'
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (16, 8),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution:
if __name__=="__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # 1: OBTAIN DATA
    # Load in F10.7 and F10.7A data (OMNIWeb):
    omniTimesData = '../solarIndices/F107/OMNIWeb/OMNIF107times.pkl'
    omniF107Data = '../solarIndices/F107/OMNIWeb/OMNIF107vals.pkl'
    omniF107AveData = '../solarIndices/F107/OMNIWeb/OMNIF107averageVals.pkl'
    times = toolbox.loadPickle(omniTimesData)
    F107 = toolbox.loadPickle(omniF107Data)
    F107A = toolbox.loadPickle(omniF107AveData)

    # Obtain FISM2 data:
    euv_data_59 = read_euv_csv_file(euv_folder + 'euv_59.csv', band=False)
    mids = 0.5 * (euv_data_59['long'] + euv_data_59['short'])
    fism2file = '../empiricalModels/irradiances/FISM2/daily_data_1947-2023.nc'
    myIrrTimesFISM2, wavelengthsFISM2, myIrrDataAllFISM2, myIrrUncAllFISM2 = obtainFism2(fism2file)
    # Rebin the data:
    myIrrDataWavelengthsFISM2, rebinnedIrrDataFISM2 = toolbox.rebin(wavelengthsFISM2, myIrrDataAllFISM2, euv_data_59,
                                                                    zero=False)
    fism2Irr = rebinnedIrrDataFISM2[:, 7:44]
    # Harmonize the times for NEUVAC and FISM2:
    correspondingIndsFISM2 = np.where((myIrrTimesFISM2 >= times[0]) & (myIrrTimesFISM2 <= times[-1]))[0]
    correspondingIrrTimesFISM2 = myIrrTimesFISM2[correspondingIndsFISM2]
    correspondingFism2Irr = fism2Irr[correspondingIndsFISM2, :]

    # Generate NEUVAC data:
    neuvacIrr, perturbedNeuvacIrr, savedPerts, cc2 = neuvac.neuvacEUV(F107, F107A, bandLim=True, tableFile=neuvac_tableFile,
                                                                      statsFiles=['corMat.pkl', 'sigma_NEUVAC.pkl'])

    # Generate EUVAC data:
    euvacFlux, euvacIrr = euvac.euvac(F107, F107A)

    # Generate HEUVAC data:
    heuvac_wav, heuvacFlux, heuvacIrr = heuvac.heuvac(F107, F107A, torr=True)

    # Generate SOLOMON data and rebin everything into the SOLOMON bins:
    # solomonIrrFISM2 = toolbox.rebin(heuvac_wav, correspondingFism2Irr, resolution=solomon.solomonTable, zero=False) # TODO: Replace with a command to LOAD IN FISM2 STAN BANDS from LISIRD
    # solomonIrrNEUVAC = toolbox.rebin(heuvac_wav, perturbedNeuvacIrr, resolution=solomon.solomonTable, zero=False) # TODO: Replace with a new function call to the freshly-fit NEUVAC STAN BANDS results
    # solomonFluxHFG, solomonIrrHFG = solomon.solomon(F107, F107A, model='HFG')
    # solomonFluxEUVAC, solomonIrrEUVAC = solomon.solomon(F107, F107A, model='EUVAC')
    # solomonIrrHEUVAC = toolbox.rebin(heuvac_wav, heuvacIrr, resolution=solomon.solomonTable, zero=False)

    #===================================================================================================================
    # NOTE: In any analysis below, only 3 bands will be considered for plotting (75 A, 475 A, 1025 A)
    #===================================================================================================================

    # 1A: Solar Spectra in Low and High Solar Activity
    euvacTable = euvac.euvacTable
    mids = 0.5 * (euvacTable[:, 1] + euvacTable[:, 2])
    xPos = np.append(euvacTable[:, 1], euvacTable[:, 2][-1])
    sortInds = np.argsort(xPos)
    xPosSorted = xPos[sortInds]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # TODO: Plot wavelength ranges and singular wavelengths separately
    # i: Low Activity:
    chosenDateLow = datetime(1985, 11, 4) # Beginning of Solar Cycle 21
    idx, val = toolbox.find_nearest(times, chosenDateLow)
    ax[0].stairs(values=correspondingFism2Irr[idx, :][sortInds[:-1]], edges=xPosSorted, label='FISM2', lw=3)
    ax[0].stairs(values=perturbedNeuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='NEUVAC', lw=3)
    ax[0].stairs(values=euvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='EUVAC', lw=3)
    ax[0].stairs(values=heuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='HEUVAC', lw=3)
    ax[0].set_yscale('log')
    ax[0].legend(loc='best')
    ax[0].grid()
    ax[0].set_xlabel('Wavelength ($\mathrm{\AA}$)')
    ax[0].set_ylabel('Irradiance (W/m$^2$)')
    ax[0].set_title('Solar Spectra during Low Solar Activity ('+str(chosenDateLow)[:-9]+')')
    # ii: High Activity:
    chosenDateHigh = datetime(1991, 1, 31) # Peak of Solar Cycle 21
    idx, val = toolbox.find_nearest(times, chosenDateHigh)
    chosenDateLow = datetime(1985, 11, 4)  # Beginning of Solar Cycle 21
    ax[1].stairs(values=correspondingFism2Irr[idx, :][sortInds[:-1]], edges=xPosSorted, label='FISM2', lw=3)
    ax[1].stairs(values=perturbedNeuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='NEUVAC', lw=3)
    ax[1].stairs(values=euvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='EUVAC', lw=3)
    ax[1].stairs(values=heuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='HEUVAC', lw=3)
    ax[1].set_yscale('log')
    ax[1].legend(loc='best')
    ax[1].grid()
    ax[1].set_xlabel('Wavelength ($\mathrm{\AA}$)')
    ax[1].set_title('Solar Spectra during High Solar Activity ('+str(chosenDateHigh)[:-9]+')')
    plt.savefig(results_dir+'sample_spectra_low_and_high_solar_activity.png', dpi=300)

    # 1B: Plot sample TIME SERIES during Low and High Solar Activity (3 bands only):
    fig, axs = plt.subplots(nrows=2, ncols=3)
    lowSolarTimeBounds = [chosenDateLow-timedelta(days=40), chosenDateLow+timedelta(days=41)]
    lowSolarTimeInds = np.where((times >= lowSolarTimeBounds[0]) & (times <= lowSolarTimeBounds[-1]))[0]
    lowSolarTimes = times[lowSolarTimeInds]
    highSolarTimeBounds = [chosenDateHigh - timedelta(days=40), chosenDateHigh + timedelta(days=41)]
    highSolarTimeInds = np.where((times >= highSolarTimeBounds[0]) & (times <= highSolarTimeBounds[-1]))[0]
    highSolarTimes = times[highSolarTimeInds]
    # Top-left: Low Solar Activity 75 A
    axs[0, 0].plot(lowSolarTimes, correspondingFism2Irr[lowSolarTimeInds, 0], label='FISM2', lw=3)
    axs[0, 0].plot(lowSolarTimes, perturbedNeuvacIrr[lowSolarTimeInds, 0], label='NEUVAC', lw=3)
    axs[0, 0].plot(lowSolarTimes, euvacIrr[lowSolarTimeInds, 0], label='EUVAC', lw=3)
    axs[0, 0].plot(lowSolarTimes, heuvacIrr[lowSolarTimeInds, 0], label='EUVAC', lw=3)
    axs[0, 0].set_ylabel('Irradiance (W/m$^2$)')
    axs[0, 0].set_title('75 $\mathrm{\AA}$')
    axs[0, 0].legend(loc='best')
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
    # Top-middle: Low Solar Activity 475 A
    axs[0, 1].plot(lowSolarTimes, correspondingFism2Irr[lowSolarTimeInds, 14], label='FISM2', lw=3)
    axs[0, 1].plot(lowSolarTimes, perturbedNeuvacIrr[lowSolarTimeInds, 14], label='NEUVAC', lw=3)
    axs[0, 1].plot(lowSolarTimes, euvacIrr[lowSolarTimeInds, 14], label='EUVAC', lw=3)
    axs[0, 1].plot(lowSolarTimes, heuvacIrr[lowSolarTimeInds, 14], label='HEUVAC', lw=3)
    axs[0, 1].set_title('425 $\mathrm{\AA}$')
    axs[0, 1].legend(loc='best')
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
    # Top-right: Low Solar Activity 1025 A
    axs[0, 2].plot(lowSolarTimes, correspondingFism2Irr[lowSolarTimeInds, -1], label='FISM2', lw=3)
    axs[0, 2].plot(lowSolarTimes, perturbedNeuvacIrr[lowSolarTimeInds, -1], label='NEUVAC', lw=3)
    axs[0, 2].plot(lowSolarTimes, euvacIrr[lowSolarTimeInds, -1], label='EUVAC', lw=3)
    axs[0, 2].plot(lowSolarTimes, heuvacIrr[lowSolarTimeInds, -1], label='HEUVAC', lw=3)
    axs[0, 2].set_title('1025 $\mathrm{\AA}$')
    axs[0, 2].legend(loc='best')
    axs[0, 2].set_xticklabels(axs[0, 2].get_xticklabels(), rotation=45, ha='right')
    # Bottom-left: High Solar Activity 75 A
    axs[1, 0].plot(highSolarTimes, correspondingFism2Irr[highSolarTimeInds, 0], label='FISM2', lw=3)
    axs[1, 0].plot(highSolarTimes, perturbedNeuvacIrr[highSolarTimeInds, 0], label='NEUVAC', lw=3)
    axs[1, 0].plot(highSolarTimes, euvacIrr[highSolarTimeInds, 0], label='EUVAC', lw=3)
    axs[1, 0].plot(highSolarTimes, heuvacIrr[highSolarTimeInds, 0], label='HEUVAC', lw=3)
    axs[1, 0].set_ylabel('Irradiance (W/m$^2$)')
    axs[1, 0].set_title('75 $\mathrm{\AA}$')
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
    # Bottom-middle: High Solar Activity 475 A
    axs[1, 1].plot(highSolarTimes, correspondingFism2Irr[highSolarTimeInds, 14], label='FISM2', lw=3)
    axs[1, 1].plot(highSolarTimes, perturbedNeuvacIrr[highSolarTimeInds, 14], label='NEUVAC', lw=3)
    axs[1, 1].plot(highSolarTimes, euvacIrr[highSolarTimeInds, 14], label='EUVAC', lw=3)
    axs[1, 1].plot(highSolarTimes, heuvacIrr[highSolarTimeInds, 14], label='HEUVAC', lw=3)
    axs[1, 1].set_title('475 $\mathrm{\AA}$')
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
    # Bottom-right: High Solar Activity 1025 A
    axs[1, 2].plot(highSolarTimes, correspondingFism2Irr[highSolarTimeInds, 14], label='FISM2', lw=3)
    axs[1, 2].plot(highSolarTimes, perturbedNeuvacIrr[highSolarTimeInds, 14], label='NEUVAC', lw=3)
    axs[1, 2].plot(highSolarTimes, euvacIrr[highSolarTimeInds, 14], label='EUVAC', lw=3)
    axs[1, 2].plot(highSolarTimes, heuvacIrr[highSolarTimeInds, 14], label='HEUVAC', lw=3)
    axs[1, 2].set_title('1025 $\mathrm{\AA}$')
    axs[1, 2].legend(loc='best')
    axs[1, 2].set_xticklabels(axs[1, 2].get_xticklabels(), rotation=45, ha='right')
    # Save the figure:
    fig.tight_layout()
    fig.suptitle('Irradiance During Low Solar Activity ('+str(lowSolarTimeBounds[0])[:-9]+' to '+
                 str(lowSolarTimeBounds[-1])[:-9]+') and High Solar Activity ('+str(highSolarTimeBounds[0])[:-9]+' to '
                 +str(highSolarTimeBounds[-1])[:-9]+')\n', fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=0.9)
    plt.savefig(results_dir+'sampleTimeSeriesSpectra_Low_and_High_Solar_Activity.png', dpi=300)

    # ==================================================================================================================
    # 2: [STATISTICS OF] PERTURBATIONS (FOR ALL MODELS)

    # ==================================================================================================================
    # 3: DISTRIBUTION FUNCTIONS (OF RESIDUALS WRT FISM2)

    # ==================================================================================================================
    # 4: INTEGRATED ENERGY (ACROSS THE SUN-FACING SIDE OF THE EARTH)

    # ==================================================================================================================
    # Exit with a zero error code:
    sys.exit(0)
#-----------------------------------------------------------------------------------------------------------------------
