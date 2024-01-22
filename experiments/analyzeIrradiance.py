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
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# TODO: Globally manage plotting settings:
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
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
    solomonIrrFISM2 = toolbox.rebin(heuvac_wav, correspondingFism2Irr, resolution=solomon.solomonTable, zero=False)
    solomonIrrNEUVAC = toolbox.rebin(heuvac_wav, perturbedNeuvacIrr, resolution=solomon.solomonTable, zero=False)
    solomonFluxHFG, solomonIrrHFG = solomon.solomon(F107, F107A, model='HFG')
    solomonFluxEUVAC, solomonIrrEUVAC = solomon.solomon(F107, F107A, model='EUVAC')
    solomonIrrHEUVAC = toolbox.rebin(heuvac_wav, heuvacIrr, resolution=solomon.solomonTable, zero=False)

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
    ax[0].stairs(values=correspondingFism2Irr[idx, :][sortInds[:-1]], edges=xPosSorted, label='FISM2')
    ax[0].stairs(values=perturbedNeuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='NEUVAC')
    ax[0].stairs(values=euvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='EUVAC')
    ax[0].stairs(values=heuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='HEUVAC')
    ax[0].set_yscale('log')
    ax[0].legend(loc='best')
    ax[0].set_xlabel('Wavelength ($\mathrm{\AA}$)')
    ax[0].set_ylabel('Irradiance (W/m$^2$)')
    ax[0].set_title('Solar Spectra during Low Solar Activity ('+str(chosenDateLow)[:-9]+')')
    # ii: High Activity:
    chosenDateHigh = datetime(1991, 1, 31) # Peak of Solar Cycle 21
    idx, val = toolbox.find_nearest(times, chosenDateHigh)
    chosenDateLow = datetime(1985, 11, 4)  # Beginning of Solar Cycle 21
    ax[1].stairs(values=correspondingFism2Irr[idx, :][sortInds[:-1]], edges=xPosSorted, label='FISM2')
    ax[1].stairs(values=perturbedNeuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='NEUVAC')
    ax[1].stairs(values=euvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='EUVAC')
    ax[1].stairs(values=heuvacIrr[idx, :][sortInds[:-1]], edges=xPosSorted, label='HEUVAC')
    ax[1].set_yscale('log')
    ax[1].legend(loc='best')
    ax[1].set_xlabel('Wavelength ($\mathrm{\AA}$)')
    ax[1].set_ylabel('Irradiance (W/m$^2$)')
    ax[1].set_title('Solar Spectra during High Solar Activity ('+str(chosenDateHigh)[:-9]+')')

    # 1B: Plot sample TIME SERIES during Low and High Solar Activity (3 bands only):

    # ------------------------------------------------------------------------------------------------------------------
    # 2: PERTURBATIONS

    # ------------------------------------------------------------------------------------------------------------------
    # 3: DISTRIBUTION FUNCTIONS (OF RESIDUALS)

    # ------------------------------------------------------------------------------------------------------------------
    # 4: INTEGRATED ENERGY

    # ------------------------------------------------------------------------------------------------------------------

    plt.figure()
    j = 3000
    plt.plot(xPosSorted[:-1], correspondingFism2Irr[j, :][sortInds[:-1]], label='FISM2')
    # plt.plot(xPosSorted[:-1], neuvacIrr[j, :], label='NEUVAC')
    plt.plot(xPosSorted[:-1], perturbedNeuvacIrr[j, :], label='Perturbed NEUVAC')
    plt.plot(xPosSorted[:-1], euvacIrr[j, :][sortInds[:-1]], label='EUVAC')
    # plt.plot(xPosSorted[:-1], heuvacIrr[j, :], label='HEUVAC')
    plt.legend(loc='best')
    # plt.yscale('log')

    # Publication-like figure of a spectrum:
    # TODO: Revise this figure so that the spectral lines correspond to POINTS, and are plotted separately.
    plt.figure()
    j = 3500
    plt.stairs(values=correspondingFism2Irr[j, :][sortInds[:-1]], edges=xPosSorted, label='FISM2')
    # plt.plot(neuvacIrr[j, :], label='NEUVAC')
    plt.stairs(values=neuvacIrr[j, :][sortInds[:-1]], edges=xPosSorted, label='NEUVAC')
    # plt.plot(perturbedNeuvacIrr[j, :], label='Perturbed NEUVAC')
    plt.stairs(values=perturbedNeuvacIrr[j, :][sortInds[:-1]], edges=xPosSorted, label='Perturbed NEUVAC')
    # plt.plot(xPosSorted[:-1], euvacIrr[j, :][sortInds[:-1]], label='EUVAC')
    plt.stairs(values=euvacIrr[j, :][sortInds[:-1]], edges=xPosSorted, label='EUVAC')
    # plt.plot(heuvacIrr[j, :], label='HEUVAC')
    # plt.stairs(values=heuvacIrr[j, :][sortInds[:-1]], edges=xPosSorted, label='HEUVAC')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()

    for i in range(euvacIrr.shape[1]):
        plt.figure()
        plt.plot(correspondingFism2Irr[:, i], label='FISM2')
        # plt.plot(neuvacIrr[:, i], label='NEUVAC')
        plt.plot(perturbedNeuvacIrr[:, i], label='Perturbed NEUVAC')
        plt.plot(euvacIrr[:, i], label='EUVAC')
        # plt.plot(heuvacIrr[:, i], label='HEUVAC')
        plt.legend(loc='best')

    sys.exit(0)
#-----------------------------------------------------------------------------------------------------------------------
