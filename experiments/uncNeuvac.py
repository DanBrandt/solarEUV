# This script performs uncertainty quantification on the NEUVAC model using a Kalman Filter.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import matplotlib, sys
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from NEUVAC.src import neuvac
from tools.EUV.fism2_process import read_euv_csv_file
from tools.processIrradiances import obtainFism2
from tools import toolbox
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Directory Management
euv_folder = '../tools/EUV/'
neuvac_tableFile = '../NEUVAC/src/neuvac_table.txt'
figures_folder = 'Uncertainty'
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    # Load in F10.7 data (OMNIWeb):
    omniTimesData = '../solarIndices/F107/OMNIWeb/OMNIF107times.pkl'
    omniF107Data = '../solarIndices/F107/OMNIWeb/OMNIF107vals.pkl'
    omniF107AveData = '../solarIndices/F107/OMNIWeb/OMNIF107averageVals.pkl'
    omniTimes = toolbox.loadPickle(omniTimesData)
    omniF107 = toolbox.loadPickle(omniF107Data)
    omniF107A = toolbox.loadPickle(omniF107AveData)
    # F10.7 data extends between 1963-11-28; 12:00 to 2023-09-27; 12:00.
    times = omniTimes
    F107 = omniF107
    F107A = omniF107A
    # Compute F10.7 uncertainties.
    rollingStdF107 = toolbox.rollingStd(F107, 2)
    rollingStdF107A = toolbox.rollingStd(F107A, 81)
    # View the results:
    plt.figure()
    plt.fill_between(np.linspace(0, len(F107) - 1, len(F107)), np.subtract(F107, rollingStdF107),
                     np.add(F107, rollingStdF107), color='cyan', linestyle='-')
    plt.plot(np.linspace(0, len(F107) - 1, len(F107)), F107, 'b-')
    plt.figure()
    plt.fill_between(np.linspace(0, len(F107A) - 1, len(F107A)), np.subtract(F107A, rollingStdF107A),
                     np.add(F107A, rollingStdF107A), color='limegreen', linestyle='-')
    plt.plot(np.linspace(0, len(F107A) - 1, len(F107A)), F107A, 'g-')

    # Generate NEUVAC data:
    neuvacIrr, perturbedNeuvacIrr, savedPerts, cc2 = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile,
                                                     statsFiles=['corMat.pkl', 'sigma_NEUVAC.pkl'])

    # Load in FISM2 data:
    euv_data_59 = read_euv_csv_file(euv_folder + 'euv_59.csv', band=False)
    mids = 0.5 * (euv_data_59['long'] + euv_data_59['short'])
    # FISM2 Results:
    fism2file = '../empiricalModels/irradiances/FISM2/daily_data_1947-2023.nc'
    myIrrTimesFISM2, wavelengthsFISM2, myIrrDataAllFISM2, myIrrUncAllFISM2 = obtainFism2(fism2file)
    # Rebin the data:
    myIrrDataWavelengthsFISM2, rebinnedIrrDataFISM2 = toolbox.rebin(wavelengthsFISM2, myIrrDataAllFISM2, euv_data_59,
                                                                    zero=False)
    rebinnedIrrUncFISM2 = np.zeros_like(rebinnedIrrDataFISM2)
    for column in range(rebinnedIrrDataFISM2.shape[1]):
        rebinnedIrrUncFISM2[:, column] = toolbox.rollingStd(rebinnedIrrDataFISM2[:, column], 2)

    # Replace bad values with NaNs:
    myIrrDataAllFISM2Fixed = rebinnedIrrDataFISM2.copy()
    myIrrDataAllFISM2Fixed[myIrrDataAllFISM2Fixed <= 0] = np.nan # For plotting

    # Make a training set from data up to Solar Cycle 25:
    times = np.array([element + +timedelta(hours=12) for element in times])
    trainIndsOMNI = np.where(times < datetime(2019, 12, 31))[0]
    trainIndsFISM2 = np.where((myIrrTimesFISM2 >= times[0]) & (myIrrTimesFISM2 < datetime(2019, 12, 31)))[0]
    trainTimesOMNI = times[trainIndsOMNI]
    trainTimesFISM2 = myIrrTimesFISM2[trainIndsFISM2]
    trainF107 = F107[trainIndsOMNI]
    trainF107A = F107A[trainIndsOMNI]
    trainNEUVAC = neuvacIrr[trainIndsOMNI, :]
    trainFISM2 = myIrrDataAllFISM2Fixed[trainIndsFISM2, :]
    trainUncFISM2 = rebinnedIrrUncFISM2[trainIndsFISM2, :]

    # Make a test set out of Solar Cycle 25:
    testIndsOMNI = np.where(times >= datetime(2019, 12, 31))[0]
    testIndsFISM2 = np.where((myIrrTimesFISM2 > datetime(2019, 12, 31)) & (myIrrTimesFISM2 <= times[testIndsOMNI][-1]))[0]
    testTimesOMNI = times[testIndsOMNI]
    testTimesFISM2 = myIrrTimesFISM2[testIndsFISM2]
    testF107 = F107[testIndsOMNI]
    testF107A = F107A[testIndsOMNI]
    testNEUVAC = neuvacIrr[testIndsOMNI, :]
    testFISM2 = myIrrDataAllFISM2Fixed[testIndsFISM2, :]
    testUncFISM2 = rebinnedIrrUncFISM2[testIndsFISM2, :]
    # ------------------------------------------------------------------------------------------------------------------
    # UNCERTAINTY ANALYSIS

    # Harmonize the times for NEUVAC and FISM2:
    correspondingIndsFISM2 = np.where((myIrrTimesFISM2 >= times[0]) & (myIrrTimesFISM2 <= times[-1]))[0]
    correspondingIrrTimesFISM2 = myIrrTimesFISM2[correspondingIndsFISM2]
    correspondingIrrFISM2 = rebinnedIrrDataFISM2[correspondingIndsFISM2, :]

    # ------------------------------------------------------------------------------------------------------------------
    # 1: Compute the normalized cross-correlation matrix between residuals in different bins.
    residualsArray = np.subtract(neuvacIrr, correspondingIrrFISM2)
    toolbox.savePickle(residualsArray, 'residualsArray.pkl')
    corMat = toolbox.mycorrelate2d(residualsArray, normalized=True)
    toolbox.savePickle(corMat, 'corMat.pkl')

    # ------------------------------------------------------------------------------------------------------------------
    # 2: Compute the normalized standard deviation of NEUVAC irradiance residuals (in each band):
    STDNeuvacResids = np.zeros(neuvacIrr.shape[1])
    for i in range(STDNeuvacResids.shape[0]):
        STDNeuvacResids[i] = np.nanstd(residualsArray[:, i])
    # Save these values to be used later for running ensembles:
    toolbox.savePickle(STDNeuvacResids, 'sigma_NEUVAC.pkl')

    # ------------------------------------------------------------------------------------------------------------------
    # 3: View the correlation matrix for the residuals of the perturbed NEUVAC irradiances alongside the base NEUVAC irradiances:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))
    pos=axs[0].imshow(corMat, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0, interpolation='none')
    axs[0].set_xlabel('Wavelength Band')
    axs[0].set_ylabel('Wavelength Band')
    axs[0].set_title('Original Correlation Matrix (NEUVAC - FISM2)')
    pos2=axs[1].imshow(cc2, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0, interpolation='none')
    axs[1].set_xlabel('Wavelength Band')
    axs[1].set_ylabel('Wavelength Band')
    axs[1].set_title('Perturbation Correlation Matrix (NEUVAC_P - NEUVAC)')
    fig.colorbar(pos, ax=axs[0])
    fig.colorbar(pos2, ax=axs[1])
    plt.savefig('Uncertainty/corMats.png', dpi=300)

    # ------------------------------------------------------------------------------------------------------------------
    # 4: Look at the Correlation between FISM2 and NEUVAC in each band, and that of (NEUVAC-FISM2)^2 and NEUVAC in each band:
    pearsonVals = []
    for i in range(neuvacIrr.shape[1]):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,6))
        # FISM2 Irradiance vs NEUVAC Irradiance
        sortInds = np.argsort(neuvacIrr[:, i])
        sortedNEUVAC = neuvacIrr[:, i][sortInds]
        sortedFISM2 = correspondingIrrFISM2[:, i][sortInds]
        popt, pcov = curve_fit(toolbox.linear, sortedNEUVAC, sortedFISM2)
        pearsonR = pearsonr(sortedNEUVAC, sortedFISM2)
        axs[0].scatter(sortedNEUVAC, sortedFISM2, color='b')
        xlims = axs[0].get_xlim()
        sample = np.linspace(xlims[0], xlims[-1], 250)
        axs[0].plot(sample, toolbox.linear(sample, *popt), 'r-', label='R='+str(np.round(pearsonR[0], 2)))
        axs[0].set_xlim(xlims)
        axs[0].set_xlabel('NEUVAC (W/m$^2$)')
        axs[0].set_ylabel('FISM2 (W/m$^2$)')
        axs[0].legend(loc='best')
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
        # Look at (Irradiance Predicted - Irradiance FISM2)^2 vs. Irradiance Predicted
        squareDiffs = toolbox.squareDiff(sortedNEUVAC, sortedFISM2)
        popt2, pcov2 = curve_fit(toolbox.linear, sortedNEUVAC, squareDiffs)
        pearsonR2 = pearsonr(sortedNEUVAC, squareDiffs)
        axs[1].scatter(sortedNEUVAC, squareDiffs, color='b')
        xlims2 = axs[0].get_xlim()
        sample2 = np.linspace(xlims2[0], xlims2[-1], 250)
        axs[1].plot(sample2, toolbox.linear(sample2, *popt2), 'r-', label='R='+str(np.round(pearsonR2[0], 2)))
        axs[1].set_xlim(xlims2)
        axs[1].set_yscale('log')
        axs[1].set_xlabel('NEUVAC')
        axs[1].set_ylabel('(NEUVAC - FISM2)$^2$ (W/m$^2$)')
        axs[1].legend(loc='best')
        axs[1].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
        fig.suptitle('FISM2 and NEUVAC Correlation and Squared Differences')
        plt.tight_layout()
        plt.savefig('Uncertainty/correlation_sqdf_band_'+str(i+1)+'.png', dpi=300)
        pearsonVals.append([pearsonR[0], pearsonR2[0]])

    # ------------------------------------------------------------------------------------------------------------------
    # Exit with a zero error code:
    sys.exit(0)