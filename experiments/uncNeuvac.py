# This script performs uncertainty quantification on the NEUVAC model using a Kalman Filter.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import matplotlib, sys
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
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
# Helper functions:
def linear(x, a, b):
    return a*x + b
def squareDiff(x, y):
    res = []
    for i in range(len(x)):
        res.append((y[i] - x[i])**2)
    return np.asarray(res)

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
    neuvacFlux, neuvacIrr = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile)

    # Load in FISM2 data:
    euv_data_59 = read_euv_csv_file(euv_folder + 'euv_59.csv', band=False)
    mids = 0.5 * (euv_data_59['long'] + euv_data_59['short'])
    # FISM2 Results:
    fism2file = '../empiricalModels/irradiances/FISM2/daily_data_1947-2023.nc'
    myIrrTimesFISM2, wavelengthsFISM2, myIrrDataAllFISM2, myIrrUncAllFISM2 = obtainFism2(fism2file, euv_data_59)
    # Rebin the data:
    myIrrDataWavelengthsFISM2, rebinnedIrrDataFISM2 = toolbox.rebin(wavelengthsFISM2, myIrrDataAllFISM2, euv_data_59,
                                                                    zero=False)
    # _, rebinnedIrrUncFISM2 = toolbox.rebin(wavelengthsFISM2, myIrrUncAllFISM2, euv_data_59,
    #                                                                zero=False, unc=True)
    rebinnedIrrUncFISM2 = np.zeros_like(rebinnedIrrDataFISM2)
    for column in range(rebinnedIrrDataFISM2.shape[1]):
        rebinnedIrrUncFISM2[:, column] = toolbox.rollingStd(rebinnedIrrDataFISM2[:, column], 2)
    # for i in range(rebinnedIrrUncFISM2.shape[1]-1):
    #     plt.figure()
    #     plt.plot(rebinnedIrrDataFISM2[:, i])
    #     plt.plot(rebinnedIrrUncFISM2[:, i])

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
    # Output the training data as a .pickle files:
    # toolbox.savePickle(trainFISM2, 'trainFISM2.pkl')
    # toolbox.savePickle(trainF107, 'trainF107.pkl')
    # toolbox.savePickle(trainF107A, 'trainF107A.pkl')
    # toolbox.savePickle(trainTimesOMNI, 'trainTimesOMNI.pkl')
    # toolbox.savePickle(trainTimesFISM2, 'trainTimesFISM2.pkl')

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

    # Perform Vector Autoregression for the FISM2 forward model:

    # # 1: Enforce stationarity by detrending the data:
    modelLag = 27
    trendWindow = 365
    # myX = rebinnedIrrDataFISM2[trainIndsFISM2, :]
    # trainFISM2trend = np.zeros_like(myX)
    # for i in range(trainFISM2trend.shape[1]):
    #     averaged_irradiance = toolbox.rollingAverage(myX[:, i], window_length=int(trendWindow), impute_edges=True)
    #     trainFISM2trend[:, i] = averaged_irradiance
    # # Divide the data by the 12-month averaged data to remove the trend:
    # detrendedFISM2 = np.divide(myX, trainFISM2trend)
    # # Employ differencing to enforce stationarity:
    # differencedDetrendedFISM2 = np.diff(detrendedFISM2, axis=0)
    # model = VAR(differencedDetrendedFISM2)
    # # 2: Fit a model with a lag of 27 days (per the suggestion of Warren, et al. 2017: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001637)
    # results = model.fit(modelLag)
    #
    # # 3: Generate a sample forecast as a sanity check:
    # # forecastResults = results.forecast(differencedDetrendedFISM2[-modelLag:, :], 5)
    # forecastResults = toolbox.forecast(differencedDetrendedFISM2[-modelLag:, :], results.coefs, results.coefs_exog.T, 5)
    # yesterday = differencedDetrendedFISM2[-1, :]
    # # View the forecast results - to invert, use the difference between the current and yesterday's F10.7, and multiply by the mean of the F10.7 of the last 6 months:
    # invertedForecastResults = toolbox.forecastInversion(forecastResults, myX, yesterday, trainFISM2trend, window=trendWindow)
    # # firstX = np.linspace(0, 26617, 26618)
    # # secondX = np.linspace(26617, 26621, 5)
    # # plt.figure()
    # # plt.plot(firstX, myX[:, 32])
    # # plt.plot(secondX, invertedForecastResults[:, 32])

    # 4: Hold off on the above - perform VAR to find a forward model for F10.7, F10.7A, AND FISM2:
    allEUVDataTrain = np.hstack( (np.vstack([trainF107, trainF107A]).T, trainFISM2) ) # trainFISM2 # np.vstack((trainF107, trainF107A)).T
    allEUVDataTest = np.hstack( (np.vstack([testF107, testF107A]).T, testFISM2) )
    allEUVUncTrain = np.hstack( (np.vstack([rollingStdF107[trainIndsOMNI], rollingStdF107A[trainIndsOMNI]]).T, trainUncFISM2) )
    allEUVUncTest = np.hstack((np.vstack([rollingStdF107[testIndsOMNI], rollingStdF107A[testIndsOMNI]]).T, testUncFISM2))
    # allEUV_forwardModel = VAR(allEUVData)
    # allEUV_result = allEUV_forwardModel.fit(modelLag)
    # allEUV_subset = allEUVData[-modelLag:, :]
    # allEUV_forecast = allEUV_result.forecast(allEUV_subset, 5)
    # plt.figure()
    # plt.plot(np.linspace(0, 26, 27), allEUV_subset[:, 0], color='b')
    # plt.plot(np.linspace(26, 31, 5), allEUV_forecast[:, 0], color='c')
    # plt.plot(np.linspace(0, 26, 27), allEUV_subset[:, 1], color='r')
    # plt.plot(np.linspace(26, 31, 5), allEUV_forecast[:, 1], color='m')

    # 5: Using the VAR as the forward model for [F10.7, F10.7A], combine it with an Unscented Kalman Filter for uncertainty quantification:

    # A: Scale ALL the FISM2 data to match the same order of magnitude as the F10.7 values:
    trainMeans = np.nanmean(allEUVDataTrain, axis=0)
    trainExps = np.array([toolbox.find_exp(element) for element in trainMeans])
    trainFactors = np.array([trainExps[0]/element for element in trainExps])
    allEUVDataTrainModified = allEUVDataTrain * trainFactors
    allEUVDataTestModified = allEUVDataTest * trainFactors
    x_state = allEUVDataTrainModified # allEUVDataTrain

    # B: Verify the VAR approach by fitting a model with a maximum lag of 27 days, and predicting 5 days out.
    forwardModel = VAR(x_state[-modelLag*2:, :])
    result = forwardModel.fit(maxlags=modelLag)
    lag_order = result.k_ar
    fcst = result.forecast(x_state[-lag_order:, :], 5)
    test = allEUVDataTestModified[:5, :]
    model_accuracy = 1 - mean_absolute_percentage_error(test, fcst)
    # print(model_accuracy)
    # C: View the results as a sanity check:
    # for i in range(x_state.shape[1]):
    #     plt.figure()
    #     plt.plot(np.linspace(0, 26, 27), x_state[-lag_order:, i], color='b')
    #     plt.plot(np.linspace(26, 31, 5), test[:, i], color='b')
    #     plt.plot(np.linspace(26, 31, 5), fcst[:, i], color='c', linestyle='--')
    #-------------------------------------------------------------------------------------------------------------------
    # Harmonize the times for NEUVAC and FISM2:
    correspondingIndsFISM2 = np.where((myIrrTimesFISM2 >= times[0]) & (myIrrTimesFISM2 <= times[-1]))[0]
    correspondingIrrTimesFISM2 = myIrrTimesFISM2[correspondingIndsFISM2]
    correspondingIrrFISM2 = rebinnedIrrDataFISM2[correspondingIndsFISM2, :]

    # Uncertainty analysis: Quantify uncertainties by comparison between NEUVAC outputs and FISM2 outputs:
    pearsons_R = []
    errorParams = []
    i = 0
    for band in range(neuvacIrr.shape[-1]):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,6))
        # FISM2 Irradiance vs NEUVAC Irradiance
        sortInds = np.argsort(neuvacIrr[:, band])
        axs[0].scatter(neuvacIrr[:, band][sortInds], correspondingIrrFISM2[:, band][sortInds], color='b')
        sample = np.linspace(neuvacIrr[:, band][sortInds][0], neuvacIrr[:, band][sortInds][-1], 1000)
        popt, pcov = curve_fit(linear, neuvacIrr[:, band][sortInds], correspondingIrrFISM2[:, band][sortInds])
        axs[0].plot(sample, linear(sample, *popt), color='r')
        axs[0].set_xlabel('NEUVAC')
        axs[0].set_ylabel('FISM2')
        # Look at (Irradiance Predicted - Irradiance FISM2)^2 vs. Irradiance Predicted
        sqdf = squareDiff(correspondingIrrFISM2[:, band], neuvacIrr[:, band])
        axs[1].scatter(neuvacIrr[:, band], sqdf, color='b')
        popt2, pcov2 = curve_fit(linear, neuvacIrr[:, band][sortInds], sqdf[sortInds])
        axs[1].plot(sample, linear(sample, *popt2), 'r-')
        pearson = pearsonr(neuvacIrr[:, band][sortInds], sqdf[sortInds])
        pearsons_R.append(pearson[0])
        errorParams.append(popt2)
        axs[1].set_yscale('log')
        axs[1].set_xlabel('NEUVAC')
        axs[1].set_ylabel('(NEUVAC - FISM2)$^2$')
        fig.suptitle('Uncertainty Analysis: Band '+str(band+1))
        plt.tight_layout()
        plt.savefig(figures_folder+'/Uncertainty_Analysis_Band_'+str(i+1)+'.png', dpi=100)
        # r_sqs.append(r_sq)
        i += 1

    # Apply the uncertainty quantification to the actual bands:
    neuvacUnc = []
    for i in range(neuvacIrr.shape[1]):
        print(linear(neuvacIrr[:, i], *errorParams[i]).shape)
        neuvacUnc.append(linear(neuvacIrr[:, i], *errorParams[i]))
    neuvacUncs = np.asarray(neuvacUnc).T

    # View the results:
    # for j in range(neuvacIrr.shape[1]):
    #     plt.figure()
    #     plt.fill_between(times, neuvacIrr[:, j]-neuvacUncs[:, j], neuvacIrr[:, j]+neuvacUncs[:, j], color='b', alpha=0.75)
    #     plt.plot(times, neuvacIrr[:, j], 'b-')

    # Save the uncertainty functions for use in the neuvac function:
    toolbox.savePickle(errorParams, 'errorParams.pkl')

    # Covariance matrix: Cross-correlation between residuals in different bins.
    residuals = []
    for i in range(neuvacIrr.shape[1]):
        residuals.append(np.subtract(neuvacIrr[:, i], correspondingIrrFISM2[:, i]))
    residualsArray = np.asarray(residuals).T

    corMat = toolbox.mycorrelate2d(residualsArray, normalized=True)
    # plt.figure()
    # plt.imshow(corMat.T, aspect='auto')
    toolbox.savePickle(corMat, 'corMat.pkl')

    # Use the covariance matrix to generate uncertainty estimates:
    meanIrradiances = np.nanmean(neuvacIrr, axis=0)

    sys.exit(0)