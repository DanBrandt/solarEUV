# This module contains various functions that can be leveraged as generic tools.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports
import pandas as pd
import numpy as np
import math, pickle
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import scipy.integrate as integ
from scipy import interpolate
#-----------------------------------------------------------------------------------------------------------------------
# Functions
def savePickle(data, pickleFilename):
    """
    Given some data (a list, dict, or array), save it is a pickle file with a user-supplied name.
    :param: data
        A variable referring to data to be saved as a pickle.
    :param: pickleFilename, str
        A string with which to name the pickle file to be saved.
    """
    with open(pickleFilename, 'wb') as pickleFile:
        pickle.dump(data, pickleFile, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(pickleFilename):
    """
    Given the name of a (pre-existing) pickle file, load its contents.
    :param: pickleFilename, str
        A string with the location/name of the filename.
    :return: var
        The loaded data.
    """
    with open(pickleFilename, 'rb') as pickleFile:
        var = pickle.load(pickleFile)
    return var

def firstNonNan(listfloats):
    """
    Find the index of the first non-NaN value in a given sequence.
    Source: https://stackoverflow.com/questions/22129495/return-first-non-nan-value-in-python-list
    :param listfloats:
    :return item:
        The element that is the first non-NaN value.
    :return idx:
        The index corresponding to the first non-NaN value.
    """
    indices = np.indices(np.asarray(listfloats).shape)[0]
    i = 0
    if np.isnan(listfloats[0]) == False:
        return listfloats[0], 0
    else:
        for item in listfloats:
            if math.isnan(item) == False:
                idx = indices[i]
                return item, idx
            i += 1

def find_nearest(array, value):
    """
    Given an array, find the index and value of an item closest to a supplied value.
    :param array: ndarray
        An array of values over which to search.
    :param value: float, int, str, datetime
        A value for which the closest value will be searched for.
    :return idx: int
        The index of the nearest value.
    :return array[idx]: float, int, str, datetime
        The actual value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def uniformSample(xdata, ydata, cadence):
    """
    Given some data with associated time stamps, resample the data so that the data have the time resolution equal to
    the given cadence.
    :param xdata: arraylike
        A list of time stamps.
    :param ydata: arraylike
        A list of data/values associated with the time stamps.
    :param cadence: int
        An integer denoting (in hours) the desired time cadence of the data.
    :return uniformXdata: ndarray
        A uniformly-sampled array of datetimes.
    :return uniformYdata: ndarray
        A uniformly-sampled array of corresponding data/values.
    """
    # Instantiate an array of timestamps with the desired cadence:
    start = datetime(xdata[0].year, xdata[0].month, xdata[0].day, 12)
    end = datetime(xdata[-1].year, xdata[-1].month, xdata[-1].day, 12)
    uniformXdata = np.arange(start, end, timedelta(hours=cadence)).astype(datetime)
    # Get indices of values nearest to the time array above:
    uniformIndices = []
    for i in range(len(uniformXdata)):
        goodIndex, goodValue = find_nearest(xdata, uniformXdata[i])
        uniformIndices.append( goodIndex )
    uniformIndicesArray = np.asarray(uniformIndices)
    # Extract the good values:
    uniformYdata = ydata[uniformIndicesArray]
    return uniformXdata, uniformYdata

def imputeData(timestamps, values, method='mean', bad_values=np.nan):
    """
    Given timeseries data, impute bad values and returned the cleaned data.
    :param timestamps: arraylike
        A 1D list or array of timestamps.
    :param values: arraylike
        A 1D list or array of timeseries data.
    :param method: str
        A string indicating the method to be used. If 'mean', 'median', 'most_frequent', or 'constant', uses the
        SimpleImputer routine from sklearn. If 'gam', uses a Generalized Additive Model to fill in the data. If 'gpr',
        use Gaussian Process Regression to perform imputation. Otherwise, cubic spline interpolation is used for
        imputation.
    :param bad_values: float, int, or NaN (type)
        The type of the data you wish to impute.
    :return cleanTimes: ndarray
        The timestamps corresponding to the cleaned data. This will differ from 'timestamps' if and only if some data
        are necessarily excluded by the chosen imputation method. For example, the edges of the data are often removed
        when the cubic spline interpolation is used for imputation, to avoid Runge's phenomenon.
    :return cleanData: ndarray
        The cleaned 1D timeseries.
    """
    cleanTimes = timestamps
    # Replace the bad values with NaNs:
    if bad_values != np.nan:
        bad_inds = np.where(values==bad_values)[0]
        values[bad_inds] = np.nan
    # Use the SimpleImputer routine from sklearn:
    if method=='mean' or method=='median' or method=='most_frequent' or method=='constant':
        imp = SimpleImputer(missing_values=np.nan, strategy=method)
        imp.fit(np.asarray(values).reshape(-1, 1))
        cleanData = imp.transform(values.reshape(-1, 1))
    elif method=='gpr':
        # TODO: Fix the Gaussian Process approach below so that sensible values are imputed:
        subset = values # [100:10000]
        DOYvals = np.array([fractionalDOY(element) for element in timestamps])
        DOYsubset = DOYvals # [100:10000]
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        XPred = np.asarray(DOYsubset).reshape(-1, 1)
        XAxis = np.linspace(0,len(subset)-1,len(subset)).reshape(-1, 1)
        X = XAxis[~np.isnan(subset)]
        y = subset[~np.isnan(subset)].reshape(-1, 1)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
        yMeanPred, yStdPred = gpr.predict(XAxis, return_std=True)
        # plt.figure(); plt.plot(XAxis, yMeanPred, linestyle='-.'); plt.plot(XAxis, subset); plt.show()
    elif method=='gam':
        # Use a GAM to parameterize perform gap-filling:
        DOY = np.array([fractionalDOY(element) for element in timestamps])
        from pygam import LinearGAM, s
        X = DOY[~np.isnan(values)]
        y = values[~np.isnan(values)]
        gam = LinearGAM(s(0)).fit(X, y)
        cleanedData = gam.predict(DOY[np.isnan(values)])
        cleanData = values.copy()
        cleanData[np.isnan(values)] = cleanedData
    else:
        # Clip the ends of the data:
        firstNonNanValue, firstNonNanIndex = firstNonNan(values)
        lastNonNanValue, lastNonNanIndex = firstNonNan(np.flip(values, -1))
        lastNonNanIndex = len(values) - lastNonNanIndex
        subset = values[firstNonNanIndex:lastNonNanIndex]
        cleanTimes = timestamps[firstNonNanIndex:lastNonNanIndex]
        # Perform the imputation
        XAxis = np.linspace(0, len(subset) - 1, len(subset)).reshape(-1, 1)
        X = XAxis[~np.isnan(subset)]
        y = subset[~np.isnan(subset)].reshape(-1, 1)
        spl = InterpolatedUnivariateSpline(X, y, k=3)
        cleanData = spl(XAxis)
        # View the results as a sanity check:
        # plt.figure();
        # plt.plot(XAxis, cleanData, linestyle='-.');
        # plt.plot(XAxis, subset);
        # plt.show()

    # plt.figure(); plt.plot(timestamps, values); plt.plot(cleanTimes, cleanData); plt.show()

    return cleanTimes, np.squeeze(cleanData)

def gapify(timeseries, bad_value=999, replace_val=np.nan):
    """
    Take a timeseries and replace the bad values (signified by 'bad_values') with NaNs, unless otherwise desired.
    :param timestamps: arraylike
        A 1D list or array of timestamps.
    :param bad_values: int or float
        A float or int corresponding to the bad values to be 'gapped out'. Default is 999.
    :param replace_val: int, float, NaN, or None
        An int, float, NaN, or Nonetype describing what will replace the bad values.
    :return gappedData:
        The gapified 1D data.
    """
    gappedData = timeseries.copy()
    bad_inds = np.where(gappedData == bad_value)[0]
    gappedData[bad_inds] = replace_val
    return gappedData

def fractionalDOY(myDatetime):
    """
    Convert a generic datetime object into a float corresponding to the fractional day of the year.
    :param: myDatetime: datetime
        A datetime object.
    :return: fracDOY: float
        A float corresponding to the fractional Day of the Year, with the decimal portion included to show contributions
        from the time of day in hours, minutes, and seconds.
    """
    fracDOY = myDatetime.timetuple().tm_yday + myDatetime.hour/24. + myDatetime.minute/3600. + myDatetime.second/86400.
    return fracDOY

# TODO: Fix the function below to handle VERY LARGE data gaps.
def rollingAverage(myData, window_length=1, impute_edges=True):
    """
    Using pandas, compute a rolling average of over 'data' using a window length of 'windowlength'. Sets the leading and
    trailing windows to the values of the original data.
    :param myData: arraylike
        The data over which to compute the rolling average.
    :param window_length: int
        The size of the window over which to average.
    :param impute_edges: bool
        A boolean determining whether or not the edges will be interpolated. Default is True.
    :return: rolled, arraylike
        The rolling average data.
    """
    myDataframe = pd.DataFrame(data=myData, columns=['Var'])
    myDataframe['Rolling'] = myDataframe['Var'].rolling(window=window_length, center=True).mean()
    if impute_edges == True:
        # Sample x-axis:
        sampleXaxis = np.linspace(0, window_length, window_length)
        middleIndex = int(0.5*window_length)
        # Use cubic interpolation to fill the gaps on the edges:
        # leadingEdgeStartingVal = myDataframe['Var'][0]
        goodLeadingVals = myDataframe['Var'][:window_length][myDataframe['Var'][:window_length] > 0]
        # leadingEdgeStartingVal = np.percentile(goodLeadingVals.values, 25)
        leadingEdgeStartingVal = np.min(goodLeadingVals.values)
        leadingEdgeMiddleVal = np.mean([leadingEdgeStartingVal, myDataframe['Var'][:window_length].values[-1]]) # myDataframe['Var'][:window_length].mean()
        leadingEndingVal = myDataframe['Var'][:window_length].values[-1]
        leadingSpline = CubicSpline([sampleXaxis[0], sampleXaxis[middleIndex], sampleXaxis[-1]],
                                    [leadingEdgeStartingVal, leadingEdgeMiddleVal, leadingEndingVal])
        leadingImputedValues = leadingSpline(sampleXaxis)
        # plt.figure(); plt.plot(sampleXaxis, myDataframe['Var'][:window_length].values); plt.plot(sampleXaxis, leadingImputedValues); plt.show()

        trailingEdgeStartingVal = myDataframe['Var'][-window_length:].values[0]
        trailingEdgeMiddleVal = myDataframe['Var'][-window_length:].mean()
        trailingEndingVal = myDataframe['Var'].values[-1]
        trailingSpline = CubicSpline([sampleXaxis[0], sampleXaxis[middleIndex], sampleXaxis[-1]],
                                    [trailingEdgeStartingVal, trailingEdgeMiddleVal, trailingEndingVal])
        trailingImputedValues = trailingSpline(sampleXaxis)
        # plt.figure(); plt.plot(sampleXaxis, myDataframe['Var'][-window_length:].values); plt.plot(sampleXaxis, trailingImputedValues); plt.show()

        myDataframe['Rolling'][:window_length] = leadingImputedValues
        myDataframe['Rolling'][-window_length:] = trailingImputedValues
    else:
        myDataframe['Rolling'][:window_length] = myDataframe['Var'][:window_length]
        myDataframe['Rolling'][-window_length:] = myDataframe['Var'][-window_length:]
    rolled = myDataframe['Rolling'].values
    return rolled

def rollingStd(myData, window_length=2, axis=-1):
    """
    Given some data, compute the rolling standard deviation. If the data is two dimensional, compute the rolling
    standard deviation along a specific axis of the data (specified by the user).
    :param myData: arraylike
        The data over which to compute the rolling average.
    :param window_length: int
        The size of the window over which to average. Default is 2.
    :param axis: int
        For 2D data, the axis along which to compute the rolling standard deviation. Defaults is -1.
    :return stdData: ndarray
        The rolling standard deviation values of the data.
    """
    # Define the generic rolling std function to be used repeatedly:
    def stdRoller(data, windowLength):
        myDataframe = pd.DataFrame(data=data, columns=['Var'])
        myDataframe['Rolling'] = myDataframe['Var'].rolling(window=windowLength, center=True).std()
        # Set the leading and trailing values equal to the mean standard deviation:
        meanStd = np.nanmean(myDataframe['Rolling'].values)
        infillData = np.full_like(myDataframe['Rolling'][:windowLength].values, fill_value=meanStd)
        myDataframe['Rolling'][:windowLength] = infillData
        myDataframe['Rolling'][-windowLength:] = infillData
        stdRes = myDataframe['Rolling'].values
        return stdRes
    # Perform the computation:
    if len(myData.shape) < 2:
        # Case for unidimensional data:
        stdData = stdRoller(myData, window_length)
    else:
        # Case for 2D data: Iterate through the data along the desired axis:
        stdData = np.zeros_like(myData)
        if axis == 0:
            # First Axis:
            for iRow in range(myData.shape[0]-1):
                stdData[iRow, :] = stdRoller(myData[iRow, :], window_length)
        elif axis == 1 or axis == -1:
            # Second Axis:
            for iCol in range(myData.shape[1]-1):
                stdData[:, iCol] = stdRoller(myData[:, iCol], window_length)
        else:
            raise ValueError('The axis specified exceeds the dimensions of the data.')
    return stdData

def normalize(myData, axis=-1):
    """
    Normalize data with respect to the mean, along the axis specified by the user.
    :param myData: ndarray
        A 1d or 2d array of data.
    :param axis: int
        The axis along which to perform normalization with respect to the mean. Default is -1.
    :return normedData: ndarray
        The normalized data.
    """
    def normFunc(data):
        return (data - np.nanmean(data)) / np.nanstd(data)
    if len(myData.shape) < 2:
        normedData = normFunc(myData)
    else:
        normedData = np.zeros_like(myData)
        if axis == 0:
            # First dimension:
            for iRow in range(myData.shape[0]-1):
                normedData[iRow, :] = normFunc(myData[iRow, :])
        elif axis==1 or axis==-1:
            # Second dimension:
            for iCol in range(myData.shape[1]-1):
                normedData[:, iCol] = normFunc(myData[:, iCol])
        else:
            raise ValueError('The axis specified exceeds the dimensions of the data.')
    return normedData

def covariates(neuvacFlux):
    """
    Given solar flux or irradiance values in various bins, compute the bin-by-covariance and correlation estimates in
    order to yield estimates of uncertainty.
    :param neuvacFlux: ndarray
        An nxm matrix of NEUVAC flux/irradiance values, where n is the number of observations and m is the number of
        wavelength bins.
    :return cov: ndarray
        The associated correlation matrix.
    """
    binStrings = ['Bin' + str(i[0] + 1) for i in enumerate(neuvacFlux[0, :])]
    df = pd.DataFrame(neuvacFlux, columns=binStrings)
    # View the data:
    # df.plot(marker='.')
    # Compute the covariances and covariances normalized by their respective standard deviations (https://stackoverflow.com/questions/63138921/covariance-of-two-columns-of-a-dataframe):
    cov = df.cov()
    # Visualize the Covariance Matrix:
    plt.matshow(cov)
    corr = df.corr()
    # Visualize the Correlation Matrix:
    plt.matshow(corr)
    # TODO: Label and save the correlation matrix.
    return corr

def corrCol(myData, otherData, saveLoc=None):
    """
    Given two sets of data, find the correlation between them. If the first set of data
    is multidimensional, correlate it with the second set by column. This function requires that
    'otherData' is 1D and equivalent to the lengths of the columns of 'myData'. Otherwise, if 'myData'
    is 1D, it must be the same length as 'otherData'.
    :param myData: ndarray
        The dependent variable data.
    :param otherData: ndarray
        The independent variable data.
    :param saveLoc: str
        A location where to save figures of the correlations. Optional argument. Default is None.
    :return fitParams: ndarray
        A poly1d object for the fit of the data. The last element is Pearson's R.
    """
    # Loop through the columns and perform the correlation:
    fitParams = []
    sortInds = np.argsort(otherData)
    sortedOtherData = otherData[sortInds]
    referenceData = np.linspace(np.nanmin(sortedOtherData), np.nanmax(sortedOtherData), num=100)
    for i in range(myData.shape[1]):
        currentData = myData[:, i][sortInds]
        # Find the best model fit, up to a polynomial of order 10:
        p, rss, order = bestPolyfit(sortedOtherData, currentData, 1)
        Rval = pearsonr(sortedOtherData, currentData)
        # View the data:
        plt.figure()
        plt.scatter(sortedOtherData, currentData, color='b')
        plt.plot(referenceData, p(referenceData), 'r-', label='Model Fit: Order '+str(order)+' (RSS= '+str(np.round(rss, 2))+')')
        text = "Pearson's R: "+str(np.round(Rval[0], 2))
        plt.plot([], [], ' ', label=text)
        plt.ylabel('$\sigma_{r_{\Theta_{\lambda}}}$')
        plt.xlabel('F10.7 (sfu)')
        plt.title('$\sigma_{r_{\Theta_{\lambda}}}$ vs. F10.7 (Band '+str(i+1)+')')
        plt.legend(loc='best')
        fitParams.append([p, Rval[0]])
        if saveLoc != None:
            plt.savefig(saveLoc+'neuvacResidStdCorrelation_Band'+str(i+1)+'.png', dpi=300)
            print('Plot saved to '+saveLoc+'neuvacResidStdCorrelation_Band'+str(i+1)+'.png')
    return fitParams

def bestPolyfit(xdata, ydata, maxOrder=5, func=None, **kwargs):
    """
    Given independent variable and dependent variable 1D data fit the data with polynomial functions of orders up to a
    user-defined limit set to 'maxOrder'. For the best-fitting model, return the parameters for the polynomial (as a
    poly1d object) along with the associated Residual Sum of Squares.
    :param xdata: ndarray
        1D independent variable data.
    :param ydata: ndarray
        1D dependent variable data.
    :param maxOrder: int
        A number below which (inclusive) to consider models to fit to the data. Default is 5.
    :param func: str
        Specifies what function should be used for fitting. Valid strings are: 'exp', 'linear', 'log', and 'cubic'. If
        this argument is passed, maxOrder is ignored.
    :return modelRes: list
        A list where the first element are the model parameters, the second is the Residual Sum of Squares, and the
        third is the model order.
    """
    # Example functions:
    if func is not None:
        if func == 'exp':
            def myFunc(x, a, b, c, d):
                return a*np.exp(-b*x + c) + d
        elif func == 'cubic':
            def myFunc(x, a, b, c, d):
                return a*x**3 + b*x**2 + c*x + d
        elif func == 'linear':
            def myFunc(x, a, b):
                return a*x + b
        elif func == 'log':
            def myFunc(x, a, b):
                a*np.log(x) + b
        elif func == 'quadratic':
            def myFunc(x, a, b, c):
                return a*x**2 + b*x + c
        else:
            raise ValueError('Invalid argument supplied for argument "func".')

    models = []
    rss_vals = []
    orders = []
    # Fit models of various orders:
    if func is not None:
        if func == 'exp':
            p0 = [2e-3, 2e-1, 3, 8e-7]
            newP0 = [2e-3, 0.1, np.nanmax(ydata), np.nanmin(ydata)]
        else:
            p0 = None
        popt, pcov = curve_fit(myFunc, xdata, ydata, p0=p0)
        # If convergence is not achieved, try again with new initializing parameters:
        if np.where(np.isinf(pcov))[0].shape[0] > 0:
            popt, pcov = curve_fit(myFunc, xdata, ydata, p0=newP0)
        p = popt
        # Plotting for a sanity check:
        sampleXdata = np.linspace(np.nanmin(xdata), np.nanmax(xdata), num=100)
        # plt.figure(); plt.scatter(xdata, ydata, color='b', label='Data'); plt.plot(sampleXdata, myFunc(sampleXdata, *popt), 'r-', label='Fit')
        # plt.plot(sampleXdata, myFunc(sampleXdata, *newP0), color='orange', linestyle='--'); plt.legend(loc='best')
        # Compute the R-squared (https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit):
        residuals = ydata - myFunc(xdata, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        rss_vals.append(r_squared)
        models.append([myFunc, p])
        modelRes = [[myFunc, p], r_squared, func]
    else:
        for i in range(maxOrder):
            coeffs = np.polyfit(xdata, ydata, i+1)
            p = np.poly1d(coeffs)
            rss_vals.append(np.sum((ydata - p(xdata)) ** 2))
            models.append(p)
            orders.append(i+1)
        # Determine which model worked the best:
        locBest = np.nanargmin(np.asarray(rss_vals))
        modelRes = [models[locBest], rss_vals[locBest], orders[locBest]]
    return modelRes

def binRMSE(xdata, ydataEst, ydataTrue, step=10, saveLoc=None, titleStr=None, normalize=False):
    """
    Given some 1D independent variable data, some 1D estimates of dependent variable data, 1D true values of dependent
    variable data, and a step size, divide the xdata into bins of width equal to the step size and compute the RMSE
    error in each bin. Then compute the correlation between the the RMSE and the binned xdata. Automatically saves
    a figure for the results at a user-defined location.
    :param xdata: ndarray
        1D independent variable data.
    :param ydataEst: ndarray
        1D estimates of dependent variable data.
    :param ydataEst: ndarray
        1D actual values of dependent variable data.
    :param step: int
        The bin width for the independent variable data. Default is 10.
    :param saveLoc: str
        A string for the location where the figure should be saved.
    :param titleStr: str
        A string for the title of the figure to be generated. Assumes that a single number representing a figure number
        or wavelength is given.
    :param normalize: Bool
        Controls whether the RMSE is normalized or not. Default is False.
    :return binCenters: list
        The bin centers for the dependent variable data.
    :return RMSE: list
        The RMSE values for each bin.
    """
    # Create the bins:
    start = round_mult(np.nanmin(xdata), step, direction='down')
    stop = round_mult(np.nanmax(xdata), step, direction='up')
    bins = np.arange(start, stop, step=step)
    binCenters = np.asarray([(a + b) / 2 for a, b in zip(bins[::2], bins[1::2])])
    # Loop the bins and compute the RMSE values:
    RMSE = []
    i = 0
    for element in binCenters:
        # Isolate the data that correspond to the given bin:
        goodInds = np.where((xdata >= bins[2*i-1]) & (xdata <= bins[2*i]))[0]
        if len(goodInds) > 0:
            goodEstData = ydataEst[goodInds]
            goodTrueData = ydataTrue[goodInds]
            rmse = mean_squared_error(goodTrueData[~np.isnan(goodTrueData)], goodEstData[~np.isnan(goodTrueData)], squared=False)
            if normalize == True:
                rmse = rmse / (np.nanmax(goodTrueData) - np.nanmin(goodTrueData))
        else:
            # If there is nothing in the bin, just record NaN for that bin:
            rmse = np.nan
        RMSE.append(rmse)
        i += 1
    # Plot the results for a sanity check:
    plt.figure()
    plt.plot(binCenters, RMSE, 'bo-')
    plt.xlabel('F10.7 (sfu)')
    plt.ylabel('RMSE (W/m$^2$/nm)')
    if titleStr != None:
        plt.suptitle('RMSE vs. F10.7: '+titleStr+' Angstroms')
    if saveLoc != None:
        plt.savefig(saveLoc+'RootMeanSquareDeviationVsF107_'+titleStr.replace('.', '_')+'.png', dpi=300)
    return binCenters, RMSE

def binCorrelation(xdata, ydataEst, ydataTrue, step=10, saveLoc=None, titleStr=None, root=False, normalize=False):
    """
    Given some 1D independent variable data, some 1D estimates of dependent variable data, 1D true values of dependent
    variable data, and a step size, divide the xdata into bins of width equal to the step size and compute the squared
    difference in each bin. Then compute the correlation between the the squared difference and the binned xdata.
    Automatically saves a figure for the results at a user-defined location.
    :param xdata: ndarray
        1D independent variable data.
    :param ydataEst: ndarray
        1D estimates of dependent variable data.
    :param ydataEst: ndarray
        1D actual values of dependent variable data.
    :param step: int
        The bin width for the independent variable data. Default is 10.
    :param saveLoc: str
        A string for the location where the figure should be saved.
    :param titleStr: str
        A string for the title of the figure to be generated. Assumes that a single number representing a figure number
        or wavelength is given.
    :param root: Bool
        Controls whether the square root of the SQDF is taken. Default is False.
    :param normalize: Bool
        Controls whether the SQDF is normalized (by dividing by the true data and multiplying by 100). Default is False.
        If True, ignores the value of 'root'.
    :return binCenters: list
        The bin centers for the dependent variable data.
    :return SQDF: list
        The Squared Difference values for each bin.
    """
    # Create the bins:
    start = round_mult(np.nanmin(xdata), step, direction='down')
    stop = round_mult(np.nanmax(xdata), step, direction='up')
    bins = np.arange(start, stop, step=step)
    binCenters = np.asarray([(a + b) / 2 for a, b in zip(bins[::2], bins[1::2])])
    # Loop the bins and compute the RMSE values:
    SQDF = []
    i = 0
    for element in binCenters:
        # Isolate the data that correspond to the given bin:
        goodInds = np.where((xdata >= bins[2*i-1]) & (xdata <= bins[2*i]))[0]
        if len(goodInds) > 0:
            goodEstData = ydataEst[goodInds]
            goodTrueData = ydataTrue[goodInds]
            N = len(goodTrueData[~np.isnan(goodTrueData)])
            sqdf = np.mean(
                np.square(np.subtract(goodEstData[~np.isnan(goodTrueData)], goodTrueData[~np.isnan(goodTrueData)])) / N)
            ylabelString = 'Squared Differences (W/m$^2$/nm)'
            realTitleString = 'Squared Differences'
            if normalize == False:
                if root == True:
                    sqdf = np.sqrt(sqdf)
                    ylabelString = 'RMSE (W/m$^2$/nm)'
                    realTitleString = 'RMSE'
            else:
                sqdf = np.mean(np.divide(
                    np.square(np.subtract(goodEstData[~np.isnan(goodTrueData)], goodTrueData[~np.isnan(goodTrueData)])),
                    goodTrueData[~np.isnan(goodTrueData)]) * 100)
                ylabelString = 'Normalized RMSE (W/m$^2$/nm)'
                realTitleString = 'Normalized RMSE'
        else:
            # If there is nothing in the bin, just record NaN for that bin:
            sqdf = np.nan
        SQDF.append(sqdf)
        i += 1
    # Plot the results for a sanity check:
    titleFontSize = 20
    fontSize = 18
    labelSize = 16
    plt.figure(figsize=(12, 8))
    plt.plot(binCenters, SQDF, 'mo-')
    plt.xlabel('F10.7 (sfu)', fontsize=fontSize)
    plt.ylabel(ylabelString, fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=labelSize)
    if titleStr != None:
        plt.suptitle(realTitleString+' vs. F10.7: '+titleStr+' Angstroms', fontsize=titleFontSize)
    if saveLoc != None:
        plt.savefig(saveLoc+realTitleString.replace(' ', '_')+'vsF107_'+titleStr.replace('.', '_')+'.png', dpi=300)
    return binCenters, SQDF

def round_mult(num, divisor, direction='down'):
    """
    Round a number to the nearest integer multiple of a given divisor.
    :param num: int or float
        The number to round down.
    :param divisor: int or float
        The number which the result must be an integer multiple of.
    :param direction: str
        Either 'down' or 'up'. Specifies whether rounding should be done up or down/
    :return rounded: float
        The resulting number.
    """
    if direction == 'down':
        return num - (num%divisor)
    else:
        return divisor*(round(num/divisor))

def stringList(myList):
    """
    Convert a list of numbers into a single string.
    :param myList: list
        A list of numbers.
    :return myStr: str
        The string.
    """
    strList = [str(element) for element in myList]
    myStr = " ".join(strList)
    return myStr

def rebin(wavelengths, data, resolution, limits=None, factor=None, zero=True):
    """
    Rebin 2D [irradiance/flux] data, where each row is a complete spectrum, and each column is a specific wavelength
    bin. Do the rebinning according to a desired resolution, and restrict the returned information to wavelength
    boundaries set by the user.
    :param wavelengths: arraylike
        A list of wavelengths at which each irradiance value is taken.
    :param data: ndarray
        A 2D array of spectrum data (irradiances or fluxes).
    :param resolution: float or dict
        The desired spectral resolution of the rebinned data. Units must be in nanometers. If argument is a dict, the
        key 'short' should contain wavelength boundaries for the beginning of bins, and the key 'long' should contain
        wavelength boundaries for the ending of bins. These boundaries should be in Angstroms.
    :param limits: list
        A list of two elements, where the first is a lower limit for wavelengths and the second is an upper limit for
        wavelengths, both in units of nanometers.
    :param factor: int
        A factor by which to upsample the data before performing rebinning. Tends to make the binning much more
        accurate, due to the use of integrals. Default is None.
    :param zero: bool
        Controls whether singular lines are set to a value of zero after they are extracted. Default is True.
    :return newWaves: ndarray
        The centers of the wavelength bins corresponding to the rebinned data.
    :return newData: ndarray
        The rebinned data as a 2D array, arranged like the input data but at the new wavelength resolution.
    """
    # Get the native wavelength resolution of the input data:
    nativeResolution = np.concatenate((np.diff(wavelengths), np.array([np.diff(wavelengths)[-1]])), axis=0)
    nativeWavelengths = wavelengths.copy()

    # Upsample the wavelengths and irradiance data so that integrals are more accurate:
    if factor is not None:
        if len(data.shape) < 2:
            f = interpolate.interp1d(wavelengths, data, kind='linear')
            wavelengths = np.linspace(wavelengths[0], wavelengths[-1], factor * len(wavelengths))
            data = f(wavelengths)
        else:
            # Replace all NaNs with zeros:
            data[np.isnan(data)] = 0
            obs = np.linspace(0, data.shape[0], data.shape[0])
            f = interpolate.interp2d(wavelengths, obs, data, kind='linear')
            wavelengths = np.linspace(wavelengths[0], wavelengths[-1], factor*len(wavelengths))
            data = f(wavelengths, obs)

    # Loop through each row the new data array, and fill it in with the rebinned data:
    if type(resolution) is not dict:
        # TODO: The code for this if cell is VERY SLOW when processing large arrays. It needs to be sped up.
        # CONDITION FOR A SINGLE WAVELENGTH RESOLUTION.
        # Compute the wavelength centers for the new bins:
        if limits is not None:
            newWaves = np.arange(limits[0], limits[-1] + resolution, resolution)
        else:
            step = np.nanmean(np.diff(wavelengths))
            newWaves = np.arange(wavelengths[0], wavelengths[-1] + step, step)
        # Instantiate the new data array:
        if len(data.shape) < 2:
            newData = np.zeros((1, newWaves.shape[0]))
        else:
            newData = np.zeros((data.shape[0], newWaves.shape[0]))
        # Loop over each spectrum (each individual observation):
        for i in tqdm(range(newData.shape[0])):
            # Loop through the wavelength intervals:
            currentSpectrum = np.zeros_like(newWaves)
            for j in range(newWaves.shape[0]-1):
                # Get the indices of elements in each wavelength bin - ASSUME that the original data are associated with
                # bin centers - apply an offset equal to the respective interval of the bin:
                binInds = np.where((wavelengths > (newWaves[j] - resolution)) & (wavelengths < (newWaves[j] + resolution)))[0]
                # Gather the elements in the bin and multiply them by the respective native bin width:
                try:
                    binElements = data[i, binInds]
                except:
                    binElements = data[binInds]

                binWidths = nativeResolution[binInds]
                binProducts = [a*b for a,b in zip(binElements, binWidths)]
                # Take the sum:
                binSum = np.sum(binProducts)

                # Integrate the bin:
                binVal = integ.trapz(binElements, wavelengths[binInds])
                # binVal = binSum/(2*resolution)

                # Add the value to the current spectrum:
                currentSpectrum[j] = binVal

            # Once a single spectrum is obtained, add it to the new data array:
            newData[i, :] = currentSpectrum
    else:
        # CONDITION FOR A BINNING WITH UNIQUE BIN WIDTHS.
        shorts = resolution['short']/10.
        longs = resolution['long']/10.
        newWaves = 0.5*(shorts + longs)

        # Instantiate the new data array:
        if len(data.shape) < 2:
            newData = np.zeros((1, newWaves.shape[0]))
        else:
            newData = np.zeros((data.shape[0], newWaves.shape[0]))

        # First go through all of the wavelengths that are singular
        myData = data
        for iWave, short in enumerate(shorts):
            long = longs[iWave]
            if (long == short):
                i = np.argmin(np.abs(wavelengths - short))
                i2 = np.argmin(np.abs(nativeWavelengths - short))
                try:
                    newData[:, iWave] = myData[:, i] * (nativeWavelengths[i2 + 1] - nativeWavelengths[i2])
                except:
                    newData[:, iWave] = myData[i] * (nativeWavelengths[i2 + 1] - nativeWavelengths[i2])
                if zero == True:
                    # Zero out bin so we don't double count it.
                    try:
                        myData[:, i] = np.zeros_like(myData[:, i])
                    except:
                        myData[i] = 0.0

        # Then go through the ranges
        for iWave, short in enumerate(shorts):
            long = longs[iWave]
            if (long != short):
                d1 = np.abs(wavelengths - short)
                iStart = np.argmin(d1)
                d2 = np.abs(wavelengths - long)
                iEnd = np.argmin(d2)
                wave_int = 0.0
                # For wavelengths at or below 0.2 nm, just compute the sum:
                if long <= 0.2:
                    for i in range(iStart + 1, iEnd + 1):
                        newData[:, iWave] += myData[:, i] * \
                                             (wavelengths[i + 1] - wavelengths[i])
                        wave_int += (wavelengths[i + 1] - wavelengths[i])
                else:
                    try:
                        newData[:, iWave] = integ.trapz(myData[:, iStart:iEnd], wavelengths[iStart:iEnd], axis=1)
                    except:
                        # Resample the wavelength data:
                        newData[:, iWave] = integ.trapz(myData[iStart:iEnd], wavelengths[iStart:iEnd])

                    # # Plotting for a sanity check:
                    # plt.figure();
                    # plt.plot(wavelengths[iStart:iEnd], myData[iStart:iEnd], marker='o')
                    # plt.scatter(newWaves[iWave], newData[:, iWave])


                # for i in range(iStart + 1, iEnd + 1):
                #     binWidths = nativeResolution[iStart + 1:iEnd + 1]
                #     try:
                #         binProducts = [a * b for a, b in zip(myData[:, i], binWidths)]
                #         binSum = np.sum(binProducts)
                #         newData[:, iWave] = binSum / (iEnd - iStart)
                #         # newData[:, iWave] += myData[:, i]  / (iEnd - iStart) # * (wavelengths[i + 1] - wavelengths[i])
                #     except:
                #         binProducts = [a * b for a, b in zip([myData[i]], binWidths)]
                #         binSum = np.sum(binProducts)
                #         newData[:, iWave] = binSum / (iEnd - iStart)
                #         # newData[:, iWave] += (myData[i] * () )/ (iEnd - iStart)
                #     wave_int += (wavelengths[i + 1] - wavelengths[i])

    # If only a single spectrum was generated, remove the singleton dimension:
    if newData.shape[0] == 1:
        newData = np.squeeze(newData)

    return newWaves, newData
