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
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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
    Given an array, find the index and value of an item closest to a supplie value.
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
    Given solar flux values in various bins, compute the bin-by-covariance and correlation estimates in order to yield
    estimates of uncertainty.
    :param neuvacFlux: ndarray
        An nxm matrix of NEUVAC flux values, where n is the number of observations and m is the number of wavelength
        bins.
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
        Coefficients of a polyomial fit to the data. The last element is Pearson's R.
    """
    # Loop through the columns and perform the correlation:
    fitParams = []
    sortInds = np.argsort(otherData)
    sortedOtherData = otherData[sortInds]
    referenceData = np.linspace(np.nanmin(sortedOtherData), np.nanmax(sortedOtherData), num=100)
    for i in range(myData.shape[1]):
        # Perform a linear fit:
        coeffs = np.polyfit(sortedOtherData, myData[:, i][sortInds], 1) # Default is a quartic function
        p = np.poly1d(coeffs)
        # Perform a second-order polynomial fit:
        coeffs2 = np.polyfit(sortedOtherData, myData[:, i][sortInds], 2)  # Default is a quartic function
        p2 = np.poly1d(coeffs2)
        # Perform a third-order polynomial fit:
        coeffs3 = np.polyfit(sortedOtherData, myData[:, i][sortInds], 3)  # Default is a quartic function
        p3 = np.poly1d(coeffs3)
        # Calculate Pearson's R:
        Rval = pearsonr(sortedOtherData, myData[:, i][sortInds])
        # View the data:
        plt.figure()
        plt.scatter(sortedOtherData, myData[:, i][sortInds], color='b')
        plt.plot(referenceData, p(referenceData), 'k-', label='Linear Fit')
        plt.plot(referenceData, p2(referenceData), 'c-', label='2nd-Order Polynomial Fit')
        plt.plot(referenceData, p3(referenceData), 'r-', label='3rd-Order Linear Fit')
        text = "Pearson's R: "+str(np.round(Rval[0], 2))
        plt.plot([], [], ' ', label=text)
        plt.xlabel('$\sigma_{\Theta_{\lambda}}$')
        plt.ylabel('F10.7 (sfu)')
        plt.title('$\sigma_{\Theta_{\lambda}}$ vs. F10.7 (Band '+str(i+1)+')')
        plt.legend(loc='best')
        fitParams.append([coeffs, Rval[0]])
        if saveLoc != None:
            plt.savefig(saveLoc+'neuvacStdCorrelation_Band'+str(i+1)+'.png', dpi=300)
            print('Plot saved to '+saveLoc+'neuvacStdCorrelation_Band'+str(i+1)+'.png')
    return fitParams