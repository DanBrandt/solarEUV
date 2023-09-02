# This module contains functions used for processing/loading in solar indices.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports
import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local imports:
from toolbox import uniformSample, imputeData, rollingAverage, savePickle
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Functions
def readF107(filename):
    """
    Load in F10.7 data from the files obtained from the FTP server provided by the Government of Canada:
    https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-en.php
    Requires ONLY a filename containing F10.7 data.
    Note that F10.7 values are daily values measured at LOCAL NOON of each day.
    Note that the OBSERVED values DO NOT correspond to 1 AU, and they may vary, while the ADJUSTED values are calibrated
    to correspond to 1 AU.
    :param filename: str
        A string containing the location of the txt file with the F10.7 data.
    :return times: ndarray
        A 1D array of datetimes corresponding to each F10.7 measurement.
    :return f107: ndarray
        A 1D array of F10.7 values.
    """
    times = []
    f107 = []
    i = 0
    # Loop through the file to collect the data:
    with open(filename, 'r') as myFile:
        for line in myFile:
            # Note that the line structure is different depending on what file is downloaded:
            if filename[-11:-4] != 'current':
                if i > 2:
                    if filename[-8:-4] == '2007':
                        elements = line.split()
                        try:
                            f107.append(float(elements[7])) # Index 6 = Observed; Index 7 = Adjusted
                        except:
                            f107.append(np.nan)
                        currentTime = pd.to_datetime(float(elements[0]), origin='julian', unit='D')
                    else:
                        elements = line.split(',')
                        try:
                            f107.append(float(elements[6])) # Index 5 = Observed; Index 6 = Adjusted
                        except:
                            elements = line.split(' ')
                            f107.append(float(elements[-1]))
                        currentTime = pd.to_datetime(float(line.split(',')[0]), origin='julian', unit='D')
                    times.append(currentTime.to_pydatetime())
            else:
                if i > 3:
                    elements = line.split()
                    f107.append(float(elements[8])) # Index 8 = Adjusted
                    currentTime = pd.to_datetime(float(elements[0]), origin='julian', unit='D')
                    times.append(currentTime.to_pydatetime())
            i += 1

    return np.asarray(times), np.asarray(f107)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    saveLoc = '../solarIndices/F107/'
    fname = '../solarIndices/F107/F107_1947_1996.txt'
    fname1 = '../solarIndices/F107/F107_1996_2007.txt'
    fname2 = '../solarIndices/F107/F107_current.txt'
    times, f107 = readF107(fname)
    times1, f1071 = readF107(fname1)
    times2, f1072 = readF107(fname2)
    # -------------------------------
    # Combine all of the outputs from the files above and view the results as a sanity check:
    allTimes = np.concatenate((times, times1, times2))
    allF107 = np.concatenate((f107, f1071, f1072))
    # Resample the times:
    uniformTimes, uniformF107 = uniformSample(allTimes, allF107, cadence=24)
    # Clean the data (through either gapification or imputation):
    cleanedTimes, cleanedF107 = imputeData(uniformTimes, uniformF107, method='interp', bad_values=0)
    # cleanedF107 = gapify(uniformF107, bad_value=0)
    # Compute the centered rolling 81-day average of F10.7:
    averagedF107 = rollingAverage(cleanedF107, window_length=81)
    # Plot as a sanity-check:
    plt.figure(); plt.plot(cleanedTimes, cleanedF107); plt.plot(cleanedTimes, averagedF107); plt.show()
    # Save the data to pickle files:
    savePickle(cleanedTimes, saveLoc+'F107times.pkl')
    savePickle(cleanedF107, saveLoc+'F107vals.pkl')
    savePickle(averagedF107, saveLoc+'F107averageVals.pkl')
    # -------------------------------
    sys.exit(0)