# This module contains functions used for processing/loading in solar irradiances.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports
import os, sys
from tqdm import tqdm
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Qt5Agg')
from netCDF4 import Dataset
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports
from tools.EUV.fism2_process import read_euv_csv_file
from tools.EUV.fism2_process import rebin_fism
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Directory Management
fism1_spectra_folder = '../empiricalModels/irradiances/FISM1/'
fism2_spectra_folder = '../empiricalModels/irradiances/FISM2/'
TIMED_spectra_folder = '../measurements/TIMED_SEE_Level_3/'
euv_folder = '../tools/EUV/'
#-----------------------------------------------------------------------------------------------------------------------

def obtainFism1(fismFiles, euv_bins, saveLoc=None):
    """
    Given muliple FISM1 .dat files, get the information from each band, using code developed by Dr. Aaron Ridley.
    :param fismfiles: arraylile
        A list or array of .dat files containing FISM1 data (the full VUV spectrum from .1nm to 195nm at 1 nm
        resolution.
    :param euv_bins: dict
        EUV bins with which to rebin the FISM1 data. Obtained from fism2_process.rebin_fism.
    :param saveLoc: str
        Optional argument that controls where pickle files are saved.
    :return irrTimes: ndarray
        A 1d array of datetimes corresponding to each set of irradiance values.
    :return irrArray: ndarray
        An ndarray containing all of the individual 59 irradiance values in each band from all .dat files.
    """
    myIrrPickleFile = 'myIrrFISM1.pkl'
    myTimePickleFile = 'myTimesFISM1.pkl'
    override = True # If true, forcefully read in the CSV data irrespective if it is previously existing.
    irrTimes = [] #np.zeros(len(fismFiles))
    numBins = len(euv_bins['long'])
    if saveLoc != None:
        searchString = saveLoc + myIrrPickleFile
    else:
        searchString = myIrrPickleFile
    if os.path.isfile(searchString) == False or override == True:
        irrArray = []
        for i in tqdm(range(len(fismFiles))): # Loop through the files.
            with open(fism1_spectra_folder+fismFiles[i]) as fismFileInfo:
                fismFileData = fismFileInfo.readlines()
                currentIrrArray = np.zeros((len(fismFileData[1:]), numBins))
                j = -1
                for line in fismFileData: # Loop over the lines in the file.
                    if j >= 0:
                        fismLineData = line.split()
                        # The first six elements of fismLineData are part of the time stamp:
                        irrTimes.append(datetime(int(fismLineData[0]), int(fismLineData[1]), int(fismLineData[2]), int(fismLineData[3])))
                        # The remaining elements of fismLineData are the irradiance in 59 wavelengths (the order must be flipped):
                        currentIrrArray[j, :] = np.flip(np.asarray(fismLineData[6:]))
                    j += 1
                irrArray.append(currentIrrArray)
        finalIrrArray = np.concatenate(irrArray)
        # Sort the data and order it properly:
        sort_indices = np.argsort(irrTimes)
        irrTimes = np.asarray(irrTimes)[sort_indices]
        for k in range(numBins):
            finalIrrArray[:, k] = finalIrrArray[:, k][sort_indices]
        # Write to a pickle (since the data takes a while to gather and read in):

        if saveLoc != None:
            myTimePkl = open(myTimePickleFile, 'wb')
            myIrrPkl = open(myIrrPickleFile, 'wb')
        else:
            myTimePkl = open(saveLoc + myTimePickleFile, 'wb')
            myIrrPkl = open(saveLoc + myIrrPickleFile, 'wb')
        pickle.dump(np.asarray(irrTimes), myTimePkl)
        pickle.dump(irrArray, myIrrPkl)
    else:
        myTimePkl = open(saveLoc+myTimePickleFile, 'rb')
        irrTimes = pickle.load(myTimePkl)
        myIrrPkl = open(saveLoc+myIrrPickleFile, 'rb')
        finalIrrArray = pickle.load(myIrrPkl)
    return irrTimes, finalIrrArray

def obtainFism2(myFism2File, euv_bins, saveLoc=None):
    """
    Given a single FISM2 netcdf4 file, load it in and rebin it into 59 GITM wavelength bins, saving it to pickle files.
    :param myFism2File: str
        The location of the NETCDF4 file.
    :param euv_bins: dict
        EUV bins with which to rebin the FISM1 data. Obtained from fism2_process.rebin_fism.
    :param saveLoc: str
        Optional argument that controls where pickle files are saved.
    :return irrTimes: ndarray
        A 1d array of datetimes corresponding to each set of irradiance values.
    :return new_fism2_irr: ndarray
        An ndarray containing all of the individual 59 irradiance values in each band from all .dat files.
    :return new_fism2_unc: ndarray
        An ndarray containing uncertainties of all the individual 59 irradiance values in each band from all .dat files.
    """
    myIrrPickleFile = 'myIrrFISM2.pkl'
    myIrrUncPickleFile = 'myIrrUncFISM2.pkl'
    myTimePickleFile = 'myTimesFISM2.pkl'
    override = True
    if saveLoc != None:
        searchString = saveLoc + myIrrPickleFile
    else:
        searchString = myIrrPickleFile
    if os.path.isfile(searchString) == False or override == True:
        fism2Data = Dataset(myFism2File)
        irradiance = fism2Data.variables['irradiance']
        wavelength = fism2Data.variables['wavelength']
        uncertainty = fism2Data.variables['uncertainty']
        dates = fism2Data.variables['date']
        datetimes = []
        for i in range(len(dates)):
            year = dates[i][:4]
            day = dates[i][4:]
            currentDatetime = datetime(int(year), 1, 1) + timedelta(int(day) - 1) + timedelta(hours=12)
            datetimes.append(currentDatetime)
        datetimes = np.asarray(datetimes)
        # Performing the rebinning:
        new_fism2_irr, fism2_ave_wave = rebin_fism(np.asarray(wavelength)*10, np.asarray(irradiance), euv_bins) # isolate_fism
        new_fism2_unc, fism2_ave_wave = rebin_fism(np.asarray(wavelength)*10, np.asarray(uncertainty), euv_bins) # isolate_fism
        # Save the data:
        if saveLoc != None:
            myTimePkl = open(myTimePickleFile, 'wb')
            myIrrPkl = open(myIrrPickleFile, 'wb')
            myIrrUncPkl = open(myIrrUncPickleFile, 'wb')
        else:
            myTimePkl = open(saveLoc + myTimePickleFile, 'wb')
            myIrrPkl = open(saveLoc + myIrrPickleFile, 'wb')
            myIrrUncPkl = open(saveLoc + myIrrUncPickleFile, 'wb')
        pickle.dump(np.asarray(datetimes), myTimePkl)
        pickle.dump(new_fism2_irr, myIrrPkl)
        pickle.dump(new_fism2_unc, myIrrUncPkl)
    else:
        myTimePkl = open(saveLoc + myTimePickleFile, 'rb')
        datetimes = pickle.load(myTimePkl)
        myIrrPkl = open(saveLoc + myIrrPickleFile, 'rb')
        new_fism2_irr = pickle.load(myIrrPkl)
        myIrrUncPkl = open(saveLoc + myIrrUncPickleFile, 'rb')
        new_fism2_unc = pickle.load(myIrrUncPkl)
    return datetimes, new_fism2_irr, new_fism2_unc

def obtainSEE(seeFile):
    """
    Given a TIMED/SEE NETCDF4 file, load in and return the timestamps, wavelengths, irradiances, and uncertainties.
    :param seeFile: str
        The NETCDF4 file containing TIMED/SEE data.
    :return datetimes: ndarray
        An array of datetimes for each TIMED/SEE spectra.
    :return wavelengths: ndarray
        A one-dimensional array of wavelengths at which there are irradiance values.
    :return irradiances: ndarray
        A two-dimensional array of irradiance values at each time.
    :return uncertainties: ndarray
        A two-dimensional array of irradiance uncertainty values at each time.
    """
    seeData = Dataset(seeFile)
    dates = np.squeeze(seeData.variables['DATE'])
    wavelengths = np.squeeze(seeData.variables['SP_WAVE'])
    irradiances = np.squeeze(seeData.variables['SP_FLUX'])
    uncertainties = np.squeeze(seeData.variables['SP_ERR_TOT'])
    precision = np.squeeze(seeData.variables['SP_ERR_MEAS'])
    datetimes = []
    for i in range(len(dates)):
        year = str(dates[i])[:4]
        day = str(dates[i])[4:]
        currentDatetime = datetime(int(year), 1, 1) + timedelta(int(day) - 1) + timedelta(hours=12)
        datetimes.append(currentDatetime)
    datetimes = np.asarray(datetimes)
    return datetimes, wavelengths, irradiances, uncertainties

def rebinSEE(SEEirradiances, wavelengths):
    """
    Take irradiances output by by getSEE and bin them according to the strictures of the bin boundaries given by 'bins'.
    :param: SEEirradiances: ndarray
        TIMED/SEE irradiances in the 195 bands collected by TIMED/SEE, in units of W/m^2/nm. The shape of irradiances
        must be n x 195, where n is the number of times (samples) for which irradiances were taken.
    :param: wavelengths: dict
        A dictionary of two elements, one with 'short' wavelengths and one with 'long' wavelengths. Returned by
        fism2_process.read_euv_csv_file. The units here are in angstroms.
    :return: rebinnedIrrData: ndarray
        Rebinned SEE data according to the supplied bin boundaries.
    """
    irradiances = SEEirradiances.copy()
    # The wavelength bands for the SEE data are in units of nanometers, and are spaced by 1 nm:
    SEEBands = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
         9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,
        18.5,  19.5,  20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,
        27.5,  28.5,  29.5,  30.5,  31.5,  32.5,  33.5,  34.5,  35.5,
        36.5,  37.5,  38.5,  39.5,  40.5,  41.5,  42.5,  43.5,  44.5,
        45.5,  46.5,  47.5,  48.5,  49.5,  50.5,  51.5,  52.5,  53.5,
        54.5,  55.5,  56.5,  57.5,  58.5,  59.5,  60.5,  61.5,  62.5,
        63.5,  64.5,  65.5,  66.5,  67.5,  68.5,  69.5,  70.5,  71.5,
        72.5,  73.5,  74.5,  75.5,  76.5,  77.5,  78.5,  79.5,  80.5,
        81.5,  82.5,  83.5,  84.5,  85.5,  86.5,  87.5,  88.5,  89.5,
        90.5,  91.5,  92.5,  93.5,  94.5,  95.5,  96.5,  97.5,  98.5,
        99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5,
       108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5,
       117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5,
       126.5, 127.5, 128.5, 129.5, 130.5, 131.5, 132.5, 133.5, 134.5,
       135.5, 136.5, 137.5, 138.5, 139.5, 140.5, 141.5, 142.5, 143.5,
       144.5, 145.5, 146.5, 147.5, 148.5, 149.5, 150.5, 151.5, 152.5,
       153.5, 154.5, 155.5, 156.5, 157.5, 158.5, 159.5, 160.5, 161.5,
       162.5, 163.5, 164.5, 165.5, 166.5, 167.5, 168.5, 169.5, 170.5,
       171.5, 172.5, 173.5, 174.5, 175.5, 176.5, 177.5, 178.5, 179.5,
       180.5, 181.5, 182.5, 183.5, 184.5, 185.5, 186.5, 187.5, 188.5,
       189.5, 190.5, 191.5, 192.5, 193.5, 194.5]) * 10

    shorts = wavelengths['short']
    longs = wavelengths['long']
    nWaves = len(shorts)
    rebinnedIrrData = np.zeros((irradiances.shape[0], 59))
    ave_wav = np.zeros(nWaves)

    # Nothing needs to be done to convert the units of the SEE data to W/m^2 since the data is spaced by 1 nm.

    # Iterate through each sample of the SEE irradiance data:
    for i in range(irradiances.shape[0]):
        # First, go through the singular wavelengths:
        for iWave, short in enumerate(shorts):
            long = longs[iWave]
            if (long == short):
                d = np.abs(SEEBands - short)
                j = np.argmin(d)
                # If the value of irradiance in this bin is at or below zero, simply set equal to zero.
                if irradiances[i][j] <= 0:
                    rebinnedIrrData[i, iWave] = 0
                else:
                    rebinnedIrrData[i, iWave] = irradiances[i][j] # * ((SEEBands[j + 1] - SEEBands[j]) / 10.0)
                # Zero out bin so we don't double count it.
                irradiances[i][j] = 0.0

        # Then, go through the ranges:
        for iWave, short in enumerate(shorts):
            long = longs[iWave]
            ave_wav[iWave] = (short + long) / 2.0
            if (long != short):
                d1 = np.abs(SEEBands - short)
                iStart = np.argmin(d1)
                d2 = np.abs(SEEBands - long)
                iEnd = np.argmin(d2)
                for j in range(iStart + 1, iEnd + 1):
                    # If the value of irradiance in this bin is at or below zero, simply set equal to zero.
                    if irradiances[i][j] <= 0:
                        rebinnedIrrData[i, iWave] = 0 # += 0
                    else:
                        rebinnedIrrData[i, iWave] += irradiances[i][j] # * ((SEEBands[j + 1] - SEEBands[j]) / 10.0)

    # sampleIdx = 18
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(SEEBands, SEEirradiances[sampleIdx, :], 'b-')
    # ax.plot(ave_wav, rebinnedIrrData[sampleIdx, :], 'ro')

    return rebinnedIrrData

def obtainNRLSSIS2(filename):
    """
    Given a NRLSSI2 NETCDF4 file, load in and return the timestamps, wavelengths, irradiances, and uncertainties.
    :param filename: str
        The NETCDF4 file containing TIMED/SEE data.
    :return datetimes: ndarray
        An array of datetimes for each TIMED/SEE spectra.
    :return wavelengths: ndarray
        A one-dimensional array of wavelengths at which there are irradiance values.
    :return bandwidths: ndarray
        The width of the corresponding band for each irradiance measurement.
    :return irradiance: ndarray
        A two-dimensional array of irradiance values at each time.
    :return uncertainties: ndarray
        A two-dimensional array of irradiance uncertainty values at each time.
    """
    NRLData = Dataset(filename)
    dates = np.squeeze(NRLData.variables['time'])
    irradiances = np.squeeze(NRLData.variables['SSI'])
    wavelengths = np.squeeze(NRLData.variables['wavelength'])
    bandwidths = np.squeeze(NRLData.variables['Wavelength_Band_Width'])
    uncertainties = np.squeeze(NRLData.variables['SSI_UNC'])
    startingEpoch = datetime(1610, 1, 1)
    datetimes = []
    for i in range(len(dates)):
        currentDatetime = startingEpoch + timedelta(days=int(dates[i]), hours=12) # Move the observations to coincide with noon of each day.
        datetimes.append(currentDatetime)
    datetimes = np.asarray(datetimes)
    return datetimes, wavelengths, bandwidths, irradiances, uncertainties

def rebinNRL(NRLirradiances, wavelengths, oldWavelengths, bandwidths):
    """
    Take irradiances output by by obtainNRLSSIS2 and bin them according to the strictures of the bin boundaries given by 'bins'.
    :param: NRLirradiances: ndarray
        TIMED/SEE irradiances in the 195 bands collected by TIMED/SEE, in units of W/m^2/nm. The shape of irradiances
        must be n x 159, where n is the number of times (samples) for which irradiances were taken.
    :param: wavelengths: dict
        A dictionary of two elements, one with 'short' wavelengths and one with 'long' wavelengths. Returned by
        fism2_process.read_euv_csv_file. The units here are in angstroms.
    :return oldWavelengths: ndarray
        The wavelengths of corresponding to the NRL data. Should be in nanometers.
    :return bandwidths: ndarray
        The width of the corresponding band for each irradiance measurement.
    :return: rebinnedIrrData: ndarray
        Rebinned SEE data according to the supplied bin boundaries.
    """
    irradiances = NRLirradiances.copy()
    # The wavelength bands for the NRL data are in units of nanometers, and have variable spacing.

    # TODO: Fix things so that only the relevant bands are considered.
    shorts = wavelengths['short']
    longs = wavelengths['long']
    mids = 0.5*(shorts + longs)
    nWaves = len(shorts)
    validWavelengthInds = np.where(oldWavelengths >= shorts[0])[0]
    validWavelengths = oldWavelengths[validWavelengthInds]
    validIrradiances = NRLirradiances[:, validWavelengthInds]
    # Isolate the GITM wavelengths between the max and min of the valid NRL wavelengths:
    lowGitmBandIndex = np.where(shorts >= validWavelengths[0])[0]
    validShorts = shorts[lowGitmBandIndex[0]:]
    validLongs = longs[lowGitmBandIndex[0]:]
    validMids = 0.5*(validShorts + validLongs)

    # Sum the NRL irradiance data into the valid GITM bins:
    rebinnedIrrData = np.zeros((irradiances.shape[0], len(validMids)))
    ave_wav = np.zeros(nWaves)
    # Iterate through each sample of the NRL irradiance data:
    for i in range(irradiances.shape[0]):
        # First, go through the singular wavelengths:
        for iWave, short in enumerate(validShorts):
            long = validLongs[iWave]
            if (long == short):
                d = np.abs(oldWavelengths - short)
                j = np.argmin(d)
                rebinnedIrrData[i, iWave] = validIrradiances[i][j]  # * ((SEEBands[j + 1] - SEEBands[j]) / 10.0)
                # zero out bin so we don't double count it.
                validIrradiances[i][j] = 0.0

        # Then, go through the ranges:
        for iWave, short in enumerate(validShorts):
            long = validLongs[iWave]
            ave_wav[iWave] = (short + long) / 2.0
            if (long != short):
                d1 = np.abs(oldWavelengths - short)
                iStart = np.argmin(d1)
                d2 = np.abs(oldWavelengths - long)
                iEnd = np.argmin(d2)
                for j in range(iStart + 1, iEnd + 1):
                    rebinnedIrrData[i, iWave] += validIrradiances[i][j]  # * ((SEEBands[j + 1] - SEEBands[j]) / 10.0)

    # sampleIdx = 18
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(SEEBands, SEEirradiances[sampleIdx, :], 'b-')
    # ax.plot(ave_wav, rebinnedIrrData[sampleIdx, :], 'ro')

    return rebinnedIrrData

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    euv_data_59 = read_euv_csv_file(euv_folder + 'euv_59.csv', band=False)
    # Load in the FISM1 data in multiple files, rebin it into 59 GITM wavelength bins, and combine it into a single pickle file:
    myFism1Files = os.listdir(fism1_spectra_folder)
    myIrrTimesFISM1, myIrrDataAllFISM1 = obtainFism1(myFism1Files, euv_data_59, saveLoc=fism1_spectra_folder)
    # Load in the FISM2 data, rebin it into 59 GITM wavelength bins, and save it to a pickle file:
    fism2file = '../empiricalModels/irradiances/FISM2/daily_data_1947-2023.nc'
    myIrrTimesFISM2, myIrrDataAllFISM2, myIrrUncAllFISM2 = obtainFism2(fism2file, euv_data_59, saveLoc=fism2_spectra_folder)
    # Load in TIMED/SEE data, rebin it into 59 GITM wavelength bins, and save it to pickle file:
    seeFile = '../measurements/TIMED_SEE_Level_3/see_L3_merged_1947-2023.ncdf'
    myIrrTimesSEE, mySEEWavelengths, myIrrDataAllSEE, myIrrUncAllSEE = obtainSEE(seeFile)
    rebinnedIrrData = rebinSEE(myIrrDataAllSEE, euv_data_59)
    # Load in NLRSIS2 data, rebin it into 59 GITM wavelength bins, and save it to pickle file:
    NRLFile = '../empiricalModels/irradiances/NRLSSI2/ssi_v02r01_daily_s18820101_e20221231_c20230123.nc'
    datetimesNRL, wavelengthsNRL, bandwidthsNRL, irradiancesNRL, uncertaintiesNRL = obtainNRLSSIS2(NRLFile)
    rebinnedNRLData = rebinNRL(irradiancesNRL, euv_data_59, wavelengthsNRL, bandwidthsNRL)
    sys.exit(0)