# This script performs the model fits for NEUVAC.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from datetime import datetime
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local Imports:
from NEUVAC.src import neuvac
from empiricalModels.models.EUVAC import euvac
from tools.EUV.fism2_process import read_euv_csv_file
from tools.processIrradiances import obtainFism2
from tools import toolbox
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Directory Management
neuvac_directory = '../NEUVAC/src/'
neuvac_tableFile = '../NEUVAC/src/neuvac_table.txt'
figures_directory = 'Figures/'
results_directory = 'Results/'
fism1_spectra_folder = '../empiricalModels/irradiances/FISM1/'
fism2_spectra_folder = '../empiricalModels/irradiances/FISM2/'
euv_folder = '../tools/EUV/'
preparedDataFolder = '../experiments/preparedData'
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Constants
euvacTable = euvac.euvacTable
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__=="__main__":
    # Load in F10.7 data (Penticton, CA):
    pentictonTimesData = '../solarIndices/F107/Penticton/F107times.pkl'
    pentictonF107Data = '../solarIndices/F107/Penticton/F107vals.pkl'
    pentictonF107AveData = '../solarIndices/F107/Penticton/F107averageVals.pkl'
    pentictonTimes = toolbox.loadPickle(pentictonTimesData)
    pentictonF107 = toolbox.loadPickle(pentictonF107Data)
    pentictonF107A = toolbox.loadPickle(pentictonF107AveData)
    # F10.7 data extends between 1947-02-14; 12:00 to 2008-02-03; 12:00.
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

    # NOTE: EUVAC, and HEUVAC return spectral fluxes. They'll need to be converted to spectral irradiance.

    euv_data_59 = read_euv_csv_file(euv_folder + 'euv_59.csv', band=False)
    mids = 0.5 * (euv_data_59['long'] + euv_data_59['short'])

    # FISM2 Results:
    fism2file = '../empiricalModels/irradiances/FISM2/daily_data_1947-2023.nc'
    myIrrTimesFISM2, wavelengthsFISM2, myIrrDataAllFISM2, myIrrUncAllFISM2 = obtainFism2(fism2file, euv_data_59)
    # Rebin the data:
    myIrrDataWavelengthsFISM2, rebinnedIrrDataFISM2 = toolbox.rebin(wavelengthsFISM2, myIrrDataAllFISM2, euv_data_59, zero=False)
    # Replace bad values with NaNs:
    myIrrDataAllFISM2Fixed = rebinnedIrrDataFISM2.copy()
    myIrrDataAllFISM2Fixed[myIrrDataAllFISM2Fixed <= 0 ] = np.nan
    # FISM2 data extends between 1947-02-14; 00:00 and 2023-08-29; 00:00.

    # Perform a non-linear fit between F10.7, F10.7A, and FISM2:
    neuvacTable = neuvac.neuvacFit([times, F107, F107A], myIrrTimesFISM2, myIrrDataAllFISM2Fixed, wavelengths=mids, label='FISM2')

    # Collect the coefficients into a table (so they can be assembled for use in a function):
    # neuvacTableSEEArr = np.asarray(neuvacTableSEE)
    # neuvacTableFISM2Arr = np.asarray(neuvacTableFISM2)
    # neuvacTable = np.concatenate((neuvacTableFISM2Arr, neuvacTableSEE), axis=0)

    # Print the coefficients to a file:
    with open(neuvac_directory+'neuvac_table.txt', 'w') as output:
        # Write the header information:
        output.write('This file contains coefficients for the current iteration of NEUVAC.\n'
                     'This file was created on '+datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')+'\n'
                     'File Authors: Brandt, Daniel A. and Ridley, Aaron J.\n'
                     'This version of NEUVAC was created by fitting nonlinear models between F10.7 and centered\n'
                     '81-day averaged F10.7 and FISM2 decomposed into 59 wavelength bands conventionally used in\n'
                     'the GITM and Aether models.\n'
                     'The file is formatted as follows:\n'
                     ' - First Column: Lower limit of the given wavelength bin in Angstroms.\n'
                     ' - Second Column: Upper limit of the given wavelength bin in Angstroms.\n'
                     ' - Third through Eighth colummns: Coefficients for the model.\n'
                     'The functional form of the model is given by:\n'
                     'Irr_i(t) = A_i * (F107(t) ** B_i) + C_i * (F107A(t) ** D_i) + E_i * (F107A(t) - F107(t)) + F_i\n'
                     'where the irradiance in bin i (Irr_i) is a function of time t, and A_i through F_i are \n'
                     'coefficients for bin i, and F107(t) and F107A(t) represent values of the F10.7 and 81-day\n'
                     'averaged F10.7 centered on the current day, respectively.\n'
                     '-----------------------------------------------------------------------------------------------\n'
                     'WAVES WAVEL A_i B_i C_i D_i E_i F_i\n')
        for i in range(neuvacTable.shape[0]):
            output.writelines(str(euv_data_59['short'][i])+' '+str(euv_data_59['long'][i])+' '+toolbox.stringList(neuvacTable[i, :])+'\n')

    neuvacFlux, neuvacIrr = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile)
    # View the result of the model fits, as a sanity check:
    # for i in range(neuvacIrr.shape[1]):
    #     plt.figure()
    #     plt.plot(myIrrTimesFISM2, myIrrDataAllFISM2Fixed[:, i], label='FISM2')
    #     plt.plot(times, neuvacIrr[:, i], label='NEUVAC')
    #     plt.legend(loc='best')
    #     plt.title('Irradiance Time Series at :'+str(np.round(myIrrDataWavelengthsFISM2[i],2))+' Angstroms')

    # TODO: DO the same as the above, but for the FISM2 STAN BANDS

    print('NEUVAC model fitting complete.')
#-----------------------------------------------------------------------------------------------------------------------
