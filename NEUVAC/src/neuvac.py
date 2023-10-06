# Empirical Model of Solar Irradiance

# Developed by:
# Aaron J. Ridley, Ph.D.
# Daniel A. Brandt, Ph.D.

#-----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
from scipy.optimize import curve_fit
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Local imports:
from tools.spectralAnalysis import spectralIrradiance
from tools.toolbox import find_nearest
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Waves Table
# Format: 0-Min, 1-Max, 2-S_1i, 3-S_Ai, 4-S_Di, 5-I_i, 6-Pi, 7-Ai
waveTable = np.array([
    [1700.00, 1750.00, 1.31491e-06, 6.71054e-06, 5.78034e-07, 0.00355128, 1.05517, 0.901612],
    [1650.00, 1700.00, 5.19285e-07, 2.62376e-06, 3.08447e-07, 0.00218156, 1.06245, 0.964892],
    [1600.00, 1650.00, 3.85348e-07, 1.73851e-06, 3.34911e-07, 0.00115310, 1.07246, 0.959562],
    [1550.00, 1600.00, 2.96220e-07, 1.29250e-06, 2.61812e-07, 0.000814814, 1.04567, 0.967804],
    [1500.00, 1550.00, 2.35326e-07, 1.21123e-06, 2.27793e-07, 0.000566574, 1.13520, 0.970257],
    [1450.00, 1500.00, 1.86793e-07, 5.96399e-07, 1.48283e-07, 0.000331058, 1.01564, 0.940506],
    [1400.00, 1450.00, 1.96396e-07, 5.84154e-07, 1.82438e-07, 0.000207013, 1.67546, 0.945697],
    [1350.00, 1400.00, 1.04362e-07, 5.02422e-07, 1.45100e-07, 0.000153277, 1.04246, 0.992749],
    [1300.00, 1350.00, 1.74403e-07, 6.32214e-07, 4.03009e-07, 0.000311075, 1.00964, 1.09381],
    [1250.00, 1300.00, 7.12738e-08, 2.44220e-07, 9.56532e-08, 9.68823e-05, 1.15737, 1.01121],
    [1200.00, 1250.00, 8.74335e-06, 5.02272e-05, 1.32536e-05, 0.00263307, 1.46273, 0.987493],
    [1215.67, 1215.67, 6.43713e-06, 5.16823e-05, 1.11399e-05, 0.00247063, 1.26340, 0.998295],
    [1150.00, 1200.00, 1.15468e-07, 2.74916e-07, 1.65125e-07, 0.000105178, 1.66887, 1.00997],
    [1100.00, 1150.00, 7.71861e-08, 2.15061e-07, 1.44227e-07, 5.16157e-05, 0.971988, 1.05634],
    [1050.00, 1100.00, 5.84127e-08, 3.08808e-07, 1.25160e-07, 4.65227e-05, 1.58808, 1.05327],
    [1000.00, 1050.00, 2.23073e-07, 6.92710e-07, 5.19444e-07, 5.44992e-05, 0.449052, 1.10271],
    [1031.91, 1031.91, 6.18723e-08, 1.21679e-07, 2.28527e-07, 3.14905e-05, 1.42684, 1.17863],
    [1025.72, 1025.72, 1.61504e-07, 4.38856e-07, 2.79663e-07, 1.06365e-05, 1.09262, 1.05186],
    [950.00, 1000.00, 1.70358e-07, 5.20531e-07, 3.86006e-07, 3.34989e-05, 0.491283, 1.09676],
    [977.02, 977.02, 1.51857e-07, 5.60743e-07, 2.74541e-07, 6.71100e-06, 1.44918, 1.04869],
    [900.00, 950.00, 7.27646e-08, 4.53511e-07, 1.91513e-07, 3.93851e-05, 1.21476, 1.06473],
    [850.00, 900.00, 1.45264e-07, 2.82927e-07, 4.22856e-07, 4.83494e-05, 1.15579, 1.14948],
    [800.00, 850.00, 6.69560e-08, 1.26613e-07, 1.76066e-07, 3.69687e-05, 1.14722, 1.12832],
    [750.00, 800.00, 3.22816e-08, 7.81757e-08, 6.32959e-08, 4.42679e-05, 0.969748, 1.06692],
    [789.36, 789.36, 1.19733e-08, 2.53334e-08, 1.58546e-08, 1.25539e-05, 1.48302, 1.00982],
    [770.41, 770.41, 7.33597e-09, 2.10650e-08, 1.63125e-08, 8.88041e-06, 1.18634, 1.06584],
    [765.15, 765.15, 4.85967e-09, 1.05567e-08, 5.42104e-09, 1.15262e-05, 1.17912, 1.03352],
    [700.00, 750.00, 1.85139e-08, 3.63837e-08, 3.29576e-08, 1.72134e-05, 1.25328, 1.06364],
    [703.36, 703.36, 5.34708e-09, 9.65120e-09, 4.54419e-09, 8.80278e-06, 1.51207, 0.972520],
    [650.00, 700.00, 1.79851e-08, 6.39605e-08, 1.86000e-08, 1.41950e-05, 1.11181, 0.945801],
    [600.00, 650.00, 1.52595e-07, 5.29641e-07, 1.41837e-07, 3.96165e-05, 1.00554, 0.949913],
    [629.73, 629.73, 4.96048e-08, 2.46454e-07, 3.12902e-08, 1.59200e-05, 1.01611, 0.846628],
    [609.76, 609.76, 2.80641e-08, 3.24530e-07, 1.81554e-08, 1.68460e-06, 0.973085, 0.793355],
    [550.00, 600.00, 1.12234e-07, 6.29889e-07, 1.56092e-07, 2.79143e-05, 0.961457, 0.970150],
    [584.33, 584.33, 7.91646e-08, 3.05430e-07, 5.14430e-08, 1.70372e-05, 0.844250, 0.881026],
    [554.31, 554.31, 2.47485e-08, 2.68042e-07, 5.40951e-08, 1.16226e-06, 1.08699, 1.01483],
    [500.00, 550.00, 1.12037e-07, 7.84515e-07, 6.32364e-08, 4.55230e-06, 1.13480, 0.816868],
    [450.00, 500.00, 1.10016e-07, 3.96192e-07, 7.37101e-08, 2.62692e-05, 1.15344, 0.865234],
    [465.22, 465.22, 9.60010e-09, 1.75358e-08, 6.91440e-11, 1.45142e-05, 1.62256, -0.203971],
    [400.00, 450.00, 5.15555e-08, 2.89821e-07, 3.85807e-08, 1.64207e-05, 1.36652, 0.893190],
    [350.00, 400.00, 3.91955e-07, 1.43942e-06, 3.16713e-07, -2.36108e-06, 1.05819, 0.910235],
    [368.07, 368.07, 1.38855e-07, 7.21254e-07, 1.01814e-07, 8.71098e-07, 1.26707, 0.890513],
    [300.00, 350.00, 1.35439e-06, 1.09238e-05, 8.24308e-07, 4.35250e-05, 1.22619, 0.816515],
    [303.78, 303.78, 7.43959e-07, 5.94012e-06, 4.05188e-07, 9.23799e-05, 1.32976, 0.796970],
    [303.31, 303.31, 5.25977e-07, 7.87164e-06, 3.07932e-07, 7.87468e-05, 0.945961, 0.759694],
    [250.00, 300.00, 9.10710e-07, 3.91586e-06, 1.20177e-06, -9.64301e-06, 1.07360, 0.958369],
    [284.15, 284.15, 8.67633e-07, 6.00671e-06, 3.97664e-07, -0.000107230, 1.20608, 0.773950],
    [256.30, 256.30, 6.44996e-08, 4.12637e-07, 1.05193e-07, 6.61853e-06, 1.48670, 1.03265],
    [200.00, 250.00, 4.83013e-07, 1.18898e-06, 8.94772e-07, 5.34779e-05, 1.04532, 1.07888],
    [150.00, 200.00, 7.13305e-07, 2.47623e-06, 9.78936e-07, 0.000261230, 1.47374, 1.01156],
    [100.00, 150.00, 4.03676e-08, 2.28270e-07, 4.43965e-08, 2.16162e-05, 1.09062, 0.970310],
    [50.00, 100.00, 1.69769e-07, 6.93618e-07, 2.89457e-07, 2.03013e-05, 1.07887, 1.06022],
    [32.00, 50.00, 1.23478e-07, 4.43644e-07, 1.75749e-07, -1.34567e-05, 1.27409, 1.01254],
    [23.00, 32.00, 6.10174e-08, 2.34313e-07, 1.10591e-07, -1.22729e-05, 0.699812, 1.04841],
    [16.00, 23.00, 2.23866e-07, 7.97533e-07, 3.03563e-07, -5.62012e-05, 0.706360, 0.987835],
    [8.00, 16.00, 3.10773e-07, 1.22767e-06, 3.74797e-07, -8.41459e-05, 1.39529, 0.963859],
    [4.00, 8.00, 1.17378e-08, 7.13970e-08, 1.38839e-08, -3.63146e-06, 0.811119, 0.920702],
    [2.00, 4.00, 3.97985e-09, 4.12085e-08, 4.71914e-09, -1.86099e-06, 1.15214, 0.916686],
    [1.00, 2.00, 3.52498e-09, 1.57342e-08, 4.03741e-09, -8.84488e-07, 0.951714, 0.943490]
    ])
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Functions:
def neuvacEUV(f107, f107a, bandLim=False):
    """
    Use a parametric model to compute solar flux in the 59 conventional wavelength bands used by Aether/GITM.
    This is the ORIGINAL NEUVAC model constructed by Aaron Ridley.
    :param f107: ndarray
        F10.7 values.
    :param f107a: ndarray
        81-day center-averaged F10.7 values; must be the same length as f107.
    :param bandLim: bool
        If True, limits the outputted bands to just those 37 used by EUVAC.
    :param calibrate: bool
        If True, applies empirically-determined correction factors to specific wavelength bands.
    :return euvFlux: ndarray
        A nxm ndarray where n is the number of EUV flux values and m is the number of wavelength bands.
    :return euvIrradiance: ndarray
        A nxm ndarray where n is the number of EUV irradiance values and m is the number of wavelength bands.
    """
    if type(f107) != np.ndarray:
        f107 = np.asarray([f107])
        f107a = np.asarray([f107a])
        solarFlux = np.zeros((1, waveTable.shape[0]))
    else:
        solarFlux = np.zeros((len(f107), waveTable.shape[0]))
    euvFlux = np.zeros_like(solarFlux)
    euvIrradiance = np.zeros_like(euvFlux)
    # Gather the relevant data:
    RidleySlopes = waveTable[:, 2:5]
    RidleyPowers = waveTable[:, 6:]
    RidleyIntercepts = waveTable[:, 5]
    WAVES = waveTable[:, 0]
    WAVEL = waveTable[:, 1]
    # Loop across the F10.7 (and F10.7A) values:
    for i in range(len(f107)):
        # Loop across the wavelengths (59 conventional wavelengths):
        k = 0
        for j in reversed(range(solarFlux.shape[1])):
            solarFlux[i, k] = RidleySlopes[j, 0] * (f107[i]**RidleyPowers[j, 0]) + \
                         RidleySlopes[j, 1] * (f107a[i]**RidleyPowers[j, 1]) + \
                         RidleySlopes[j, 2] * (f107a[i] - f107[i]) + RidleyIntercepts[j]
            wvavg = (WAVEL[j] + WAVES[j])/2.
            euvFlux[i, k] = solarFlux[i, k] * wvavg * 1e-10 / (6.626e-34 * 2.998e8) # / 10.
            # Calculation of irradiance:
            dWave = WAVEL[j] - WAVES[j]
            if WAVEL[j] != WAVES[j]:
                euvIrradiance[i, k] = 0.1 * dWave * spectralIrradiance(euvFlux[i, k], wvavg, dWave)
            else:
                # if WAVEL[j] < 1000:
                #     factor = 500
                # else:
                #     factor = 1000
                euvIrradiance[i, k] = wvavg * spectralIrradiance(euvFlux[i, k], wvavg)  # waveTable[j, 0]
            k += 1
    if bandLim: # Returns values ONLY for those corresponding to the wavelengths used by EUVAC
        return euvFlux[:, 7:44] #15:52]
    return euvFlux, np.squeeze(euvIrradiance)

def correlatedNEUVAC(meanSpectra, corrModels, f107):
    """
    From the mean NEUVAC EUV Spectra, and values of F107 and F107A, compute the resulting perturbed spectrum with
    associated error bars related to the statistical properties of covariance/correlation between the wavelength bands.
    :param meanSpectra: ndarray
        The mean NEUVAC spectrum.
    :param corrModels: list
        A list of poly1d objects that capture correlated noise in NEUVAC with respect to F10.7.
    :param f107: ndarray, int, or float
        Values of F10.7.
    :param f107a: ndarray, int, or float
        Values of 81-day average F10.7.
    :param euvFluxCorr: list
        A list containing the results in two elements: Both are nxm ndarray where n is the number of EUV irradiance
        values and m is the number of wavelength bands, with correlated noise added in. The first element is the upper
        result and the second element is the lower result.
    """
    if type(f107) != np.ndarray:
        f107 = np.asarray([f107])
        # f107a = np.asarray([f107a])
    # Loop through the samples and compute the results:
    upperResults = []
    lowerResults = []
    for i in range(len(f107)):
        # Add in the correlated noise:
        meanSpectraWithNoiseUpper = []
        meanSpectraWithNoiseLower = []
        for j in range(len(meanSpectra)):
            meanSpectraWithNoiseUpper.append(corrModels[j](f107[i])) # meanSpectra[j] + corrModels[j](f107[i])
            meanSpectraWithNoiseLower.append(corrModels[j](f107[i])) # meanSpectra[j] - corrModels[j](f107[i])
        upperResults.append(meanSpectraWithNoiseUpper)
        lowerResults.append(meanSpectraWithNoiseLower)
    euvFluxCorr = np.squeeze(np.asarray([lowerResults, upperResults]))
    return euvFluxCorr

def neuvacFit(f107Data, irrTimes, irrData):
    """
    Calculate entirely new empirical parametric fits between F10.7 data and solar EUV irradiance data, irrespective
    of the number of wavelength bands the irradiance data is split into.
    :param f107Data: list
        A list where the first element is an arraylike of datetimes for each F10.7 value, the second element is an
        arraylike of F10.7 values, and the third element is an arraylike of centered running 81-day averaged F10.7
        values.
    :param irrTimes: arraylike
        An arraylike of datetimes for each solar EUV spectra in irrData.
    :param irrData: ndarray
        An array of solar EUV irradiance measurements or estimates (from FISM, TIMED/SEE, etc.), arranged such that
    :param neuvacTable: ndarray
        An array of coefficients with which to compute the irradiance in each bin.
    """
    import matplotlib.pyplot as plt

    # Functional form for the empirical model:
    # Irr_i(t) = A_i * (F107(t)**B_i) + C_i * (F107A(t)**D_i) + E_i * (F107A(t) - F107(t))**F_i
    def irrFunc(F107input, A, B, C, D, E, F):
        F107, F107A = F107input
        return A*(F107**B) + C*(F107A**D) + E*(F107A-F107) + F

    # Isolate the valid times for performing the fit:
    f107times = f107Data[0]
    f107 = f107Data[1]
    f107A = f107Data[2]
    validInds = np.where((f107times >= irrTimes[0]) & (f107times <= irrTimes[-1]))[0]
    f107TimesSubset = f107times[validInds]
    f107Subset = f107[validInds]
    f107ASubset = f107A[validInds]

    # Ensure that the time resolution is harmonized between the subset F10.7 data and the irradiance data, so that only
    # the elements co-located in time will be considered for fitting:
    f107TimesSubsetNearest = []
    f107SubsetNearest = []
    f107ASubsetNearest = []
    for i in range(len(irrTimes)):
        coLocatedInfo = find_nearest(f107TimesSubset, irrTimes[i])
        f107TimesSubsetNearest.append(f107TimesSubset[coLocatedInfo[0]])
        f107SubsetNearest.append(f107Subset[coLocatedInfo[0]])
        f107ASubsetNearest.append(f107ASubset[coLocatedInfo[0]])
    f107Predictors = np.array([np.asarray(f107SubsetNearest), np.asarray(f107ASubsetNearest)])

    # Loop through each individual band and perform the fit, returning the obtained coefficients:
    fitParams = []
    for j in range(irrData.shape[1]):
        if j >= 3:
            nonNanLocs = ~np.isnan(irrData[:, j])
            fitResult = curve_fit(irrFunc, f107Predictors[:, nonNanLocs], irrData[:, j][nonNanLocs])
            fitPopt, fitPcov = fitResult
            fig, axs = plt.subplots(1, 3, figsize=(24, 10))
            # Irradiance vs. F10.7:
            axs[0].scatter(f107Predictors[0], irrData[:, j])
            axs[0].set_xlabel('F10.7 (sfu)')
            axs[0].set_ylabel('Irradiance (W/m$^2$/nm)')
            axs[0].set_title('Band '+str(j+1)+': Irradiance vs. F10.7')
            # Irradiance vs. F10.7A:
            axs[1].scatter(f107Predictors[1], irrData[:, j])
            axs[1].set_xlabel('F10.7A (sfu)')
            axs[1].set_ylabel('Irradiance (W/m$^2$/nm)')
            axs[1].set_title('Band ' + str(j+1) + ': Irradiance vs. F10.7A')
            # Fit results:
            pred = irrFunc(f107Predictors, *fitPopt)
            axs[2].plot(f107TimesSubset, irrData[:, j][1:], label='TIMED/SEE')
            axs[2].plot(f107TimesSubset, pred[1:], label='NEUVAC')
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Irradiance (W/m$^2$/nm)')
            axs[2].set_title('Model Results')
            axs[2].legend(loc='best')
            # Ylims (always set to mean +/- 3 sigma):
            meanIrr = np.nanmean(irrData[:, j])
            if j+1 == 13 or j+1 == 14:
                axs[0].set_ylim([0, 5*meanIrr])
                axs[1].set_ylim([0, 5*meanIrr])
                axs[2].set_ylim([-meanIrr, 5*meanIrr])
            elif j+1 == 17:
                axs[0].set_ylim([0, 2.2 * meanIrr])
                axs[1].set_ylim([0, 2.2 * meanIrr])
                axs[2].set_ylim([-meanIrr, 2.2 * meanIrr])
            # Appending:
            fitParams.append(fitResult)
            # Saving the figure:
            plt.savefig('Fitting/Band_'+str(j+1)+'_NEUVAC_fit.png', dpi=300)

    return fitParams
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution (for debugging):
if __name__ == '__main__':
    # Generate some sample data (TODO: Replace with real data):
    from random import randint
    f107 = np.array([randint(0, 200) for p in range(0, 10)])
    f107a = np.array([randint(0, 200) for p in range(0, 10)])
    neuvacFlux = neuvacEUV(f107, f107a)