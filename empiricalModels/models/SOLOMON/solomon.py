# Code for computing solar irradiance according to Solomon and Qian 2005.
# Reference: Solomon, S. C. and Qian, L. (2005) Solar extreme-ultraviolet irradiance for general circulation models,
# Journal of Geophysical Research: Space Physics, 110, A10, 10.1029/2005JA011160

#-----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Global Variable:
solomonTable = np.array([
        [1, 0.5, 4, 5.010e1, 0, 2.948e2, 5.010e1, 6.240e-1, 3.188e4, 7.847e5],
        [2, 4, 8, 1.0e4, 0, 7.6e3, 1.0e4, 3.710e-1, 3.643e4, 8.968e5],
        [3, ],
        [4, ],
        [5, ],
        [6, ],
        [7, ],
        [8, ],
        [9, ],
        [10,],
        [11, ],
        [12,],
        [13, ],
        [14,],
        [15, ],
        [16,],
        [17, ],
        [18,],
        [19,],
        [20, ],
        [21, ],
        [22, ],
        [23, ],
    ])
# The table above is in units of Angstroms - source is Table A1 from Solomon and Qian 2005.
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
def solomon(F107, F107A):
    """
    Compute the solar EUV irradiance in 23 standard bands.
    :param F107: ndarray
        Values of the F10.7 solar flux.
    :param F107A: ndarray
        Values of the 81-day averaged solar flux, centered on the present day.
    :return solomonFlux: ndarray
        Values of the solar radiant flux in 23 distinct wavelength bands.
    :return solomonIrr: ndarray
        Values of the solar EUV irradiance in 23 distinct wavelength bands.
    """
    r1 = 0.0138*(F107 - 71.5) + 0.005*(F107 - F107A + 3.9)
    r2 = 0.5943*(F107 - 71.5) + 0.381*(F107 - F107A + 3.9)

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Execution:
if __name__ == '__main__':
    F107 = np.array([20, 25, 40, 70, 85, 84, 72, 58, 59, 49, 37, 21])
    # F107A = averageF107(F107)
    myFlux, myIrr = solomon(F107, F107)