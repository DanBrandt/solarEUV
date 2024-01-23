# solarEUV
Contains the NEUVAC empirical model of solar EUV irradiance for use primarily in thermosphere-ionosphere models, 
developed primarily by Dr. Aaron J. Ridley, with analysis and minor contributions from Dr. Daniel A. Brandt.

This repository also contains code for comparing and analyzing the performance of NEUVAC in comparison to other 
empirical models such as EUVAC, FISM, and HEUVAC.

# Installation
solarEUV should run right out of the box. It can be obtained with a simple git clone command, and the relevant modules 
may then be loaded as usual. If running it proves difficult, then you may wish to install the modules in the 
'requirements.txt' file. In Linux, this may be done with the command:
> pip install -r requirements.txt

# Usage
solarEUV contains modules for **4** different EUV irradiance models. These models include:
* EUVAC
* HEUVAC
* SOLOMON
* NEUVAC

We note that SOLOMON in the literature can either refer to the empirical model between F10.7 and 81 day-averaged F10.7 
centered on the current day (hereafter F10.7A) and EUV irradiance in 22 overlapping bands as described by Solomon and
Qian 2005, or it can refer to _any_ EUV irradiance data summed into those 22 overlapping bins (referred to as the STAN 
BANDS). In this package, SOLOMON only refers to the former, though functionality does exist to run all other models in
the STAN BANDS.

## Finding your way around

There are few folders in this package:
* **empiricalModels**: Contains code and data for EUVAC, HEUVAC, and SOLOMON, as well as FISM.
* **experiments**: Contains code and figures related to the publication associated with NEUVAC. In this folder, the file
_fitNeuvac.py_ s used for actually performing the NEUVAC fits between F10.7, F10.7A, and FISM2, while _uncNeuvac.py_ 
contains code for computing the correlation matrix used to enable running NEUVAC ensembles, as well as generating plots 
of the squared difference between NEUVAC and FISM2 in different bands.
* **measurements**: Contains data from SDO/EVE and TIMED/SEE. Much of the data here isn't used at all.
* **NEUVAC**: Contains the code for running NEUVAC.
* **solarIndices**: Contains F10.7 solar index data, from both OMNIWeb and Penticton.
* **tools**: Contains code for miscellaneous helper functions. In this folder appears the following:
    * _EUV_: Contains numerous functions within fism2_process.py for reading in and rebinning FISM2 data.
    * _processIndices.py_: Contains functions for reading in, downloading, and cleaning OMNIWeb data.
    * _processIrradiances.py_: Contains functions for reading in data from TIMED/SEE, SDO/EVE, and FISM.
    * _spectralAnalysis.py_: Contains functions for converting between solar spectral irradiance and solar spectral flux.
    * _toolbox.py_: Contains miscellaneous helper functions that mainly focus on directory management, loading and saving data, statistics, and fitting.

Running any of the EUV models is straightforward, as shown in the example below.

# Example

[Forthcoming]