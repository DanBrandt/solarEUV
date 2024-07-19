# solarEUV
Contains the NEUVAC empirical model of solar EUV irradiance for use primarily in thermosphere-ionosphere models, 
developed primarily by Dr. Aaron J. Ridley, with analysis and contributions from Dr. Daniel A. Brandt.

This repository also contains code for comparing and analyzing the performance of NEUVAC in comparison to other 
empirical models such as EUVAC, HEUVAC, and FISM2.

# Installation
solarEUV should run right out of the box. It can be obtained with a simple git clone command, and the relevant modules 
may then be loaded as usual. If running it proves difficult, then you may wish to install the modules in the 
'requirements.txt' file. In Linux, this may be done with the command:
> pip install -r requirements.txt

# Usage
solarEUV contains modules for **4** different EUV irradiance models. These models include:
* NEUVAC
* EUVAC
* HEUVAC
* SOLOMON

**A note about running HEUVAC in particular:** The base code of HEUVAC is in Fortran, written by Dr. Phil Richards. In
order to run it on a Unix-based OS, the script compile.sh should be run first. This requires gfortran, so if that is 
not installed on your system, please install it first. When this is completed, navigate (via terminal) to the directory
empiricalModels/models/HEUVAC and do the following:

> . compile.sh

This should only have to be done once. After this is completed, you can navigate back to the top directory and proceed 
as normal, running HEUVAC simply by calling the wrapper function in the module heuvac.py. 

We note that SOLOMON in the literature can either refer to the empirical model between F10.7 and 81 day-averaged F10.7 
centered on the current day (hereafter F10.7A) and EUV irradiance in 22 overlapping bands as described by Solomon and
Qian 2005, or it can refer to _any_ EUV irradiance data summed into those 22 overlapping bins (referred to as the STAN 
BANDS). In this package, SOLOMON only refers to the former, though functionality does exist to run all other models in
the STAN BANDS.

## Finding your way around

There are few folders in this package:
* **empiricalModels**: Contains code and data for EUVAC, HEUVAC, and SOLOMON, as well as FISM:
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

To import any of the models (while in the top directory 'solarEUV') simply do as follows:

<ins>NEUVAC</ins>
> from NEUVAC.src import neuvac
> 
> neuvacIrr, perturbedNeuvacIrr, _, _ = neuvac.neuvacEUV(F107, F107A, tableFile=neuvac_tableFile, statsFiles=['corMat.pkl', 'sigma_NEUVAC.pkl']

<ins>EUVAC</ins>
> from empiricalModels.models.EUVAC import euvac
> 
> euvacFlux, euvacIrr, _, _, _ = euvac.euvac(F107, F107A, statsFiles=['corMatEUVAC.pkl', 'sigma_EUVAC.pkl'])

<ins>HEUVAC</ins>
> from empiricalModels.models.HEUVAC import heuvac
> 
> heuvac_wav, heuvacFlux, heuvacIrr, _, _, _ = heuvac.heuvac(F107, F107A, torr=True, statsFiles=['corMatHEUVAC.pkl', 'sigma_HEUVAC.pkl'])

<ins>SOLOMON</ins>
> from empiricalModels.models.SOLOMON import solomon
> 
> solomonFluxHFG, solomonIrrHFG = solomon.solomon(F107, F107A, model='HFG')
>
> solomonFluxEUVAC, solomonIrrEUVAC = solomon.solomon(F107, F107A, model='EUVAC')

Please note the following for the above:
* neuvac_tableFile: Holds all the coefficients of the most recent fit of NEUVAC. The file is located here: /NEUVAC/src/neuvac_table.txt
* corMat.pkl: Holds the normalized correlation matrix for all the NEUVAC bands. Located in the folder 'experiments'. This is similarly true for 'corMatEUVAC.pkl' and 'corMatHEUVAC.pkl'.
* sigma_NEUVAC.pkl: Holds the normalized standard deviation of the residuals with respect to FISM2. Located in the folder 'experiments'. This is similarly true for 'sigma_EUVAC.pkl' and 'sigma_HEUVAC.pkl'.

Running any of the EUV models is straightforward, as shown in the examples below.

# Example 1: EUV Irradiance Time Series

[Forthcoming]

# Example 2: Running Irradiance Ensembles

[Forthcoming]
Due to the unique construction of NEUVAC, at present, we only recommend running ensembles for NEUVAC, and not any of the
other models.

# Scientific Ethics
In using this code, depending on the module used, proper citations should be given to the original authors responsible
for developing each model or dataset:

<ins>NEUVAC:</ins> [Brandt and Vega, in review](https://essopenarchive.org/users/539460/articles/1139162-neuvac-nonlinear-extreme-ultraviolet-irradiance-model-for-aeronomic-calculations)

<ins>EUVAC:</ins> [Richards, et al. 1994](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/94ja00518)

<ins>HEUVAC</ins> [Richards, et al. 2006](https://www.sciencedirect.com/science/article/pii/S0273117705008288?casa_token=zEhwbyXrC8MAAAAA:qHFmKe0ZDE4gMsAX9qAHESvPyoEDFLBlhHuLaEsIwYFykhFXN79--XttCW-QDg1sA4wgD54ysFc)

<ins>SOLOMON</ins> [Solomon and Qian, 2005](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2005JA011160)

<ins>FISM2:</ins> [Chamberlin, et al. 2020](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2020SW002588)
