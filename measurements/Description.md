# LCLS-SC (EIC) Injector Measurements

### Measurements:

Measurements at the LCLS EIC injector were taken between October 20th - November 5th 2019.

The HDF5 files in this folder will contain: 

1. Image data of the respective monitor. Virtual cathod camera (VCC) images capture the laser shape, with pixel values corresponding to intensity. Electron beam distributions are captured from the YAG (Yttrium Aluminum Garnet) scintillator.
2. Each sample includes the associated Process Variable (PV) machine information. Variable information, units, and associated Astra information is available here: 

[PV Mapping](https://github.com/slaclab/lcls-sc-inj-surrogate/tree/master/pv_mapping)

### Measured Data Compilation: 

In order to pull the image and PV data, the following information is needed: 

1. Data location at SLAC (archive folder location).
2. Time interval for desired data.

Using [LCLS-Tools](https://github.com/slaclab/lcls-tools), using the database building scripts, the desired data is compiled into the HDF5 files for processing.

### Measured Data Processing

Notebooks for processing the measured data are made available. Data from each month was processes separately and combined to create datasets for surrogate model training/updating. 

The processed data is available as well. 

