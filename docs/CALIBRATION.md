# Calibration Module

The PyCCAPT calibration module provides workflows for calibrating, processing, and analyzing
atom probe tomography (APT) data.


## Jupyter Data Processing Workflows

The calibration workflows are designed to streamline the following APT tasks:

### 1. Data Cropping

- *Description*: Crop atom probe datasets collected with PyCCAPT or imported from formats such
  as EPOS, POS, ATO, and CSV.
- *Usage*: Define the region of interest (ROI) to focus on specific areas of the dataset.

### 2. Time of Flight Calibration

- *Description*: Perform time-of-flight (TOF) calibration to correct flight-time distortions.
- *Usage*: Improve the accuracy of spatial information in the APT dataset.

### 3. Convert Time-of-Flight to Mass-to-Charge Ratio

- *Description*: Calibrate the mass-to-charge ratio (m/c) of ions in the dataset.
- *Usage*: Enhance the accuracy of quantitative analysis by ensuring precise MC values.

### 4. 3D Reconstruction

- *Description*: Reconstruct the 3D spatial distribution from atom probe data.
- *Usage*: Visualize the spatial distribution of atoms within the material.

### 5. Range the Mass-to-Charge Ratio

- *Description*: Define a range for the mass-to-charge ratio to filter ions based on specific MC values.
- *Usage*: Focus on ions within a specific MC range for analysis.

### 6. Visualization

- *Description*: Visualize the atom probe data using various plotting and visualization techniques.
- *Usage*: Gain insights into the data through 2D and 3D visualizations.

### 7. T0 and Flight Path Calculation

- *Description*: Calculate T0 and ion flight-path length.
- *Usage*: Essential for precise quantitative analysis and data interpretation.

## Data Structures

For calibration data structures, see [Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md).
PyCCAPT also supports converting HDF5 data to EPOS, POS, and CSV outputs.
Example workflows are available in
[`tutorials`](https://github.com/mmonajem/pyccapt/tree/main/pyccapt/calibration/tutorials/jupyter_files).

## Additional Features

In addition to the core workflows listed above, the calibration module includes advanced features:

- **Data Analysis**: Perform advanced analysis such as spatial distribution map (SDM), isosurface
  generation, and radial distribution function (RDF) calculation.
- **Data Export**: Export atom probe data to various file formats, including EPOS, POS, ATO, and CSV.
- **Data Import**: Import atom probe data from various file formats, including EPOS, POS, ATO, and CSV.
- **Data Filtering**: Filter atom probe data based on specific criteria, such as mass-to-charge ratio (MC) or spatial
  coordinates.


For usage examples and code snippets, explore the Jupyter notebooks in
the [`tutorials`](https://github.com/mmonajem/pyccapt/tree/main/pyccapt/calibration/tutorials/jupyter_files)
or [`colab`](https://github.com/mmonajem/pyccapt/tree/main/pyccapt/calibration/tutorials/colab)
directories of the PyCCAPT repository.

