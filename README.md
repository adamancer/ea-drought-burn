# ea-drought-related-burn-severity

This repository contains code for evaluating the effect of vegetation
mortality on wildfire burn severity in the Woolsey Fire. It was developed
as part of the 
[CU Boulder Earth Data Analytiscs Certificate Program](https://earthlab.colorado.edu/earth-data-analytics-professional-graduate-certificate).


## Background

The Woolsey Fire burned nearly 100,000 acres near Malibu, CA in November 2018.
A four-year drought preceded the fire, resulting in widespread dieback of
grass, shrubs, and trees in and around the area that burned. This project
seeks to understand how the dieback affected the severity of the fire. Did
areas where more vegetation die burn more severely during the fire? Can
pre-existing conditions be used to inform planning for future wildfires?

We will use the random-forest machine-learning algorithm to evaluate how
dieback affected burn severity and will select explanatory and response
variables based on linear regressions of vegetation, climate, and topographic
data for the area in and around the Woolsey Fire scar.


## Installation

The following software is required to install this package using the
instructions below:

+ [Git](https://git-scm.com/downloads)
+ [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Once you have Git and Miniconda installed, open the command line and run the
following commands to set up the environment needed to run this package:

```
conda env create --file environment.yml
conda activate ea-drought-related-burn-severity
```

Then install the package itself with:

```
git clone https://github.com/adamancer/ea-drought-related-burn-severity
cd ea-drought-related-burn-severity
pip install -e .
```


## Usage

**The data required to run the notebooks in this repository is not currently
available for download. Instructions for downloading the data will be added
as soon as the data is available.**

The package includes two directories:

+ **notebooks** contains a set of Jupyter Notebooks used to explore and model
  climate, vegetation, and burn severity data
+ **ea_drought_related_burn_severity** contains a set of utility functions
  used by the notebooks to read, process, and plot raster data

You can run the included notebooks as follows:

```
conda activate ea-drought-related-burn-severity
cd path/to/ea-drought-related-burn-severity
jupyter notebook
```

You can also access the utility functions defined in this package directly.
For example, the `plot_bands` function simplifies plotting an xarray.DataArray
using the earthpy library:

```
import rioxarray as rxr

from ea_drought_related_burn_severity.utils import plot_bands

xda = rxr.open_rasterio("path/to/raster.tif", masked=True)
plot_bands(xda)
```