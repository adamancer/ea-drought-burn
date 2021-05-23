# ea-drought-burn

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
conda activate ea-drought-burn
```

Then install the package itself with:

```
git clone https://github.com/adamancer/ea-drought-burn
cd ea-drought-burn
pip install -e .
```


## Usage

**The data required to run the notebooks in this repository is not currently
available for download. Instructions for downloading the data will be added
as soon as the data is available.**

The package includes two directories:

+ **ea_drought_burn** contains a set of utility functions
  used by the notebooks to read, process, and plot raster data
+ **notebooks** contains a set of Jupyter Notebooks used to explore and model
  climate, vegetation, and burn severity data
+ **reports** contains documents summarizing the results of the project (in
  progress)

You can run the project notebooks as follows:

```
conda activate ea-drought-burn
cd path/to/ea-drought-burn
jupyter notebook
```

You can also create and view a report summarizing the project. The
instructions below update the HTML report in the reports directory of this
repository, but you can change the `html_path` variable to another location
if needed.

```python
import os

from ea_drought_burn.config import PROJ_DIR
from ea_drought_burn.run import notebook_to_html

# Update HTML report
nb_path = os.path.join(PROJ_DIR, "notebooks", "project-report.ipynb")
html_path = os.path.join(PROJ_DIR, "reports", "project-report.htm")
notebook_to_html(nb_path, html_path)

# Open report in default browser
os.startfile(html_path)
```

The utility functions in ea_drought_burn can be accessed directly. For example, 
the `plot_bands` function simplifies plotting an xarray.DataArray using the
earthpy library:

```python
import rioxarray as rxr
from ea_drought_burn.utils import plot_bands

xda = rxr.open_rasterio("path/to/raster.tif", masked=True)
plot_bands(xda)
```
