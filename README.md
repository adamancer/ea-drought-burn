# ea-drought-burn

[![DOI](https://zenodo.org/badge/359515921.svg)](https://zenodo.org/badge/latestdoi/359515921)

This repository contains code for evaluating the effect of vegetation
mortality on burn severity during the Woolsey Fire. It was developed as
part of the
[CU Boulder Earth Data Analytics Certificate Program](https://earthlab.colorado.edu/earth-data-analytics-professional-graduate-certificate).


## Background

The Woolsey Fire burned nearly 100,000 acres near Malibu, CA in November 2018.
A four-year drought preceded the fire, resulting in widespread dieback of
grass, shrubs, and trees in and around the area that burned. This project
seeks to understand how the dieback affected the severity of the fire. Did
areas where more vegetation die burn more severely during the fire? Can
satellite-based estimates of dieback be used to inform planning for future
wildfires?

In this repository, I've used the random-forest machine-learning algorithm
to try to evaluate these questions.


## Installation

The following software is required to install this package using the
instructions below:

+ [Git](https://git-scm.com/downloads)
+ [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Once you have Git and Miniconda installed, open the command line and run the
following commands to set up the environment needed to run this package:

```
git clone https://github.com/adamancer/ea-drought-burn
cd ea-drought-burn
conda env create --file environment.yml
conda activate ea-drought-burn
```

Then install the ea-drought-burn scripts using:

```
pip install -e .
```


## Usage

*The data required for this project is not currently available for download.
Some data was prepared by other researchers, and I do not have permission to
share it. To run the notebooks in this repository, please contact me to
obtain a copy of the data as a zip file, then extract it to
**~/earth-analytics/data/woolsey-fire**.*

The package includes the following directories:

+ **ea_drought_burn** contains a set of utility functions used by the
  notebooks to read, process, and plot raster data.

+ **notebooks** contains a set of Jupyter Notebooks used to explore and model
  climate, vegetation, and burn severity data related to the Woolsey Fire.

+ **reports** contains documents summarizing the results of the project.


### Notebooks

Notebooks include:

+ **0-run-all-notebooks.ipynb** runs all notebooks in the notebooks directory

+ **1-load-data.ipynb** loads and provides reference info for the data used in
  this project. The full list of references is below.

+ **2-data-exploration.ipynb** includes plots and descriptions of the
  data used in the burn-severity model.

+ **3-random-forest.ipynb** runs a random-forest model that predicts
  burn severity for the Woolsey Fire based on pre-fire conditions. Variables,
  sampling strategies, and the area of interest are all adjustable.
  By default, individual runs are saved to
  **~/earth-analytics/data/woolsey-fire/outputs/models**.

+ **4-view-model-results.ipynb** allow you to view and compare results of
  previous models.

+ **5-project-report.ipynb** generates an HTML blog post summarizing some
  results of this project for a general audience.

Once you have a copy of the data in the right place, you can run the
notebooks using the Jupyter Notebook interface:

```
conda activate ea-drought-burn
cd path/to/ea-drought-burn
jupyter notebook
```

Open and run **0-run-all-notebooks.ipynb** in the Jupyter Notebook interface
to run all notebooks at once and recreate the HTML report in the **reports**
directory.


### Utility functions

The utility functions defined in ea_drought_burn can be accessed directly. For
example, the `plot_bands` function simplifies plotting an xarray.DataArray
using the earthpy library:

```python
import rioxarray as rxr

from ea_drought_burn.utils import plot_bands


xda = rxr.open_rasterio("path/to/raster.tif", masked=True)
plot_bands(xda)
```


## Citation

Please see the [Zenodo record](https://doi.org/10.5281/zenodo.4798754) for this repository for a version-specific citation.


## References

Data and publications used in this repository include:

+ CA State Boundary. Available from:
  https://data.ca.gov/dataset/ca-geographic-boundaries/resource/3db1e426-fb51-44f5-82d5-a54d7c6e188b.


+ Dagit R, Contreras S, Daukiss R, Spyrka A, Quelly N, Foster K, Nickmeyer A,
  Rousseau B, Chang E. How can we save our native trees? Drought and Invasive
  Beetle impacts on Wildland Trees and Shrublands in the Santa Monica
  Mountains. Final Report for Los Angeles County Contract CP-03-44. 2017.
  Available from:
  https://www.rcdsmm.org/wp-content/uploads/2016/04/Drought-and-Invasive-Beetle-impacts-RCDSMM-1.2.18.pdf.


+ Eidenshink J, Schwind B, Brewer K et al. A Project for Monitoring
  Trends in Burn Severity. Fire Ecol. 2007;3:3-21.
  doi:[10.4996/fireecology.0301003](https://doi.org/10.4996/fireecology.0301003).


+ Foster K, Queally N, Nickmeyer A, Rousseau N. Appendix: Santa Monica
  Mountains Ecological Forecasting II: Utilizing NASA Earth Observations to
  Determine Drought Dieback and Insect-related Damage in the Santa Monica
  Mountains, California. 2017A. Avalable from:
  https://www.rcdsmm.org/wp-content/uploads/2016/04/Drought-and-Tree-Appendices_12.15.17.pdf.


+ Foster K, Queally N, Nickmeyer A, Rousseau N. Utilizing NASA Earth
  Observations to Determine Drought Dieback and Insect-related Damage in the
  Santa Monica Mountains, California. 2017B. Available from:
  https://develop.larc.nasa.gov/2017/fall/posters/2017Fall_JPL_SantaMonicaMountainsEcoII_Poster.pdf


+ Gao B. NDWI—A normalized difference water index for remote sensing of
  vegetation liquid water from space. Remote Sens Environ. 1996;58(3):257-266.
  doi:[10.1016/S0034-4257(96)00067-3](https://doi.org/10.1016/S0034-4257(96)00067-3).


+ Hicke JA, Johnson MC, Hayes JL, Presiler HK. Effects of bark beetle-caused
  tree mortality on wildfire. For Ecol Manag. 2013;271:81-90.
  doi:[10.1016/j.foreco.2012.02.005](https://doi.org/10.1016/j.foreco.2012.02.005).


+ McCune B, Keon D. Equations for potential annual direct incident
  radiation and heat load. Jour Veg Sci. 2002;13(4):603-606.
  doi:[10.1111/j.1654-1103.2002.tb02087.x](https://doi.org/10.1111/j.1654-1103.2002.tb02087.x).


+ National Interagency Fire Center. Historic Perimeters Combined
  2000-2018. 2019. Available from:
  https://data-nifc.opendata.arcgis.com/datasets/historic-perimeters-combined-2000-2018.


+ National Interagency Fire Center. Interagency Fire Perimeter History -
  All Years. 2021. Available from:
  https://data-nifc.opendata.arcgis.com/datasets/4454e5d8e8c44b0280258b51bcf24794_0.


+ Rao K, Williams AP, Flefil JF, Konings AG. SAR-enhanced mapping of
  live fuel moisture content. Remote Sens Environ. 2020;245:111797.
  doi:[10.1016/j.rse.2020.111797](https://doi.org/10.1016/j.rse.2020.111797).


+ PRISM Climate Group, Oregon State University https://prism.oregonstate.edu,
  created October 2017.


+ PRISM Climate Group, Oregon State University https://prism.oregonstate.edu,
  created June 2021.
