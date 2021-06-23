"""Defines module-wide constants"""
import os

import earthpy as et




# CRS to use for all geographic data
CRS = "EPSG:32611"

# Path to directory containing the woolsey-fire folder
DATA_DIR = os.path.join(et.io.HOME, "earth-analytics", "data")

# Path to the project directory (do not change)
PROJ_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
