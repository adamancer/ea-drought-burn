"""Defines utility functions for working with geospatial data"""
from functools import wraps
import hashlib
import itertools
import json
import os
import re
import tempfile
import warnings
import zipfile

import earthpy.plot as ep
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy_groupies as npg
from rasterio.plot import plotting_extent as rasterio_plotting_extent
import rioxarray as rxr
import requests
import seaborn as sns
from shapely.geometry import box
from skimage.measure import block_reduce
import xarray as xr




def checksum(path, size=8192):
    """Generates MD5 checksum for a file

    Parameters
    ----------
    path: str
        path of file to hash
    size: int (optional)
        size of block. Must be multiple of 128.

    Returns
    -------
        MD5 checksum of file
    """
    if size % 128:
        raise ValueError("Size must be a multiple of 128")
    with open(path, "rb") as f:
        md5_hash = hashlib.md5()
        while True:
            chunk = f.read(size)
            if not chunk:
                break
            md5_hash.update(chunk)
        return md5_hash.hexdigest()


def checksum_walk(path, output=None):
    """Walks path and generates a MD5 checksum for each file

    Parameters
    ----------
    path: str
        path of file to hash
    output: str (optional)
        path to which to write the checksums. If not given, a checksums.json
        file is added in each directory.

    Returns
    -------
    None
    """

    if output is not None:
        checksum_path = os.path.abspath(output)
        checksum_dir = os.path.dirname(checksum_path)
        try:
            with open(output, "r") as f:
                checksums = json.load(f)
        except IOError:
            checksums = {}

    for root, dirs, files in os.walk(path):
        if files:

            # Look for existing checksums
            if output is None:
                checksum_path = os.path.join(root, "checksums.json")
                try:
                    with open(checksum_path, "r") as f:
                        checksums = json.load(f)
                except IOError:
                    checksums = {}

            # Checksum each data file
            for fn in files:
                if (
                    fn != "checksums.json"
                    and os.path.splitext(fn)[-1] != ".md5"
                ):

                    fp = os.path.abspath(os.path.join(root, fn))

                    # Use the relative path as the key if using one file
                    if output:
                        key = fp[len(checksum_dir):].replace("\\", "/") \
                                                    .lstrip("/")
                    else:
                        output = fn

                    try:
                        checksums[key]
                    except KeyError:
                        checksums[key] = checksum(fp)

            # Save checksums as JSON
            with open(checksum_path, "w") as f:
                json.dump(checksums, f, indent=2)


def download_file(url, path):
    """Downloads file at url to path

    Parameters
    ----------
    url: str
        url to download
    path: str
        path to download to

    Returns
    -------
    None
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def join_and_makedirs(*args):
    """Joins path and creates parent directories if needed


    Parameters
    ----------
    args: str
        one or more path segments

    Returns
    -------
    str
        the joined path
    """
    path = os.path.join(*args)

    # Get parent directory if file
    parents = path
    if re.search(r"\.[a-z]{2,4}$", parents, flags=re.I):
        parents = os.path.dirname(path)

    # Create parent directories as needed
    try:
        os.makedirs(parents)
    except OSError:
        pass

    return path


def zip_shapefile(gdf, path):
    """Exports GeoDataFrame to shapefile and zips it

    The resulting zip file can be use to search USGS Earth Explorer.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        the data frame to export
    path: str
        the path to the zip file

    Returns
    -------
    None
    """

    # Write shapefile files to a temporary directory
    tmpdir = tempfile.TemporaryDirectory()
    stem = os.path.basename(os.path.splitext(path)[0])
    gdf.to_file(os.path.join(tmpdir.name, f"{stem}.shp"))

    # Zip the temporary files
    with zipfile.ZipFile(path, "w") as f:
        for fn in os.listdir(tmpdir.name):
            f.write(os.path.join(tmpdir.name, fn), fn, zipfile.ZIP_DEFLATED)


def create_sampling_mask(xda, counts, seed=None, path=None):
    """Creates mask for a sample of a data array

    Parameters
    ----------
    xda: xarray.DataArray
        array on which to base the sampling mask
    counts: int, dict, or iterable
        counts for sampling mask. If a dict with string, keys are used to
        label the returned data array.
    seed: int (optional)
        seed for the random number generator
    path: str (optional)
        save mask to path if given

    Returns
    -------
    xarray.DataArray
        array containing each mask as a band
    """

    # Convert counts to dict
    if isinstance(counts, int):
        counts = [counts]
    if not isinstance(counts, dict):
        counts = {i: c for i, c in enumerate(counts)}
    if not counts:
        raise ValueError("Must provide counts")

    # Create 1D arrays of values and indexes
    vals = np.ravel(xda)
    indexes = np.arange(len(vals))[np.isfinite(vals)]

    # Create the sample
    rng = np.random.default_rng(seed)
    pool = rng.choice(indexes, sum(counts.values()), replace=False).tolist()

    # Map the sample back to a 2D array for each band
    masks = {}
    for key, count in counts.items():

        # Take the sample from the pool
        sample, pool = pool[:count], pool[count:]

        # Build mask based on the sample
        mask = np.full(vals.shape, False)
        for i in sample:
            mask[i] = True

        # Reshape to a 2D data array and add to output
        masks[key] = xr.DataArray(mask.reshape(xda.shape),
                                  coords={"y": xda.y, "x": xda.x},
                                  dims=["y", "x"])

        # Add band attribute for when bands are concatenated
        masks[key]["band"] = len(masks)

    # Create and label data array from masks
    sampling_mask = xr.concat(masks.values(), dim="band")
    if not isinstance(key, int):
        sampling_mask.attrs["long_name"] = list(counts.keys())
    sampling_mask["spatial_ref"] = 0
    sampling_mask["spatial_ref"].attrs = xda["spatial_ref"].attrs

    # Write mask to raster file if path given
    if path:
        sampling_mask.rio.to_raster(path)

    # Convert to a true-false mask
    return sampling_mask


def load_nifc_fires(path, fire_ids=None, crs=None, **kwargs):
    """Loads fires matching the given IDs from NIFC shapefile

    Parameters
    ----------
    path: str
        path to the NIFC fire perimeter shapefile
    fire_ids: list-like (optional)
        list of fire IDs to return. If not provided, all fires are returned.
    crs: str or int (optional)
        the CRS to convert the raster to as an EPSG code
    kwargs (optional):
        any keyword argument accepted by `geopandas.read_file`

    Returns
    -------
    geopandas.GeoDataFrame
        dataframe of matching fires
    """

    # Load the NIFC fire shapefile
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        nifc_fires = gpd.read_file(path, **kwargs)

    # Reproject to given CRS if needed
    if crs and nifc_fires.crs != crs:
        nifc_fires.to_crs(crs, inplace=True)

    # Limit dataframe to required columns
    nifc_fires = nifc_fires[[
        "agency",
        "uniquefire",
        "incidentna",
        "complexnam",
        "fireyear",
        "perimeterd",
        "gisacres",
        "geometry"
    ]]

    # Convert perimeterd to date
    # FIXME: Using datetimes breaks gdf.to_file
    #nifc_fires["perimeterd"] = pd.to_datetime(nifc_fires["perimeterd"])

    if fire_ids:
        return nifc_fires[nifc_fires.uniquefire.isin(fire_ids)]
    return nifc_fires


def copy_xr_metadata(xobj, other):
    """Copies metadata from one xarray object to another

    Parameters
    ---------
    xarr: xarray.DataArray or xarray.Dataset
        the array/dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray or xarray.Datset
       Copy of other converted to type of xobj with metadata applied
    """
    if isinstance(xobj, xr.DataArray):
        return copy_array_metadata(xobj, other)
    return copy_dataset_metadata(xobj, other)


def copy_array_metadata(xarr, other):
    """Copies metadata from an array to another object

    Looks at the shape and length of xarr and other to decide which
    metadata to copy.

    Parameters
    ---------
    xarr: xarray.DataArray
        the array to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray
       Copy of other with metadata applied
    """

    # Convert a list, etc. to an array
    if isinstance(other, (list, tuple)):
        other = np.array(other)

    # If arrays have the same shape, copy all metadata
    if xarr.shape == other.shape:
        return xarr.__class__(other, dims=xarr.dims, coords=xarr.coords)

    # If arrays have the same number of layers, copy scalar coordinates
    # and any other coordinates with same the length as the array
    if len(xarr) == len(other):
        coords = {
            k: v for k, v in xarr.coords.items()
            if not v.shape or v.shape[0] == len(xarr)
        }
        dims = [d for d in xarr.dims if d in coords]
        return xarr.__class__(other, dims=dims, coords=coords)

    # If arrays have the same xy, copy spatial and scalar coordinates
    # to each band in the other array
    if xarr.shape[-2:] == other.shape[-2:]:
        for xband in iterarrays(xarr):
            coords = {k: v for k, v in xband.coords.items()
                      if k not in xband.dims}

            obands = []
            for oband in iterarrays(other):
                obands.append(xband.__class__(oband,
                                              dims=xband.dims,
                                              coords=xband.coords))

            return xr.concat(obands, dim="band")

    raise ValueError("Could not copy xr metadata")


def copy_dataset_metadata(xdat, other):
    """Copies metadata from a dataset to another object

    Parameters
    ---------
    xarr: xarray.Dataset
        the dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.Dataset
       Copy of other with metadata applied
    """
    xarr = xdat.to_array(dim="band")
    return copy_array_metadata(xarr, other).to_dataset(dim="band")


def concat_arrays(xdas, dim="band", names=None):
    """Concatenates a list of arrays along a numeric band

    Parameters
    ---------
    xdas: list of xarray.DataArrays
        arrays to concatenate
    dim: str (optional)
        name of the dim to concatenate along
    names: list of str (optional)
        long names for each band to be stored in the long_name attr

    Returns
    -------
    xarray.DataArray
        array with each of the source arrays as a band
    """
    bands = []
    for band in xdas:

        # Set or update dim
        try:
            del band[dim]
        except KeyError:
            pass
        band = band.squeeze()
        band[dim] = len(bands)

        bands.append(band)

    xda = xr.concat(bands, dim=dim)

    # Assign names to long_name
    if names:
        xda.attrs["long_name"] = names

    return xda


def iterarrays(obj):
    """Iterates through an array or dataset

    Parameters
    ---------
    obj: xarray.DataArray or xarray.Dataset
        the object to iterate

    Returns
    -------
    iterable
        list or similar of the children of the given object
    """
    if isinstance(obj, xr.Dataset):
        return obj.values()
    return obj if len(obj.shape) > 2 else [obj]


def to_numpy_array(obj, dim="band"):
    """Converts an object to a numpy array

    Parameters
    ----------
    obj: array-like
        a numpy array, xarray object or any other object that can converted
        to a numpy array using numpy.array()
    dim: str
        the name of dimension of the new array when converting a dataset

    Returns
    -------
    numpy.array
        an array based on the given object
    """
    if isinstance(obj, xr.Dataset):
        xobj = xobj.to_array(dim=dim)
    if isinstance(obj, xr.DataArray):
        return obj.values
    if isinstance(obj, (list, tuple)):
        return np.array(obj)
    return obj


def plotting_extent(xobj):
    """Calculates plotting extent for an xarray object for matplotlib

    Parameters
    ----------
    xobj: xarray.DataArray or xarray.Dataset
        the xarray object to scale

    Returns
    -------
    tuple of float
        left, right, bottom, top
    """
    for xarr in iterarrays(xobj):
        return rasterio_plotting_extent(xarr, xarr.rio.transform())


def open_raster(path, crs=None, crop_bound=None, **kwargs):
    """Opens, reprojects, clips, and squeezes a raster file

    Parameters
    ----------
    path: str
        the path to a raster file
    crs: str or int (optional)
        the CRS to convert the raster to as an EPSG code
    crop_bound (optional):
        the geometry to clip the data to
    kwargs (optional):
        any keyword argument accepted by `rioxarray.open_rasterio`

    Returns
    -------
    xarray.DataArray
        the reprojected, clipped, and squeezed array

    """
    kwargs.setdefault("masked", True)
    xda = rxr.open_rasterio(path, **kwargs)

    # Reproject to CRS
    if crs is not None and xda.rio.crs != crs:
        xda = xda.rio.reproject(crs)

    # Reproject to CRS
    if crop_bound is not None:
        if crop_bound.crs != xda.rio.crs:
            crop_bound = crop_bound.to_crs(xda.rio.crs)
        xda = xda.rio.clip(crop_bound, drop=True, from_disk=True)

    return xda.squeeze()


def reproject_match(xda, match_xda, **kwargs):
    """Forces reprojection to use exact x, y coordinates of original array

    The rioxarray reproject_match function can produce small differences in
    coordinates (around -4e10) that break xarray.align, so this function
    tries to guarantee that the coordinates are exactly the same.

    Parameters
    ----------
    xda: xarray.DataArray
        the data array to reproject
    match_xda: xarray.DataArray
        the data array to match
    kwargs(optional):
        any keyword argument accepted by the `rio.reproject_match` method

    Returns
    -------
    xda.DataArray
        the reprojected data array
    """

    # Reproject using the built-in rio method
    reproj = xda.rio.reproject_match(match_xda, **kwargs)

    # Copy metadata from original. This manually overwrites the metadata from
    # the reprojection, including the decimal offset.
    reproj = copy_xr_metadata(match_xda, reproj)

    # Reindex reprojection. Some arrays end up with indexes that don't match
    # the reprojected coordinates, which produces weird calculation errors
    # (for example, operations between arrays of the same sizes producing
    # result arrays of a different, incorrect size). Force the reindex to
    # ensure that the x and y indexes are up-to-date.
    reproj = reproj.reindex({"x": reproj.x, "y": reproj.y})

    return reproj


def find_scenes(src):
    """Finds and groups all files that are part of a scene

    Parameters
    ----------
    src: str
        the path to a directory containing Landsat or Sentinel data where
        each band is a separate file

    Returns
    -------
    dict of dicts
        dict of scenes. Each scene is arranged by band number.

    """

    # Satellite-specific regex patterns
    patterns = {
        "landsat": r"((?<=band)(\d)|[a-z]+_qa)\.tif$",
        "sentinel": r"_B([01]?[0-9]A?)\.jp2$",
    }

    scenes = {}
    for root, dirs, files in os.walk(src):
        for pattern in patterns.values():
            for fn in files:

                try:
                    band = re.search(pattern, fn).group(1).lstrip("0")
                except AttributeError:
                    pass
                else:
                    # Get scene using the path to the file
                    segments = root.split(os.sep)
                    scene = segments.pop(-1)
                    while not re.search(r"\d", scene):
                        scene = segments.pop(-1)

                    scenes.setdefault(scene, {})[band] = os.path.join(root, fn)

    return scenes


def stack_scene(scene, reproj_to=None, **kwargs):
    """Stacks all files that are part of a scene in a single array

    Parameters
    ----------
    scene: dict
        dict organized as band: path for each band in the scene
    reproj_to: xarray.DataArray (optional)
        an array to reporject the data to. If not given, reproject all bands
        to the highest resolution in the source data.
    kwargs (optional):
        any keyword argument accepted by `open_raster`

    Returns
    -------
    xarray.DataArray
        array with each band as a layer
    """

    def sortable(val):
        """Returns a sortable representation of the band number"""
        return val.zfill(16 + len(re.search("[a-z]*$", val.lower()).group()))

    layers = []
    attrs = {}
    for band in sorted(scene, key=sortable):

        layers.append(open_raster(scene[band], **kwargs))

        # Update band number
        del layers[-1]["band"]
        layers[-1] = layers[-1].squeeze()
        layers[-1]["band"] = len(layers)

        attrs.setdefault("band_name", []).append(band)
        attrs.setdefault("long_name", []).append(os.path.basename(scene[band]))

    # Reproject layers to the same resolution. Reproject to the highest
    # resolution in the input data if no reference array is provided.
    if reproj_to is None:
        res = min([lyr.rio.resolution()[0] for lyr in layers])
        reproj_to = [lyr for lyr in layers if lyr.rio.resolution()[0] == res][0]
    layers = [reproject_match(lyr, reproj_to) for lyr in layers]

    xda = xr.concat(layers, dim="band")
    for key, val in attrs.items():
        xda.attrs[key] = val

    return xda


def get_long_name(xda, name):
    """Get layer from array corresponding to a value in long_name

    Parameters
    ----------
    xda: xarray.DateArray
        an array with a long_name attribute
    name: str
        the name of the band to return

    Returns
    -------
    xarray.DataArray
        the band matching the given name
    """
    names = xda.attrs["long_name"]
    try:
        return xda[names.index(name)].copy()
    except ValueError:
        matches = [i for i, n in enumerate(names) if n.startswith(name)]
        if len(matches) == 1:
            return xda[matches[0]].copy()
        raise ValueError(f"Could not resolve {name} uniquely")


def plot_regression(x, y, ax, color="gray", **kwargs):
    """Plots linear regression with correlation coefficient

    Parameters
    ----------
    x: numpy.array or similar
        x data for the regression
    y: numpy.array or similar
        y data for the regression
    ax: matplotlib.Axes object
        the axis on which to plot
    color: str or tuple (optional)
        the color of the points in a format recognized by matplotlib

    Returns
    -------
    None
    """

    # Format axis based on kwargs
    ax.set(**kwargs)

    # Plot linear regression
    sns.regplot(x="x",
                y="y",
                data={"x": x, "y": y},
                #x_estimator=np.mean,
                ax=ax,
                color=color)

    # Calculate correlation coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = np.corrcoef(x, y)[0, 1] ** 2

    ax.annotate(f"r²={r2:.2f}",
                (0.98, 0.98),
                xycoords="axes fraction",
                ha="right",
                va="top")


def create_figure(n_rows, n_cols, title=None):
    """Creates a figure based on the supplied arguments

    Parameters
    ----------
    n_rows: int
        number of rows in the figure
    n_cols: int
        number of columns in the figure
    title: str (optional)
        title of the figure

    Returns
    -------
    tuple
        Tuple of (figure, axes) where axes is a list
    """

    n_rows = int(n_rows)
    n_cols = int(n_cols)

    width = 8 * n_cols
    height = n_rows * width / n_cols

    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(width, height),
                             constrained_layout=True)

    if title:
        fig.suptitle(title, fontsize=32)

    # Create list of axes
    try:
        axes = list(itertools.chain.from_iterable(axes))
    except TypeError:
        # Chaining fails 1D arrays, but this does need to be a list
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

    return fig, axes


def aggregate(xda, idx_or_size, func=np.nanmean, fill_value=np.nan):
    """Aggregates a 2D array using an index array or block size

    Parameters
    ----------
    xda: xarray.DataArray
        the array with the data to aggregate
    idx_or_size: xarray.DataArray, int, or tuple
        either an array with each pixel indexed based on the value in the
        source array or the size in pixels of the desired grid
    func: callable (optional)
        the numpy function used to aggregate each block
    fill_value: int or float (optional)
        the value to use for missing values when cells do not fit
        perfectly into the original array


    Returns
    -------
    numpy.array
        array containing the aggregated values
    """

    if isinstance(idx_or_size, int):
        idx_or_size = (idx_or_size, idx_or_size)

    # Coerce array to a numpy array
    arr = to_numpy_array(xda)

    # Use scipy to split array into grid if block size given
    if isinstance(idx_or_size, (list, tuple)):
        return block_reduce(arr, idx_or_size, func=func, cval=fill_value)

    # Use numpy_groupies to group on an index array
    idx = np.ravel(to_numpy_array(idx_or_size))

    # Use nodata from index if set
    nodata = fill_value
    if isinstance(idx_or_size, xr.DataArray):
        nodata = idx_or_size.rio.nodata

    # Set data to fill_value where nodata in index
    vals = np.ravel(arr)
    vals[idx == nodata] = fill_value

    return npg.aggregate(idx, vals, func=func, fill_value=fill_value)


def agg_to_raster(xda, idx_or_size, func=np.nanmean, fill_value=np.nan):
    """Aggregates data and saves as a raster matching the original

    Parameters
    ----------
    xda: xarray.DataArray
        the array with the data to aggregate
    idx_or_size: xarray.DataArray, int, or tuple
        either an array with each pixel indexed based on the value in the
        source array or the size in pixels of the desired grid
    func: callable (optional)
        the numpy function used to aggreaget each block
    fill_value: int or float (optional)
        the value to use for missing values when cells do not fit perfectly
        into the original array


    Returns
    -------
    xarray.DataArray
        array containing the aggregated values with geospatial information
    """
    agg = np.ravel(aggregate(xda, idx_or_size, func, fill_value=fill_value))

    if isinstance(idx_or_size, int):
        idx_or_size = (idx_or_size, idx_or_size)

    # Convert block_reduce aggregate back to a georeferenced array
    if isinstance(idx_or_size, (list, tuple)):
        xda = xda.copy()
        arr = xda.values
        n_rows, n_cols = xda.shape[-2:]
        row_size, col_size = idx_or_size

        i = 0
        row = 0
        col = 0
        while row < n_rows:
            while col < n_cols:
                arr[row:row+row_size, col:col+col_size] = agg[i]
                i += 1
                col += col_size
            row += row_size
            col = 0

    # Convert index-based aggregate back to a georeferenced array
    else:
        xda = idx_or_size.copy()
        for i, val in enumerate(agg):
            xda = xr.where(xda == i, val, xda)
    return xda


def extract_grid(xda, path=None, nodata=0):
    """Extracts grid matching input array

    Used to extract the grid from an array where the data is coarser than the
    resolution (for example, 4-km blocks that have been resampled to 30-m
    resolution).

    Parameters
    ----------
    xda: xarray.DataArray
        the gridded array
    path: str (optional)
        save indexed grid to path if given
    nodata: int (optional)
        nodata value for the grid array

    Returns
    -------
    xarray.DataArray
        an array with each pixel indexed based on the value in the source array
    """

    # Create a copy of the input array
    grid = xda.squeeze()

    # Iterate through each row to find column boundaries
    cols = []
    for i, row in enumerate(xda.values):
        cols.append(0)
        last = None
        for j, col in enumerate(row):
            if col != last:
                cols.append(j)
            last = col

    # Iterate through each column to find row boundaries
    rows = []
    for i, row in enumerate(xda.values.T):
        rows.append(0)
        last = None
        for j, col in enumerate(row):
            if col != last:
                rows.append(j)
            last = col

    # Get unique values for boundaries
    rows = sorted(set(rows))
    cols = sorted(set(cols))

    # Create the reference grid based on cell boundaries
    grid = grid.where(np.isfinite(grid), nodata)
    arr = grid.values

    # Assign an index to each grid cell
    val = 1
    for i, i_start in enumerate(rows):

        try:
            i_end = rows[i + 1]
        except IndexError:
            i_end = None

        for j, j_start in enumerate(cols):
            try:
                j_end = cols[j + 1]
            except IndexError:
                j_end = None

            arr[i_start:i_end, j_start:j_end] = val
            val += 1

    # Write grid to raster with nodata set to 0
    xda = xr.DataArray(arr,
                       coords={"y": grid.y,
                               "x": grid.x,
                               "spatial_ref": grid.spatial_ref},
                       dims=["y", "x"])

    xda = xda.astype(int)
    xda.rio.write_nodata(nodata, inplace=True)
    if path is not None:
        xda.rio.to_raster(path)

    return xda


def bounds_to_gdf(xda, name="Bounds"):
    """Converts raster bounds to GeoDataFrame

    Parameters
    ----------
    xda: xarray.DataArray
        the source array
    names: str (optional)
        name for the row in the GeoDataFrame

    Returns
    -------
    geopandas.GeoDataFrame
        Dataframe containing the bounds as a geometry
    """
    return gpd.GeoDataFrame(
        {"name": name, "geometry": [box(*xda.rio.bounds())]},
        crs=xda.rio.crs
    )


def plot_xarray(func):
    """Wraps an xarray object to allow plotting with earthpy

    Parameters
    ----------
    func: callable
        an earthpy.plot function

    Returns
    -------
    mixed
        result of the wrapped function
    """

    @wraps(func)
    def wrapped(*args, **kwargs):

        # Convert first argument to a masked array to plot with earthpy
        args = list(args)
        arr = to_numpy_array(args[0])

        # Automatically assign extent for plots if rio accessor is active
        if func.__name__.startswith("plot_"):
            try:
                kwargs.setdefault("extent", plotting_extent(args[0]))
            except AttributeError:
                # Fails if rio accessor has not been loaded
                pass

        # HACK: Masked arrays cannot be stretched because they are not
        # handled intuitively by the np.percentile function used by the
        # earthpy internals. To get around that, the decorator forces NaN
        # values to 0 when stretch is True.
        if kwargs.get("stretch"):
            arr = to_numpy_array(args[0].fillna(0))
        else:
            arr = np.ma.masked_invalid(arr)

        return func(arr, *args[1:], **kwargs)
    return wrapped


@plot_xarray
def hist(*args, **kwargs):
    """Plots histogram based on an xarray object"""
    return ep.hist(*args, **kwargs)


@plot_xarray
def plot_bands(*args, **kwargs):
    """Plots bands based on an xarray object"""
    return ep.plot_bands(*args, **kwargs)


@plot_xarray
def plot_rgb(*args, **kwargs):
    """Plots RGB based on an xarray object"""
    return ep.plot_rgb(*args, **kwargs)


def draw_legend(im_ax, bbox=(1.05, 1), titles=None, cmap=None, classes=None):
    """Create a custom legend with a box for each class in a raster.

    This is an exact copy of the `earthpy.plot.draw_legend` function except
    that it uses a relative font size in the legend to allow the text to
    scale with image size, plus a couple tweaks to allow it to run outside of
    its origianl context.

    Parameters
    ----------
    im_ax : matplotlib image object
        This is the image returned from a call to imshow().
    bbox : tuple (default = (1.05, 1))
        This is the bbox_to_anchor argument that will place the legend
        anywhere on or around your plot.
    titles : list (optional)
        A list of a title or category for each unique value in your raster.
        This is the label that will go next to each box in your legend. If
        nothing is provided, a generic "Category x" will be populated.
    cmap : str (optional)
        Colormap name to be used for legend items.
    classes : list (optional)
        A list of unique values found in the numpy array that you wish to plot.

    Returns
    ----------
    matplotlib.pyplot.legend
        A matplotlib legend object to be placed on the plot.
    """

    # Lazy load imports to keep this function self-contained so that it can
    # be removed cleanly if a better solution presents itself
    from matplotlib import patches as mpatches
    from matplotlib.colors import ListedColormap

    try:
        im_ax.axes
    except AttributeError:
        raise AttributeError(
            "The legend function requires a matplotlib axis object to "
            "run properly. You have provided a {}.".format(type(im_ax))
        )

    # If classes not provided, get them from the im array in the ax object
    # Else use provided vals
    if classes is not None:
        # Get the colormap from the mpl object
        cmap = im_ax.cmap.name

        # If the colormap is manually generated from a list
        if cmap == "from_list":
            cmap = ListedColormap(im_ax.cmap.colors)

        colors = ep.make_col_list(
            nclasses=len(classes), unique_vals=classes, cmap=cmap
        )
        # If there are more colors than classes, raise value error
        if len(set(colors)) < len(classes):
            raise ValueError(
                "There are more classes than colors in your cmap. "
                "Please provide a ListedColormap with the same number "
                "of colors as classes."
            )

    else:
        classes = list(np.unique(im_ax.axes.get_images()[0].get_array()))
        # Remove masked values, could next this list comp but keeping it simple
        classes = [
            aclass for aclass in classes if aclass is not np.ma.core.masked
        ]
        colors = [im_ax.cmap(im_ax.norm(aclass)) for aclass in classes]

    # If titles are not provided, create filler titles
    if not titles:
        titles = ["Category {}".format(i + 1) for i in range(len(classes))]

    if not len(classes) == len(titles):
        raise ValueError(
            "The number of classes should equal the number of "
            "titles. You have provided {0} classes and {1} titles.".format(
                len(classes), len(titles)
            )
        )

    patches = [
        mpatches.Patch(color=colors[i], label="{lab}".format(lab=titles[i]))
        for i in range(len(titles))
    ]
    # Get the axis for the legend
    ax = im_ax.axes
    return ax.legend(
        handles=patches,
        bbox_to_anchor=bbox,
        loc=2,
        borderaxespad=0.0,
        prop={"size": "small"},
    )
