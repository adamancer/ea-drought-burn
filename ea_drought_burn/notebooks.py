"""Defines functions for clearing and running notebooks"""
import glob
import os

import nbformat
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import (
    ClearOutputPreprocessor,
    ExecutePreprocessor
)
from traitlets.config import Config

from .config import PROJ_DIR




NOTEBOOK_DIR = os.path.realpath(os.path.join(PROJ_DIR, "notebooks"))




def run_notebook(path):
    """Runs a notebook and saves the output

    Parameters
    ----------
    path: str
        path to a Jupyter Notebook

    Returns
    -------
    None
    """

    with open(_get_path(path), encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor()
        ep.preprocess(nb, {})

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def clear_notebook(path):
    """Clears output from a notebook

    Parameters
    ----------
    path: str (optional)
        path to a Jupyter Notebook

    Returns
    -------
    None
    """

    with open(_get_path(path), encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        pp = ClearOutputPreprocessor()
        pp.preprocess(nb, {})

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def notebook_to_html(nb_path, html_path, run=False):
    """Converts notebook to HTML and strips specified blocks

    Source: https://github.com/jupyter/nbconvert/issues/1194#issuecomment-839360322

    Parameters
    ----------
    nb_path: str
        path to a Jupyter Notebook
    html_path: str
        path to output HTML to
    run: bool (optional)
        whether to run the notebook before converting it

    Returns
    -------
    None
    """

    nb_path = _get_path(nb_path)

    # Run the notebook first if specified
    if run:
        run_notebook(nb_path)

    # Setup config
    cfg = Config()

    # Configure tag removal - be sure to tag your cells to remove using the words below!
    cfg.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    cfg.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
    cfg.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
    cfg.TagRemovePreprocessor.enabled = True

    # Configure and run out exporter
    cfg.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor"
    ]

    # Configure and run out exporter - returns a tuple - first element with
    # html, second with notebook metadata
    output = HTMLExporter(config=cfg).from_filename(nb_path)

    # Write to output html file
    with open(html_path,  "w", encoding="utf-8") as f:
        f.write(output[0])


def run_notebooks(path=NOTEBOOK_DIR):
    """Run all notebooks in the given directory

    Parameters
    ----------
    path: str (optional)
        path to a directory containing one or more Jupyter Notebooks

    Returns
    -------
    None
    """
    for fp in sorted(glob.iglob(os.path.join(path, "*.ipynb"))):
        print(f"Running {fp}")
        run_notebook(fp)


def clear_notebooks(path=NOTEBOOK_DIR):
    """Clears output from all notebooks on path

    Parameters
    ----------
    path: str (optional)
        path to a directory containing one or more Jupyter Notebooks

    Returns
    -------
    None
    """
    for fp in sorted(glob.iglob(os.path.join(path, "*.ipynb"))):
        print(f"Clearing {fp}")
        clear_notebook(fp)


def _get_path(path):
    """Verifies path to file, checking NOTEBOOK_DIR if not found

    Parameters
    ----------
    path: str
        path to a Jupyter Notebook from either the working directory
        or NOTEBOOK_DIR

    Returns
    -------
    str
        path to the Jupyter Notebook
    """
    try:
        open(path)
    except OSError as err:
        path = os.path.join(NOTEBOOK_DIR, path)
        try:
            open(path)
        except OSError:
            raise err
    return path
