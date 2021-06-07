# Defines functions for clearing and running notebooks
import glob
import os
import shutil

import nbformat
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import (
    ClearOutputPreprocessor,
    ExecutePreprocessor,
    TagRemovePreprocessor
)
from traitlets.config import Config

from .config import PROJ_DIR




NOTEBOOK_DIR = os.path.join(PROJ_DIR, "notebooks")




def run_notebook(path):
    """Runs a notebook and saves the output"""

    with open(_get_path(path)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor()
        ep.preprocess(nb, {})

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

        
def notebook_to_html(nb_path, html_path, run=False):
    """Converts notebook to HTML and strips input, output, and code blocks
    
    Source: https://github.com/jupyter/nbconvert/issues/1194#issuecomment-839360322
    """
    
    
    nb_path = _get_path(nb_path)

    # Run the notebook first if specified
    if run:
        run_notebook(nb_path)

    # Setup config
    c = Config()

    # Configure tag removal - be sure to tag your cells to remove using the words below!
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
    c.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
    c.TagRemovePreprocessor.enabled = True

    # Configure and run out exporter
    c.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor"
    ]

    # FIXME: Not needed? The original post doesn't use the exporter variable
    #exporter = HTMLExporter(config=c)
    #exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

    # Configure and run out exporter - returns a tuple - first element with
    # html, second with notebook metadata
    output = HTMLExporter(config=c).from_filename(nb_path)

    # Write to output html file
    with open(html_path,  "w", encoding="utf-8") as f:
        f.write(output[0])


def clear_notebook(path):
    """Clears output from a notebook"""

    with open(_get_path(path)) as f:
        nb = nbformat.read(f, as_version=4)
        pp = ClearOutputPreprocessor()
        pp.preprocess(nb, {})

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

        
def clear_notebooks(path=NOTEBOOK_DIR):
    """Clears output from all notebooks on path"""
    for fp in glob.iglob(os.path.join(path, "*.ipynb")):
        clear_notebook(fp)
                           
                           
def _get_path(path):
    """Verifies path to file, checking NOTEBOOK_DIR if not found"""
    try:
        open(path)
    except OSError as err:
        path = os.path.join(NOTEBOOK_DIR, path)
        try:
            open(path)
        except OSError:
            raise err        
    return path
                           