import os
import sys


def setup_imports_nb():
    try:
        # In case you run this cell from a .py file later via an IDE that defines __file__
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined in a standard Jupyter notebook cell, so use os.getcwd()
        # os.getcwd() usually points to the directory where the notebook file is located,
        # or where the Jupyter server was started.
        # Be sure this is the correct directory for your structure.
        notebook_parent_dir = os.getcwd()

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_forecasting_eval_utils"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

