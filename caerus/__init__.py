from caerus.caerus import (
    fit,
    gridsearch,
	makefig,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
#__version__ = '0.1.0'

# Automatic version control
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# module level doc-string
__doc__ = """
caerus - a powerful detection analysis library for local minima and maxima
=====================================================================

**caerus** 
This package determines the local-minima with the corresponding local-maxima 
across time series data without the need of normalization procedures. 
See README.md file for more information.


Main Features
-------------
Here are just a few of the things that caerus does well:
  
  - Input data is a simple vector of values for which the order matters.
  - Ouput contains detected start-stop regions of local minima and maxima.
  - Output figures are created.
  - Gridsearch is possible
"""
