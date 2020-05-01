from caerus.caerus import (
    caerus,
    )
# from caerus.caerus import (
#     fit,
#     gridsearch,
# 	makefig,
    # download_example,
# )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
caerus
=====================================================================

Description
-----------
caerus is a powerful detection analysis library for local minima and maxima.
This package determines the local-minima with the corresponding local-maxima 
across time series data without the need of normalization procedures. 
See README.md file for more information.

Here are just a few of the things that caerus does well:
  
- Input data is a simple vector of values for which the order matters.
- Ouput contains detected start-stop regions of local minima and maxima.
- Output figures are created.
- Gridsearch is possible


Example
-------
>>> import caerus as caerus
>>> model = caerus.fit(X)
>>> fig,ax = caerus.plot(model)

References
----------
https://github.com/erdogant/caerus

"""
