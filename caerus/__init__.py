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
__version__ = '0.1.2'

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


Examples
--------
>>> from caerus import caerus
>>> cs = caerus()
>>> X = cs.download_example()
>>> cs.fit(X)
>>> fig = cs.plot()

References
----------
8 https://github.com/erdogant/caerus
* https://github.com/erdogant/findpeaks

"""
