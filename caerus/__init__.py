from caerus.caerus import (
    fit,
    gridsearch,
	makefig,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
caerus - a powerful detection analysis library for local minima and maxima
=====================================================================

**caerus** is a Python package providing that determines the local-minima with 
the corresponding local-maxima within the given time-frame. The method is build
using a forward rolling window to iteratively evaluate thousands of windows. 
For each window a score of percentages is computed from the start-to-stop 
position. The resulting matrix is a [window x length dataframe] for which only 
the high scoring percentages, e.g. those above a certain value (minperc) are 
used. The best scoring percentages is then aggregated by sum per time-point 
followed by a cut using the threshold. The resulting regions are subsequently 
detected, and represent the starting-locations of the trade. The stop-locations 
are determined based on the distance and percentage of te start-locations.
As an example, if you want to have best regions, use threshold=1, minperc=high 
and nlargest=1 (small).


Main Features
-------------
Here are just a few of the things that caerus does well:
  
  - Input data is a simple vector of values for which the order matters.
  - Ouput contains detected start-stop regions of local minima and maxima.
  - Output figures are created.
  - Gridsearch
"""