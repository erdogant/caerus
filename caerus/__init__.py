import logging
from caerus.caerus import caerus
import caerus.helper as helper

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.1'

# Setup package-level logger
_logger = logging.getLogger('caerus')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {message}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)

if not _logger.hasHandlers():  # avoid duplicate handlers if re-imported
    _logger.addHandler(_log_handler)

_logger.setLevel(logging.DEBUG)
_logger.propagate = True  # allow submodules to inherit this handler



# module level doc-string
__doc__ = """
caerus
=====================================================================

caerus is a powerful detection analysis library for local minima and maxima.
This package determines the local-minima with the corresponding local-maxima across time series data without the need of normalization procedures.
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
>>> results = cs.fit(X)
>>> fig = cs.plot()
>>> # Results in the dataframe
>>> results['df']

References
----------
* https://github.com/erdogant/caerus
* https://github.com/erdogant/findpeaks

"""