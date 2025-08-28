"""References.

Name : caerus.py
Author : E.Taskesen
Contact : erdogant@gmail.com
Date : May 2020
"""

# %% Libraries

# Custom helpers
try:
    import caerus.helper as helper
except:
    import helper
    
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
import requests

import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)
# if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {message}', style='{', datefmt='%d-%m-%Y %H:%M:%S')


# %%
class caerus():
    """Compute the local minima with the corresponding local-maxima within the given time-frame.

    Description
    -----------
    In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments.
    He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

    **caerus** is a python package providing that determines the local-minima with
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

    Here are just a few of the things that caerus does well:
        - Ouput contains detected start-stop regions of local minima and maxima.
        - Figures are created.
        - Parameter gridsearch.
        - Designed for the detection of complex trend movements.

    Parameters
    ----------
    window : int [1,..,len(X)], default : 50
        Window size that is used to determine whether there is an increase in percentage. start location + window.
        50 : (default) Smaller window size is able to pickup better local-minima.
        1000 : Larger window size will more stress on global minma.
    minperc : float [0,..,100], default : 3
        Minimum percentage to declare a starting position with window relevant.
        Note that nlargest is used to identify the top n largest percentages as stopping location.
    nlargest : float [1,..,inf], default : 10
        Identify the top n percentages, and thus stop-regions (couples to start-region).
        The larger this number, the larger the stop-region untill it is limited by minperc.
    threshold : float [0,..,1], default : 0.25
        Required to optimize for the maximum depth of the local-minima.
        At the ith location, k windows (eg 50) are overlaid and the percentages are determined.
        The socre is determined by (percentage(i-start,k-stop)) >= minperc (eg 3), and normalized for the maximum number of windows used at position i.
        In best case scenarion, all window result in percentage>minperc and will have score 50/50=1.
    return_as_dict : Bool (default : True)
        Return results in a dictionary.

    Examples
    --------
    >>> from caerus import caerus
    >>> cs = caerus()
    >>> X = cs.download_example()
    >>> cs.fit(X)
    >>> cs.plot()
    """

    def __init__(self, window=50, minperc=3, nlargest=10, threshold=0.25, extb=0, extf=10):
        """Initialize distfit with user-defined parameters."""
        self.window = window
        self.minperc = minperc
        self.nlargest = nlargest
        self.threshold = threshold
        self.extb = extb
        self.extf = extf

    def fit(self, X, window=None, minperc=None, threshold=None, nlargest=None, return_as_dict=True, verbose=3):
        """Detect optimal optima and minima.

        Parameters
        ----------
        X : array-like : 1D array.
            Data such as stock prices in a 1D vector.
        verbose : Int, [0..5]. The higher the number, the more information is printed.
            0: None,  1: ERROR,  2: WARN,  3: INFO (default),  4: DEBUG, 5 : TRACE

        Raises
        ------
        Exception
            1D array should be of type 1D numpy array or list.

        Returns
        -------
        Object.
        X : array-like : 1D array.
            Input 1D vector.
        simmat : np.array
            Simmilarity matrix
        loc_start : list of int
            list of indexes containing start positions
        loc_stop : list of int
            list of indexes containing stop positions
        loc_start_best : list of int
            list of indexes containing the best starting positions
        loc_stop_best : list of int
            list of indexes containing the best stopping positions
        agg : 1D array-like
            Aggregated 1D array
        df : pd.DataFrame
            Results in the form of a dataframe.

        """
        if window is not None: self.window=window
        if minperc is not None: self.minperc=minperc
        if threshold is not None: self.threshold=threshold
        if nlargest is not None: self.nlargest=nlargest
        
        # Handle verbose logging - use instance verbose if not overridden
        current_verbose = verbose if verbose is not None else self.verbose
        verbose_previous = helper.get_logger()
        helper.set_logger(current_verbose)

        # Check inputs
        X = helper._check_input(X)

        logger.info(f'Start run with window: {self.window}, minperc: {self.minperc}, nlargest: {self.nlargest} and threshold: {self.threshold}')
        if self.window>X.shape[0]:
            logger.warning(f'Window size {self.window} is larger then number of datapoints {X.shape[0]}. Max window size is set to {X.shape[0]}')
            self.window = np.minimum(self.window, X.shape[0])

        # Run over all windows
        simmat = helper._compute_region_scores(X, window=self.window)
        # Keep only percentages above minimum
        simmat_hot = simmat[simmat>self.minperc]
        # Find local minima-start-locations
        loc_start, outagg = helper._regions_detect_start(simmat_hot, self.minperc, self.threshold, extb=self.extb, extf=self.extf)
        # Find regions that are local optima for the corrersponding local-minima
        loc_stop = helper._regions_detect_stop(simmat_hot, loc_start, self.nlargest, extb=self.extb, extf=self.extf)

        # If nothing is detected, try to explain
        if loc_start is None:
            loc_start_best = None
            loc_stop_best = None
            logger.info(f'No regions detected with current parameters.')
            percok = simmat_hot.isna().sum().sum() / (simmat_hot.shape[0] * simmat_hot.shape[1])
            plt.hist(simmat.values.flatten(), bins=50)
            plt.grid(True)
            plt.xlabel('Percentage difference (be inspired to set minperc)')
            plt.ylabel('Frequency')
            plt.title('Perctange difference distribution')
            logger.warning(f'Percentage {percok * 100} does not reach the minimum of {self.minperc} difference. Tip: lower "minperc"')
        else:
            # Find regions that are local optima for the corrersponding local-minima
            loc_start_best, loc_stop_best = helper._get_locs_best(X, loc_start, loc_stop)

        # Store
        results = {}
        results['X'] = X
        results['simmat'] = simmat
        results['loc_start'] = loc_start
        results['loc_stop'] = loc_stop
        results['loc_start_best'] = loc_start_best
        results['loc_stop_best'] = loc_stop_best
        results['agg'] = outagg
        results['df'] = helper.to_df(results)
        
        # Restore logger
        if verbose_previous is not None:
            helper.set_logger(verbose_previous)

        # Store in self
        self.results = results
        if return_as_dict:
            return results

    # Perform gridsearch to determine best parameters
    def gridsearch(self, X, window=np.arange(50, 550, 50), minperc=np.arange(1, 20, 1), threshold=0.25, return_as_dict=False):
        """Gridsearch to find best fit.

        Parameters
        ----------
        X : array-like : 1D array.
            Input 1D vector.

        Returns
        -------
        Object containing dict with key ['gridsearch'] such as cs.gridsearch
        balances : np-array
            results of balances across various levels of: window x minperc
        trades : np-array
            results of trades across various levels of: window x minperc

        """
        logger.info('Gridsearch..')
        # Check inputs
        X = helper._check_input(X)

        # Start gridsearch
        out_balance = np.zeros((len(minperc), len(window)))
        out_trades = np.zeros((len(minperc), len(window)))

        # Run
        for k in tqdm(range(0, len(window)), desc='[caerus] Gridsearch..', disable=helper.disable_tqdm()):
            for i in range(0, len(minperc)):
                # Compute start-stop locations - use instance verbose level
                self.fit(X, window=window[k], minperc=minperc[i], threshold=threshold, nlargest=1, return_as_dict=False, verbose=self.verbose)
                # Store
                perf=pd.DataFrame()
                perf['portfolio_value'] = X.values.copy()
                perf['asset'] = X.values.copy()
                perf['invested'] = 0
                perf['invested'].iloc[helper.region2idx(np.vstack((self.results['loc_start_best'], self.results['loc_stop_best'])).T)]=1
                performanceMetrics = helper.risk_performance_metrics(perf)
                # Compute score
                out_balance[i, k] = performanceMetrics['winning_balance']
                out_trades[i, k] = performanceMetrics['winning_trades']

        # Store
        results = {}
        results['balances'] = out_balance
        results['trades'] = out_trades
        results['window'] = window
        results['minperc'] = minperc
        self.results_gridsearch=results
        if return_as_dict:
            return results

    # Make final figure
    def plot(self, threshold=None, figsize=(25, 15), visible=True):
        """Plot results.

        Parameters
        ----------
        threshold : float [0,..,1], default : 0.25
            Required to optimize for the maximum depth of the local-minima.
            At the ith location, k windows (eg 50) are overlaid and the percentages are determined.
        figsize : tuple, optional
            Figure size. The default is (25,15).

        Returns
        -------
        None.

        """
        fig, axs = None, None
        if hasattr(self, 'results_gridsearch'):
            fig, axs = helper._plot_gridsearch(self.results_gridsearch, figsize=figsize, visible=visible)
        if hasattr(self, 'results'):
            fig, axs = helper._plot(self.results, threshold=self.threshold, figsize=figsize, visible=visible)
        # Return
        return fig, axs

    def set_verbose(self, verbose):
        """Set the verbose level for this instance.
        
        Parameters
        ----------
        verbose : str or int
            Verbose level ('silent', 'critical', 'warning', 'info', 'debug')
        """
        self.verbose = verbose
        helper.set_logger(verbose=verbose)
    
    def get_verbose(self):
        """Get the current verbose level for this instance.
        
        Returns
        -------
        str or int
            Current verbose level
        """
        return self.verbose

    #  Import example dataset from github.
    def download_example(self, name='btc', overwrite=True):
        """Import example dataset from github.

        Parameters
        ----------
        name : str, optional
            name of the file to download.

        Returns
        -------
        tuple containing dataset and response variable (X,y).

        """
        if name=='facebook':
            url='https://erdogant.github.io/datasets/facebook_stocks.zip'
            getfeat='close'
        else:
            url='https://erdogant.github.io/datasets/BTCUSDT.zip'
            getfeat='Close'

        curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        filename = wget.filename_from_url(url)
        PATH_TO_DATA = os.path.join(curpath, filename)

        # Create dir
        if not os.path.isdir(curpath):
            os.makedirs(curpath, exist_ok=True)

        # Check file exists.
        if not os.path.isfile(PATH_TO_DATA) or overwrite:
            logger.info('Downloading example dataset..')
            df = wget.download(url, PATH_TO_DATA)

        # Import local dataset
        logger.info('Import dataset..')
        df = pd.read_csv(PATH_TO_DATA)
        # Return
        return df[getfeat].values


# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url, ext=True):
        """Return filename."""
        urlname = os.path.basename(url)
        if not ext: _, ext = os.path.splitext(urlname)
        return urlname

    def download(url, writepath):
        """Download.

        Parameters
        ----------
        url : str.
            Internet source.
        writepath : str.
            Directory to write the file.

        Returns
        -------
        None.

        """
        r = requests.get(url, stream=True)
        with open(writepath, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

