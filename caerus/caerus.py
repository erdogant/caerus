"""References.

Name : caerus.py
Author : E.Taskesen
Contact : erdogant@gmail.com
Date : May 2020
"""

# %% Libraries
import caerus.utils.csutils as csutils
import caerus.utils.csplots as csplots
import pandas as pd
import numpy as np
from tqdm import tqdm
# Custom helpers
from caerus.utils.ones2idx import region2idx
from caerus.utils.risk_performance_metrics import risk_performance_metrics
import wget
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'


# Class
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
        verbose : Int, [0..5]. Default : 3
            The higher the number, the more information is printed.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5 : TRACE

        """
        if window is not None: self.window=window
        if minperc is not None: self.minperc=minperc
        if threshold is not None: self.threshold=threshold
        if nlargest is not None: self.nlargest=nlargest
        if verbose>=2 and self.window>X.shape[0]:
            print('[caerus] >Window size (%.0d) is larger then number of datapoints (%.0d). Max window size is set to [%.0d]' %(self.window, X.shape[0], X.shape[0]))
            self.window = np.minimum(self.window, X.shape[0])

        # Check inputs
        X = csutils._check_input(X)
        # Run over all windows
        simmat = csutils._compute_region_scores(X, window=self.window, verbose=verbose)
        # Keep only percentages above minimum
        simmat_hot = simmat[simmat>self.minperc]
        # Find local minima-start-locations
        [loc_start, outagg] = csutils._regions_detect_start(simmat_hot, self.minperc, self.threshold, extb=self.extb, extf=self.extf)
        # Find regions that are local optima for the corrersponding local-minima
        loc_stop = csutils._regions_detect_stop(simmat_hot, loc_start, self.nlargest, extb=self.extb, extf=self.extf, verbose=verbose)

        # If nothing is detected, try to explain
        if loc_start is None:
            loc_start_best = None
            loc_stop_best = None
            if verbose>=3: print('[caerus] >No regions detected with current paramters.')
            percok = simmat_hot.isna().sum().sum() / (simmat_hot.shape[0] * simmat_hot.shape[1])
            plt.hist(simmat.values.flatten(), bins=50)
            plt.grid(True)
            plt.xlabel('Percentage difference (be inspired to set minperc)')
            plt.ylabel('Frequency')
            plt.title('Perctange difference distribution')
            if verbose>=2: print('[caerus] >Warning >[%.0f%%] does not reach the minimum of %.1f%% difference. Tip: lower "minperc"' %(percok * 100, self.minperc))
        else:
            # Find regions that are local optima for the corrersponding local-minima
            [loc_start_best, loc_stop_best] = csutils._get_locs_best(X, loc_start, loc_stop)

        # Store
        results = {}
        results['X'] = X
        results['simmat'] = simmat
        results['loc_start'] = loc_start
        results['loc_stop'] = loc_stop
        results['loc_start_best'] = loc_start_best
        results['loc_stop_best'] = loc_stop_best
        results['agg'] = outagg
        results['df'] = csutils.to_df(results)
        # Store in self
        self.results = results
        if return_as_dict:
            return results

    # Perform gridsearch to determine best parameters
    def gridsearch(self, X, window=np.arange(50, 550, 50), minperc=np.arange(1, 20, 1), threshold=0.25, return_as_dict=False, verbose=3):
        """Gridsearch to find best fit.

        Parameters
        ----------
        X : array-like : 1D array.
            Input 1D vector.
        verbose : Int, [0..5]. Default : 3
            The higher the number, the more information is printed.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5 : TRACE

        Returns
        -------
        Object containing dict with key ['gridsearch'] such as cs.gridsearch
        balances : np-array
            results of balances across various levels of: window x minperc
        trades : np-array
            results of trades across various levels of: window x minperc

        """
        if verbose>=3: print('[caerus] Gridsearch..')
        # Check inputs
        X = csutils._check_input(X)
        verbose_tqdm = (True if (verbose==0) else False)

        # Start gridsearch
        out_balance = np.zeros((len(minperc), len(window)))
        out_trades = np.zeros((len(minperc), len(window)))
        # Run
        for k in tqdm(range(0, len(window)), disable=verbose_tqdm):
            for i in range(0, len(minperc)):
                # Compute start-stop locations
                self.fit(X, window=window[k], minperc=minperc[i], threshold=threshold, nlargest=1, return_as_dict=False, verbose=0)
                # Store
                perf=pd.DataFrame()
                perf['portfolio_value'] = X.values.copy()
                perf['asset'] = X.values.copy()
                perf['invested'] = 0
                perf['invested'].iloc[region2idx(np.vstack((self.results['loc_start_best'], self.results['loc_stop_best'])).T)]=1
                performanceMetrics = risk_performance_metrics(perf)
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
    def plot(self, threshold=None, figsize=(25, 15)):
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
        if hasattr(self, 'results_gridsearch'):
            csplots._plot_gridsearch(self.results_gridsearch, figsize=figsize)
        if hasattr(self, 'results'):
            csplots._plot(self.results, threshold=self.threshold, figsize=figsize)

    #  Import example dataset from github.
    def download_example(self, name='btc', verbose=3):
        """Import example dataset from github.

        Parameters
        ----------
        name : str, optional
            name of the file to download.
        verbose : int, optional
            Print message to screen. The default is 3.

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
        PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))

        # Create dir
        if not os.path.isdir(curpath):
            os.makedirs(curpath, exist_ok=True)

        # Check file exists.
        if not os.path.isfile(PATH_TO_DATA):
            if verbose>=3: print('[caerus] Downloading example dataset..')
            wget.download(url, curpath)

        # Import local dataset
        if verbose>=3: print('[caerus] Import dataset..')
        df = pd.read_csv(PATH_TO_DATA)
        # Return
        return df[getfeat].values
