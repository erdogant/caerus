#--------------------------------------------------------------------------
# Name        : caerus.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : May. 2020
#--------------------------------------------------------------------------

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
        In best case scenarion, all window result in percentage>minperc and will hve score 50/50=1. 

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

    # main - Detection of optimal localities for investments
    def fit(self, X, verbose=3):
    
        # Convert to dataframe
        if isinstance(X, pd.DataFrame): raise Exception('[caerus] >Error: Input data must be of type numpy-array or list.')
        if 'numpy' in str(type(X)) or 'list' in str(type(X)): X = pd.Series(X)
        if X.shape[0]!=X.size: raise Exception('[caerus] >Error : Input dataframe can only be a 1D-vector.')

        # reset index
        X.reset_index(drop=True, inplace=True)
        # Run over all windows
        simmat = compute_region_scores(X, window=self.window, verbose=verbose)
        # Keep only percentages above minimum
        simmat = simmat[simmat>self.minperc]
        # Find local minima-start-locations
        [loc_start, outagg] = regions_detect_start(simmat, self.minperc, self.threshold, extb=self.extb, extf=self.extf)
        # Find regions that are local optima for the corrersponding local-minima
        loc_stop = regions_detect_stop(simmat, loc_start, self.nlargest, extb=self.extb, extf=self.extf, verbose=verbose)
        # Find regions that are local optima for the corrersponding local-minima
        [loc_start_best, loc_stop_best] = get_locs_best(X, loc_start, loc_stop)
    
        self.X = X
        self.simmat=simmat
        self.loc_start=loc_start
        self.loc_stop=loc_stop
        self.loc_start_best=loc_start_best
        self.loc_stop_best=loc_stop_best
        self.agg=outagg

    # Make final figure
    def plot(self, threshold=None, figsize=[25,15]):
        df = self.X
        loc_start = self.loc_start
        loc_stop = self.loc_stop
        loc_start_best = self.loc_start_best
        loc_stop_best = self.loc_stop_best
        simmat = self.simmat
        if threshold is None:
            threshold = self.threshold
            
        # agg = out['agg']
        
        [fig,(ax1,ax2,ax3)]=plt.subplots(3,1, figsize=figsize)
        # Make heatmap
        ax1.matshow(np.flipud(simmat.T))
        ax1.set_aspect('auto')
        # ax1.gca().set_aspect('auto')
        ax1.grid(False)
        ax1.set_ylabel('Perc.difference in window\n(higher=better)')
        ax1.set_xlim(0,simmat.shape[0])
        
        xlabels = simmat.columns.values.astype(str)
        I=np.mod(simmat.columns.values,10)==0
        xlabels[I==False]=''
        xlabels[-1]=simmat.columns.values[-1].astype(str)
        xlabels[0]=simmat.columns.values[0].astype(str)
        ax1.set_yticks(range(0,len(xlabels)))
        ax1.set_yticklabels(np.flipud(xlabels))
        ax1.grid(True, axis='x')
        
        # make aggregated figure
        # Normalized successes across the n windows for percentages above minperc.
        # 1 depicts that for location i, all of the 1000 windows of different length was succesfull in computing a percentage above minperc
        [outagg, I] = agg_scores(simmat, threshold)
        ax2.plot(outagg)
        ax2.grid(True)
        ax2.set_ylabel('Cummulative\n#success windows')
        ax2.set_xlim(0,simmat.shape[0])
        ax2.hlines(threshold,0,simmat.shape[0], linestyles='--',  colors='r')
        ax2.vlines(loc_start_best,0,1, linestyles='--',  colors='g')
        ax2.vlines(loc_stop_best,0,1, linestyles='--',  colors='r')
    
        # Plot local minima-maxima
        ax3.plot(df.iloc[loc_start_best],'og', linewidth=1)
        ax3.plot(df.iloc[loc_stop_best],'or', linewidth=1)
    
        # Plot region-minima-maxima
        ax3.plot(df,'k', linewidth=1)
        for i in range(0,len(loc_start)):
            ax3.plot(df.iloc[np.arange(loc_start[i][0],loc_start[i][1])],'g', linewidth=2)
            # ax3.plot(df.iloc[np.arange(loc_stop[i][0],loc_stop[i][1])],'r', linewidth=2)
            # ax3.plot(df.iloc[loc_stop[i]], 'or', linewidth=2)
            for k in range(0,len(loc_stop[i])):
                ax3.plot(df.iloc[np.arange(loc_stop[i][k][0],loc_stop[i][k][1])],'r', linewidth=2)
        
        ax3.set_ylabel('Input value')
        ax3.set_xlabel('Time')
        ax3.grid(True)
        ax3.set_xlim(0,simmat.shape[0])
        ax3.vlines(loc_start_best,df.min(),df.max(), linestyles='--',  colors='g')
        ax3.vlines(loc_stop_best,df.min(),df.max(), linestyles='--',  colors='r')
    
        plt.show()
    
        return(fig)

    #  Import example dataset from github.
    def download_example(name='btc', verbose=3):
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
        if name=='btc':
            url='https://erdogant.github.io/datasets/BTCUSDT.zip'
        else:
            url='https://erdogant.github.io/datasets/facebook_stocks.zip'
    
        curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    
        # Check file exists.
        if not os.path.isfile(PATH_TO_DATA):
            if verbose>=3: print('[classeval] Downloading example dataset..')
            wget.download(url, curpath)
    
        # Import local dataset
        if verbose>=3: print('[caerus] Import dataset..')
        df = pd.read_csv(PATH_TO_DATA)
    
        # Return
        return df
    

#%% Perform gridsearch to determine best parameters
def gridsearch(df, window=None, perc=None, threshold=0.25, showplot=True, verbose=3):
    if verbose>=3: print('[CAERUS] Gridsearch..')

    if isinstance(window, type(None)):
        windows=np.arange(50,550,50)
    if isinstance(perc, type(None)):
        perc=np.arange(0,1,0.1)
    if showplot:
        [fig,(ax1,ax2)]=plt.subplots(2,1)

    out_balance = np.zeros((len(perc),len(windows)))
    out_trades  = np.zeros((len(perc),len(windows)))

    for k in tqdm(range(0,len(windows)), disable=(True if verbose==0 else False)):
        for i in range(0,len(perc)):
            # Compute start-stop locations
            getregions=fit(df, window=windows[k], minperc=perc[i], threshold=threshold, nlargest=1, verbose=0)
            # Store
            perf=pd.DataFrame()
            perf['portfolio_value'] = df.values.copy()
            perf['asset']           = df.values.copy()
            perf['invested']        = 0
            perf['invested'].iloc[region2idx(np.vstack((getregions['loc_start_best'], getregions['loc_stop_best'])).T)]=1
            performanceMetrics = risk_performance_metrics(perf)
            # Compute score
            out_balance[i,k] =performanceMetrics['winning_balance']
            out_trades[i,k]=performanceMetrics['winning_trades']

        if showplot:
            #label = list(map(( lambda x: 'window_' + x), windows.astype(str)))
            ax1.plot(perc,out_balance[:,k], label='window_'+str(windows[k]))
            ax2.plot(perc,out_trades[:,k], label='window_'+str(windows[k]))


    if showplot:
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('Percentage')
        ax1.set_ylabel('winning_balance')
        ax2.grid(True)
        ax2.set_xlabel('Percentage')
        ax2.set_ylabel('Nr Trades')
        plt.show()
    
    out_balance  = pd.DataFrame(index=perc, data=out_balance, columns=windows)
    out_trades   = pd.DataFrame(index=perc, data=out_trades, columns=windows)
    return(out_balance, out_trades)

#%% Merge regions
def get_locs_best(df, loc_start, loc_stop):
    loc_start_best=np.zeros(len(loc_start)).astype(int)
    loc_stop_best=np.zeros(len(loc_start)).astype(int)
    for i in range(0,len(loc_start)):
        loc_start_best[i]=df.iloc[loc_start[i][0]:loc_start[i][1]+1].idxmin()

        tmpvalue=pd.DataFrame()
        for k in range(0,len(loc_stop[i])):
            idx_start=np.minimum(loc_stop[i][k][0], df.shape[0]-1)
            idx_stop=np.minimum(loc_stop[i][k][1]+1, df.shape[0])
            tmpvalue = pd.concat((tmpvalue, df.iloc[idx_start:idx_stop]))

        loc_stop_best[i]=tmpvalue.idxmax()[0]
    return(loc_start_best, loc_stop_best)
    
#%% Merge regions
def regions_merge(data, extb=5, extf=5):
    out=None
    if not isinstance(data,type(None)):
        data    = np.array(data)
        idx     = np.argsort(data[:,0])
        data    = data[idx,:]
        loc_bin = np.zeros(np.max(data)+1)

        # Add ones to array
        if data.shape[0]==1:
            loc_bin[np.arange(data[0][0], data[0][1]+1)]=1
        else:
            for i in range(0,len(data)):
                if i<len(data)-1:
                    # Check whether in each others range
                    if data[i][1]+extf>=data[i+1][0]-extb:
                        XtraOnes=np.arange(data[i][1], data[i+1][0])
                        # Add ones to array
                        loc_bin[XtraOnes]=1
                
                # Add ones to array
                loc_bin[np.arange(data[i][0], np.minimum(data[i][1]+1, len(loc_bin)))]=1
            
        # Find the merged indexes
        out = ones2region(loc_bin)
    return(out)

#%% Compute scores using a forward rolling window
def compute_region_scores(df, window=1000, verbose=3):
    # Compute percentage score for each 
    # 1. Window
    # 2. Position
    
    # Start with empty dataframe
    out=pd.DataFrame()

    # Reverse dataframe to create forward-rolling window
    df=df[::-1]
    for i in tqdm(range(2,window), disable=(True if verbose==0 else False)):
        dfperc = df.rolling(i).apply(compute_percentage, raw=True)[::-1] #.values.flatten()
        out[i]=dfperc
    
    out[np.isinf(out)]=np.nan
    # out.fillna(value=0, inplace=True)

    return(out)

#%% Aggregation of scores over the windows and intervals
def agg_scores(out, threshold=0):
    outagg=np.nansum(out>0, axis=1)
    # Normalize for the window size that is used. Towards the end smaller windows are only avaialbe which is otherwise unfair for the threshold usage.
    windowCorrectionFactor=np.ones_like(outagg)*out.shape[1]
    tmpvalue=np.arange(1,out.shape[1])[::-1]
    windowCorrectionFactor[-len(tmpvalue):]=tmpvalue

    outagg = outagg/windowCorrectionFactor
    I=outagg>threshold
    return(outagg, I)
    
#%% Detect starting positions for regions
def regions_detect_start(out, minperc, threshold, extb=5, extf=5):
    # Find start-locations
    [outagg, I] = agg_scores(out, threshold)
    locs_start=ones2region(I)
    
    if len(locs_start)==0:
        locs_start=None

    # Merge regions if only seperated with few intervals
    locs_start = regions_merge(locs_start, extb=extb, extf=extf)

    return(locs_start, outagg)

#%% Detect stop locations based on the starting positions
def regions_detect_stop(out, locs_start, nlargest, extb=5, extf=5, verbose=0):
    # Find stop-locations
    locs_stop=None
    if not isinstance(locs_start,type(None)):

        locs_stop=[]
#        out[np.isinf(out)]=np.nan
        
        for i in range(0,len(locs_start)):
            if verbose>=4: print('[CAERUS] Region %s' %(i))
            # Take window sizes with maximum percentages
            # getloc=out.iloc[locs_start[i][0]:locs_start[i][1]+1,:].idxmax(axis=1)

            # Get window size and add to starting indexes
            startlocs= np.arange(locs_start[i][0], locs_start[i][1]+1)
            
            getloc=[]
            getpos=out.iloc[locs_start[i][0]:locs_start[i][1]+1,:]
            
            # Run over all positions to find the top-n maximum ones
            for k in range(0,getpos.shape[0]):
                tmplocs = getpos.iloc[k,:].nlargest(nlargest).index.values
                tmplocs = tmplocs+startlocs[k]
                getloc=np.append(np.unique(getloc), tmplocs)
            
            getloc = np.sort(np.unique(getloc)).astype(int)

            # Merge if required
            getloc=idx2region(getloc)
            getloc=regions_merge(getloc, extb=extb, extf=extf)

            # Compute mean percentages per region and sort accordingly
            loc_mean_percentage=[]
            for p in range(0,len(getloc)):
                loc_mean_percentage.append(np.nanmean(out.iloc[getloc[p][0]:getloc[p][1]+1,:]))
            loc_mean_percentage=np.array(loc_mean_percentage)
            idx=np.argsort(loc_mean_percentage)[::-1]
            getloc=np.array(getloc)[idx]
            
            locs_stop.append(getloc.tolist())

    return(locs_stop)

#%% Compute percentage
def compute_percentage(r):
    perc=percentage_getdiff(r[0],r[-1])
    return(perc) 


#%% Compute percentage between current price and starting price
def percentage_getdiff(current_price, previous_price):
    assert isinstance(current_price, float)
    assert isinstance(previous_price, float)

    if current_price>previous_price:
        # Increase
        diff_perc=(current_price-previous_price)/previous_price*100
    else:
        # Decrease
        diff_perc=-(previous_price-current_price)/previous_price*100
    
    return(diff_perc)


