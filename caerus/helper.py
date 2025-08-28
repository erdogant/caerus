"""Helper file for Caerus"""
"""This code is part of caerus and is not designed for usage in other code."""
#--------------------------------------------------------------------------
# Name        : caerus.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : May. 2020
#--------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)
# if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {message}', style='{', datefmt='%d-%m-%Y %H:%M:%S')

# %%
# =============================================================================
# ONES2IDX
# =============================================================================

# Convert to index
def ones2region(data, value=1):
    Param = {}
    Param['value'] = value
    
    # Convert to one hot array
    data=(data==Param['value']).astype(int)
    
    # Append zero to begin and end to include starting and stopping ones
    data=np.append(data,0)
    data=np.append(0,data)
    
    # Determine boundaries
    boundaries=np.diff(data,1)  
    start=np.where(boundaries==1)[0]
    stop=np.where(boundaries==-1)[0]-1
    idx=list(zip(start,stop))
        
    # END
    return(idx)

# Convert to index
def idx2region(data):
#    if np.min(data)==np.max(data):
#        idxbin=np.arange(np.min(data), np.max(data)+1)
#    else:
#        idxbin=np.arange(np.min(data), np.max(data))
    idxbin=np.arange(np.min(data), np.max(data)+1)

    locbin=np.isin(idxbin,data).astype(int)
    out=ones2region(locbin)
    for i in range(0,len(out)):
        out[i]=(idxbin[out[i][0]], idxbin[out[i][1]])
        
    return(out)

# Convert index to ones
def region2ones(data):
    out=np.zeros(np.max(data)+1).astype(int)
    for i in range(0,len(data)):
        out[np.arange(data[i][0],data[i][1]+1)]=1

    return(out)

# Convert index to ones
def region2idx(data):
    out=ones2idx(region2ones(data))
    return(out)
    
# Convert index to ones
def ones2idx(data):
    out = np.where(data)[0]
    return(out)

# %%
# =============================================================================
# UTILS
# =============================================================================

# utils
def _check_input(X):
    # Convert to dataframe
    if isinstance(X, pd.DataFrame):
        raise Exception(logger.error('Input data must be of type numpy-array or list.'))
    if 'numpy' in str(type(X)) or 'list' in str(type(X)):
        X = pd.Series(X)
    if X.shape[0] != X.size:
        raise Exception(logger.error('Input dataframe can only be a 1D-vector.'))

    X.reset_index(drop=True, inplace=True)
    # Return
    return X

# Aggregation of scores over the windows and intervals
def _agg_scores(out, threshold=0):
    outagg=np.nansum(out>0, axis=1)
    # Normalize for the window size that is used. Towards the end smaller windows are only avaialbe which is otherwise unfair for the threshold usage.
    windowCorrectionFactor = np.ones_like(outagg)*out.shape[1]
    tmpvalue = np.arange(1, out.shape[1])[::-1]

    if len(tmpvalue)>0:
        windowCorrectionFactor[-len(tmpvalue):] = tmpvalue

    outagg = outagg/windowCorrectionFactor
    I=outagg>threshold
    return(outagg, I)

# Merge regions
def _get_locs_best(df, loc_start, loc_stop):
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
    
# Merge regions
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


# Compute scores using a forward rolling window
def _compute_region_scores(df, window=1000):
    # Compute percentage score for each 
    # 1. Window
    # 2. Position
    
    # Start with empty dataframe
    out=pd.DataFrame()

    # Reverse dataframe to create forward-rolling window
    df=df[::-1]
    for i in tqdm(range(2,window), disable=disable_tqdm(), desc='[caerus] Compute region scores..'):
        dfperc = df.rolling(i).apply(_compute_percentage, raw=True)[::-1] #.values.flatten()
        out[i]=dfperc
    
    out[np.isinf(out)]=np.nan
    # out.fillna(value=0, inplace=True)

    return(out)

# Detect starting positions for regions
def _regions_detect_start(out, minperc, threshold, extb=5, extf=5):
    # Find start-locations
    [outagg, I] = _agg_scores(out, threshold)
    locs_start=ones2region(I)
    
    if len(locs_start)==0:
        locs_start=None

    # Merge regions if only seperated with few intervals
    locs_start = regions_merge(locs_start, extb=extb, extf=extf)

    return(locs_start, outagg)


# Detect stop locations based on the starting positions
def _regions_detect_stop(out, locs_start, nlargest, extb=5, extf=5):
    # Find stop-locations
    locs_stop=None
    if not isinstance(locs_start, type(None)):
        locs_stop=[]
        # out[np.isinf(out)]=np.nan

        # for i in range(0, len(locs_start)):
        for i, _ in enumerate(locs_start):
            logger.debug(f'Working on region {i}')
            # Take window sizes with maximum percentages
            # getloc=out.iloc[locs_start[i][0]:locs_start[i][1]+1,:].idxmax(axis=1)

            # Get window size and add to starting indexes
            startlocs= np.arange(locs_start[i][0], locs_start[i][1] + 1)

            getloc=[]
            getpos=out.iloc[locs_start[i][0]:locs_start[i][1] + 1, :]

            # Run over all positions to find the top-n maximum ones
            for k in range(0, getpos.shape[0]):
                tmplocs = getpos.iloc[k, :].nlargest(nlargest).dropna().index.values
                tmplocs = tmplocs + startlocs[k]
                getloc=np.append(np.unique(getloc), tmplocs)

            getloc = np.sort(np.unique(getloc)).astype(int)

            # Merge if required
            getloc=idx2region(getloc)
            getloc=regions_merge(getloc, extb=extb, extf=extf)

            # Compute mean percentages per region and sort accordingly
            loc_mean_percentage=[]
            for p in range(0, len(getloc)):
                xtmp = out.iloc[getloc[p][0]:getloc[p][1] + 1, :]
                meanPerc = np.nanmean(xtmp)
                loc_mean_percentage.append(meanPerc)
            loc_mean_percentage=np.array(loc_mean_percentage)
            idx=np.argsort(loc_mean_percentage)[::-1]
            getloc=np.array(getloc)[idx]

            locs_stop.append(getloc.tolist())

    return(locs_stop)

# Compute percentage
def _compute_percentage(r):
    perc=_percentage_getdiff(r[0], r[-1])
    return(perc)


# Compute percentage between current price and starting price
def _percentage_getdiff(current_price, previous_price):
    assert isinstance(current_price, float)
    assert isinstance(previous_price, float)

    if current_price>previous_price:
        # Increase
        diff_perc=(current_price - previous_price) / previous_price * 100
    else:
        # Decrease
        diff_perc=-(previous_price - current_price) / previous_price * 100

    return(diff_perc)


# Create labels
def to_df(results):
    """To Pandas DataFrame.

    Parameters
    ----------
    results : Output from the fit model.

    Returns
    -------
    df : Pandas DataFrame.

    """
    df = pd.DataFrame(data=results['X'], columns=['X'])
    df['labx']=0
    df['peak']=False
    df['valley']=False

    # Stop = peak
    if results['loc_stop_best'] is not None:
        df['peak'].iloc[results['loc_stop_best']]=True
    # Start = Valley
    if results['loc_start_best'] is not None:
        df['valley'].iloc[results['loc_start_best']]=True

        for i in range(0, len(results['loc_start'])):
            idx_valley = np.arange(results['loc_start'][i][0], results['loc_start'][i][1])
            df['labx'].iloc[idx_valley] = i + 1

            for k in range(0, len(results['loc_stop'][i])):
                df['labx'].iloc[np.arange(results['loc_stop'][i][k][0], results['loc_stop'][i][k][1])] = i + 1

    return df


# %%
# =============================================================================
# PLOTS
# =============================================================================
def _plot_graph(out, xlabel='Time', ylabel='Input value', figsize=(15,8)):
    df = out['df']
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['X'].loc[df['peak']], 'or', linewidth=1)
    ax.plot(df['X'].loc[df['valley']], 'og', linewidth=1)
    ax.plot(df['X'], 'k', linewidth=0.5)
    ax.vlines(np.where(df['peak'])[0], df['X'].min(), df['X'].max(), linestyles='--', colors='r')
    ax.vlines(np.where(df['valley'])[0], df['X'].min(), df['X'].max(), linestyles='--', colors='g')

    uilabx = np.unique(df['labx'])
    for labx in uilabx:
        if labx>0:
            Iloc = df['labx']==labx
            plt.plot(np.where(Iloc)[0], df['X'].loc[Iloc].values, marker='.', linewidth=0.1)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)

    return ax

# %%
# =============================================================================
# RISK METRICS
# =============================================================================

def risk_performance_metrics(perf):
    # Setup values for computations
    if not np.any(perf.columns=='portfolio_value'):
        perf['portfolio_value']=None
#    if not np.any(perf.columns=='max_drawdown'):
#        perf['max_drawdown']=None
        
    if np.any(perf.columns=='positions'):
        #zipline
        positions = np.array(list(map(lambda x: x!=[], perf.positions.values)), dtype='int')
    elif np.any(perf.columns=='invested'):
        #our approach
        positions = perf.invested
    else:
        logger.info('Positions can not be retrieved.')
        return None

    portfolio_value = perf.portfolio_value.values
    close = perf.asset.values

    # Determine trading boundaries
#    trade_boundaries=np.diff(positions,1)  
#    trades_start=np.where(trade_boundaries==1)[0]
#    trades_stop=np.where(trade_boundaries==-1)[0]
#    idx_trades=list(zip(trades_start,trades_stop))
    idx_trades=ones2region(positions)
    
    # Retrieve amount win/lose for the particular trade
    trades_values = np.array(list(map(lambda x: portfolio_value[x[1]]-portfolio_value[x[0]], idx_trades)))
    trades_values = trades_values[np.isnan(trades_values)==False]
    trades_win=[]
    trades_lose=[]

    if np.any(trades_values>0):
        trades_win = trades_values[trades_values>0]
    if np.any(trades_values<=0):
        trades_lose = trades_values[trades_values<=0]
        

    # Compute various performances
    out = dict()
    out['ratio_profit']         = ratio_profit(portfolio_value)
    out['ratio_buy_and_hodl']   = ratio_profit(close)
    out['expectancy']           = expectancy(trades_win,trades_lose)
    out['largest_losing_trade'] = largest_losing_trade(trades_lose)
    out['win_percentage']       = win_percentage(trades_win,trades_lose)
    out['profit_factor']        = profit_factor(trades_win,trades_lose)
    out['maximum_drawdown']     = maximum_drawdown(perf, idx_trades)
    out['number_of_trades']     = len(idx_trades)
    out['winning_trades']       = len(trades_win)
    out['losing_trades']        = len(trades_lose)
    out['winning_balance']      = np.sum(trades_win)
    out['losing_balance']       = np.sum(trades_lose)
    out['Total_balance']        = np.sum(trades_values)
    
    return(out)

# Metrics
def ratio_profit(portfolio_value):
    #perf.algorithm_period_return
    portfolio_value=portfolio_value[~np.isnan(portfolio_value)]
    return (portfolio_value[-1]/portfolio_value[0])-1

def win_percentage(trades_win,trades_lose):
    # Win Percentage tells us how many winning trades that we have in relation to the total number of trades taken.
    # For example: 77 Winning Trades / 140 Total Trades = 55% Win Percentage
    out = 0
    if (len(trades_win)+len(trades_lose))>0:
        out = len(trades_win) / (len(trades_win)+len(trades_lose))
    return(out)

def expectancy(trades_win,trades_lose):
    # Evaluating a trading strategy or system at any given account size
    # Expectancy = (Winning Percentage x Average Win Size) – (Losing Percentage x Average Loss Size)
    expectcy       = 0
    winningPerc    = 0
    losingPerc     = 0
    trades_win_nr  = len(trades_win)
    trades_lose_nr = len(trades_lose)
    
    if (trades_lose_nr+trades_win_nr)>0:
        winningPerc = trades_win_nr / (trades_lose_nr+trades_win_nr)
    if (trades_lose_nr+trades_win_nr)>0:
        losingPerc = trades_lose_nr / (trades_lose_nr+trades_win_nr)
    
    if len(trades_win)>0 and len(trades_lose)>0:
        expectcy = (winningPerc*np.nan_to_num(np.mean(trades_win))) - (losingPerc*np.nan_to_num(np.mean(trades_lose)))

    return(expectcy)

def largest_losing_trade(trades_lose):
    # It will help in system design and stress testing of your strategy. Your largest losing trade is self-explanatory and measures the amount of the single largest loss incurred.
    # We want to keep our largest loss within a reasonable range, so as not to jeopardy blowing up our account. It is essential to know the reason behind your largest loss so that you can try to contain it in the future.
    out=0
    if len(trades_lose)>0:
        out=np.min(trades_lose)
    return(out)

def profit_factor(trades_win,trades_lose):
    # R is the profit factor which takes the total of your winning trades divided by the absolute value of your losing trades to determine your profitability.
    # So, if I have a winning percentage of only 40%, buy my winners are much larger than my losers, I will still turn a profit.  For example, take a look at the below table to further illustrate this point:
    #
    #10 Total Trades	
    #4 sum Winners	19900
    #6 sum Losers	12034
    #R=	19900/12034 = 1.65
    #
    # For example, let’s say that during the course of a year, your winning trades resulted in $ 26,500 profits and your losing trades resulted in $15,250 in losses.
    # Then based on this, your profit factor would be:
    # $26,500 / $ 15,250 = 1.74
    #
    # profit_factor
    # [<1] trading system or strategy is unprofitable, 
    # [1.10-1.40] is considered a moderately profitable system, 
    # [1.41-2.00] is quite good, and finally, 
    # [>2.01] is considered excellent.
    #

    losers_sum=0
    winners_sum=0
    out=0

    if len(trades_win)>0:
        winners_sum=np.sum(trades_win)
    if len(trades_lose)>0:
        losers_sum=np.abs(np.sum(trades_lose))
    if losers_sum>0:
        out=winners_sum/losers_sum
    else:
        out=2.99 # magic number
    
    if (winners_sum==0) and (losers_sum==0):
        out=0
        
    return(out)
    
def maximum_drawdown(perf, idx_trades):
    # Evaluate risk-adjusted returns
    # Maximum draw down represents how much money you lose from a recent account high before making a new high in your account.  For example, if you just had a high of $100,000 and then pull back to $85,000 before exceeding $100,000, then you would have a maximum draw down of 15% ($85,000/%100,000).  The maximum draw down is probably the most valuable key performance indicator you should monitor when it comes to trading performance.  
    # Minimize your draw downs may be one of the most important factors!
    
    # Maximum Drawdown measures a portfolio’s largest peak to valley decline prior to a new peak in the equity curve
    # Assume your trading account started at $10,000 and increased to $15,000 over a certain period, and then fell to $7,000 after a string of losing trades. 
    # Later, it rebounded a bit increasing to $9,000. Soon after, another string of losses resulted in the account falling to $ $6,000. 
    # After some time, you were able to get back on track and get the account level to $ 18,000.

    # The answer is: 40% was your max drawdown.
    # $6,000 (lowest low prior to new peak ) / $15,000 ( highest peak prior to a new peak )
    maxdrawdown=None
    if np.any(perf.columns=='returns'):
        maxdrawdown=np.zeros(len(idx_trades))
        for i in range(0, len(idx_trades)):
            high=np.nanmax(perf.returns.iloc[idx_trades[i][0]:idx_trades[i][1]+2])
            low=np.nanmin(perf.returns.iloc[idx_trades[i][0]:idx_trades[i][1]+2])
            if high!=0:
                maxdrawdown[i]=low/high
        
        maxdrawdown = maxdrawdown[~np.isin(maxdrawdown, [-np.inf, np.inf, np.nan])]
        maxdrawdown = np.nanmean(maxdrawdown)

    return(maxdrawdown)

# %%
def _plot_gridsearch(out, 
                     xlabel1='Percentage', ylabel1='winning_balance', 
                     xlabel2='Percentage', ylabel2='Nr Trades', 
                     figsize=(15,8), visible=True):
    """Plot gridsearch results.

    Parameters
    ----------
    out : dict
        Dictionary with gridsearch results.
    xlabel1, ylabel1, xlabel2, ylabel2 : str
        Axis labels.
    figsize : tuple
        Figure size.
    visible : bool, default=True
        Set figure visibility.
    """
    logger.info('Creating plot gridsearch..')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    # Set figure visibility
    fig.set_visible(visible)
    ax1.set_visible(visible)
    ax2.set_visible(visible)

    # Make plots across the gridsearch
    for k in range(len(out['window'])):
        ax1.plot(out['minperc'], out['balances'][:, k], label=f'window_{out["window"][k]}')
        ax2.plot(out['minperc'], out['trades'][:, k], label=f'window_{out["window"][k]}')

    ax1.grid(True)
    ax1.set_xlabel(xlabel1)
    ax1.set_ylabel(ylabel1)

    ax2.grid(True)
    ax2.set_xlabel(xlabel2)
    ax2.set_ylabel(ylabel2)

    if visible:
        plt.show()

    return fig, (ax1, ax2)


# Make plot
def _plot(out, threshold=0.25, xlabel='Time', ylabel='Input value', figsize=(25,15), visible=True):
    logger.info('Creating Plot..')
    df = out['X']
    loc_start = out['loc_start']
    loc_stop = out['loc_stop']
    loc_start_best = out['loc_start_best']
    loc_stop_best = out['loc_stop_best']
    simmat = out['simmat']

    # Top plot
    # agg = out['agg']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    # Set figure visibility
    fig.set_visible(visible)
    ax1.set_visible(visible)
    ax2.set_visible(visible)
    ax3.set_visible(visible)

    # Make heatmap
    ax1.matshow(np.flipud(simmat.T))
    ax1.set_aspect('auto')
    # ax1.gca().set_aspect('auto')
    ax1.grid(False)
    ax1.set_ylabel('Perc.difference in window\n(higher=better)')
    ax1.set_xlim(0, simmat.shape[0])

    xlabels = simmat.columns.values.astype(str)
    Iloc=np.mod(simmat.columns.values, 10)==0
    xlabels[Iloc==False]=''
    xlabels[-1]=simmat.columns.values[-1].astype(str)
    xlabels[0]=simmat.columns.values[0].astype(str)
    ax1.set_yticks(range(0, len(xlabels)))
    ax1.set_yticklabels(np.flipud(xlabels))
    ax1.grid(True, axis='x')

    # make aggregated figure
    # Normalized successes across the n windows for percentages above minperc.
    # 1 depicts that for location i, all of the 1000 windows of different length was succesfull in computing a percentage above minperc
    [outagg, I] = _agg_scores(simmat, threshold)
    ax2.plot(outagg)
    ax2.grid(True)
    ax2.set_ylabel('Cummulative\n#success windows')
    ax2.set_xlim(0, simmat.shape[0])
    ax2.hlines(threshold, 0, simmat.shape[0], linestyles='--', colors='r')

    # Plot local maxima
    if loc_stop_best is not None:
        ax2.vlines(loc_stop_best, 0, 1, linestyles='--', colors='r')
        ax3.plot(df.iloc[loc_stop_best], 'or', linewidth=1)
    # Plot local minima
    if loc_start_best is not None:
        ax2.vlines(loc_start_best, 0, 1, linestyles='--', colors='g')
        ax3.plot(df.iloc[loc_start_best], 'og', linewidth=1)

        for i in range(0,len(loc_start)):
            ax3.plot(df.iloc[np.arange(loc_start[i][0],loc_start[i][1])],'g', linewidth=2)
            for k in range(0,len(loc_stop[i])):
                ax3.plot(df.iloc[np.arange(loc_stop[i][k][0],loc_stop[i][k][1])],'r', linewidth=2)
    # Plot region-minima-maxima
    ax3.plot(df,'k', linewidth=1)

    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    ax3.grid(True)
    ax3.set_xlim(0,simmat.shape[0])
    if loc_start_best is not None:
        ax3.vlines(loc_start_best,df.min(),df.max(), linestyles='--', colors='g')
    if loc_stop_best is not None:
        ax3.vlines(loc_stop_best,df.min(),df.max(), linestyles='--', colors='r')

    if visible:
        plt.show()

    # Return
    return fig, (ax1, ax2, ax3)

# %% Verbosity
# =============================================================================
# Functions for verbosity
# =============================================================================
def convert_verbose_to_old(verbose):
    """Convert new verbosity to the old ones."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if isinstance(verbose, str) or verbose>=10:
        status_map = {
            60: 0, 'silent': 0, 'off': 0, 'no': 0, None: 0,
            40: 1, 'error': 1, 'critical': 1,
            30: 2, 'warning': 2,
            20: 3, 'info': 3,
            10: 4, 'debug': 4}
        return status_map.get(verbose, 0)
    else:
        return verbose

def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[caerus] WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose

def get_logger():
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info', return_status: bool = False):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)

    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert verbose to numeric level
    if verbose in (0, None):
        log_level = logging.CRITICAL + 10  # silent
    elif isinstance(verbose, str):
        levels = {
            'silent': logging.CRITICAL + 10,
            'off': logging.CRITICAL + 10,
            'no': logging.CRITICAL + 10,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        log_level = levels.get(verbose.lower(), logging.INFO)
    elif isinstance(verbose, int):
        log_level = verbose
    else:
        log_level = logging.INFO

    # Set package logger
    logger = logging.getLogger('caerus')
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    if return_status:
        return log_level

def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


def check_logger(verbose: [str, int] = None):
    """Check the logger."""
    if verbose is not None: set_logger(verbose)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.critical('CRITICAL')
