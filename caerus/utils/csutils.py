"""This code is part of caerus and is not designed for usage of seperate parts."""
#--------------------------------------------------------------------------
# Name        : caerus.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : May. 2020
#--------------------------------------------------------------------------

from caerus.utils.ones2idx import ones2region, idx2region
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# %% utils
def _check_input(X):
    # Convert to dataframe
    if isinstance(X, pd.DataFrame): raise Exception('[caerus] >Error: Input data must be of type numpy-array or list.')
    if 'numpy' in str(type(X)) or 'list' in str(type(X)): X = pd.Series(X)
    if X.shape[0]!=X.size: raise Exception('[caerus] >Error : Input dataframe can only be a 1D-vector.')
    # reset index
    X.reset_index(drop=True, inplace=True)
    return X

# %% Aggregation of scores over the windows and intervals
def _agg_scores(out, threshold=0):
    outagg=np.nansum(out>0, axis=1)
    # Normalize for the window size that is used. Towards the end smaller windows are only avaialbe which is otherwise unfair for the threshold usage.
    windowCorrectionFactor = np.ones_like(outagg)*out.shape[1]
    tmpvalue = np.arange(1, out.shape[1])[::-1]
    windowCorrectionFactor[-len(tmpvalue):]=tmpvalue

    outagg = outagg/windowCorrectionFactor
    I=outagg>threshold
    return(outagg, I)

# %% Merge regions
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
def _compute_region_scores(df, window=1000, verbose=3):
    # Compute percentage score for each 
    # 1. Window
    # 2. Position
    
    # Start with empty dataframe
    out=pd.DataFrame()

    # Reverse dataframe to create forward-rolling window
    df=df[::-1]
    for i in tqdm(range(2,window), disable=(True if verbose==0 else False)):
        dfperc = df.rolling(i).apply(_compute_percentage, raw=True)[::-1] #.values.flatten()
        out[i]=dfperc
    
    out[np.isinf(out)]=np.nan
    # out.fillna(value=0, inplace=True)

    return(out)

#%% Detect starting positions for regions
def _regions_detect_start(out, minperc, threshold, extb=5, extf=5):
    # Find start-locations
    [outagg, I] = _agg_scores(out, threshold)
    locs_start=ones2region(I)
    
    if len(locs_start)==0:
        locs_start=None

    # Merge regions if only seperated with few intervals
    locs_start = regions_merge(locs_start, extb=extb, extf=extf)

    return(locs_start, outagg)

#%% Detect stop locations based on the starting positions
def _regions_detect_stop(out, locs_start, nlargest, extb=5, extf=5, verbose=0):
    # Find stop-locations
    locs_stop=None
    if not isinstance(locs_start,type(None)):
        locs_stop=[]
        # out[np.isinf(out)]=np.nan
        
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
                xtmp = out.iloc[getloc[p][0]:getloc[p][1]+1,:]
                meanPerc = np.nanmean(xtmp)
                loc_mean_percentage.append(meanPerc)
            loc_mean_percentage=np.array(loc_mean_percentage)
            idx=np.argsort(loc_mean_percentage)[::-1]
            getloc=np.array(getloc)[idx]
            
            locs_stop.append(getloc.tolist())

    return(locs_stop)

# %% Compute percentage
def _compute_percentage(r):
    perc=_percentage_getdiff(r[0],r[-1])
    return(perc) 


# %% Compute percentage between current price and starting price
def _percentage_getdiff(current_price, previous_price):
    assert isinstance(current_price, float)
    assert isinstance(previous_price, float)

    if current_price>previous_price:
        # Increase
        diff_perc=(current_price-previous_price)/previous_price*100
    else:
        # Decrease
        diff_perc=-(previous_price-current_price)/previous_price*100
    
    return(diff_perc)


# %% Create labels
def to_df(results):
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
            df['labx'].iloc[idx_valley]=i+1

            for k in range(0,len(results['loc_stop'][i])):
                df['labx'].iloc[np.arange(results['loc_stop'][i][k][0], results['loc_stop'][i][k][1])]=i+1

    return df
