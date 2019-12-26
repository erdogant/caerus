#%% Libraries
import numpy as np

#%% Compute percentage between current price and starting price
def getdiff(current_price, previous_price):
    assert isinstance(current_price, float)
    assert isinstance(previous_price, float)

    if current_price>previous_price:
        # Increase
        diff_perc=(current_price-previous_price)/previous_price*100
    else:
        # Decrease
        diff_perc=-(previous_price-current_price)/previous_price*100
    
    return(diff_perc)

#%% Compute various statistics for list of percentages
def getfeat(L, maxlen=20):
    # Input is a list (L) of percentages that is relative to the starting-position.
    # 1. maximum percentage in list
    # 2. number of items in monochronical increasing order in list
    # 3. weighed average
    out = dict()

    # Model is trained on the maximum length. If exceeded, results will be flawed as the extracted features will be different. 
    # As an example, the longer it gets, the more signifacnt it will become
    L=L[-maxlen:]
    
    Iincr=[True]
    Idecr=[True]
    if len(L)>1:
        Iincr=np.diff(L)>=0
        Idecr=np.diff(L)<=0
        
    # Some features for percentage locker
    out['max']     = L[-1]
    out['sumincr'] = sum(Iincr)
    out['wmean']   = np.average(L, weights=np.arange(1,len(L)+1))

    # Some features for stoploss
    out['sumdecr'] = sum(Idecr)
    out['sumneg']  = sum(np.array(L)<0)
    return(out)
    
#%% end