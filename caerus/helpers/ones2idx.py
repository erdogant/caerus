""" This function searches for stretches of ones in a 1D array and then converts it to start-stop indices

	A = ones2region(data, <optional>)
	A = idx2region(data, <optional>)
	A = region2ones(data, <optional>)
	A = ones2idx(data, <optional>)
	A = region2idx(data, <optional>)

 INPUT:
   data:           1D numpy array
                   np.array([0,0,0,1,1,1,1,1,0,0,0,1,0])
   
 OUTPUT
	output

 DESCRIPTION
   This function searches for stretches of ones in a 1D array and then converts it to start-stop indices

 EXAMPLE
   from ethelpers.ones2idx import ones2region, idx2region, region2ones, ones2idx

   data=np.array([1,0,0,1,1,1,1,1,0,0,0,1])
   A = ones2region(data)

   data=np.array([1,2,3,4,5,8,9,10,15,16,20])
   A = idx2region(data)

   data=[(0,0),(3,7),(11,11)]
   A = region2ones(data)
   
   data=np.array([1,0,0,1,1,1,1,1,0,0,0,1])
   A = ones2idx(data)
   
   # Test
   data=np.array([1,0,0,1,1,1,1,1,0,0,0,1])
   idx2ones(ones2region(data))==data

"""

#--------------------------------------------------------------------------
# Name        : ones2idx.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : March. 2019
#--------------------------------------------------------------------------

import numpy as np

#%% Convert to index
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

#%% Convert to index
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

#%% Convert index to ones
def region2ones(data):
    out=np.zeros(np.max(data)+1).astype(int)
    for i in range(0,len(data)):
        out[np.arange(data[i][0],data[i][1]+1)]=1

    return(out)

#%% Convert index to ones
def region2idx(data):
    out=ones2idx(region2ones(data))
    return(out)
    
#%% Convert index to ones
def ones2idx(data):
    out = np.where(data)[0]
    return(out)