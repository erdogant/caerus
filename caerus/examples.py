# # %%
# import caerus as cs
# print(dir(cs))
# print(cs.__version__)



import numpy as np
import caerus as caerus
print(caerus.__version__)


# %% Import class
from caerus import caerus
print(dir(caerus))

cs = caerus()
X = cs.download_example()['close'].values
cs.fit(X)
cs.plot()


# %%
df = cs.download_example()
X = df['close'].values
out = cs.fit(df['close'], window=50, minperc=3, threshold=0.25, nlargest=10)
out = cs.fit(df, window=50, minperc=3, threshold=0.25, nlargest=10)

out = cs.fit(X, window=50, minperc=3, threshold=0.25, nlargest=10)
fig = cs.makefig(out)

# Best parameters
[out_balance, out_trades]=caerus.gridsearch(df)

# Shuffle
df = picklefast.load('../DATA/STOCK/btc1h.pkl')['close']
np.random.shuffle(df)
outNull=cs.fit(df, window=50, minperc=3, nlargest=10, threshold=0.25)
plt.figure();plt.hist(outNull['agg'], bins=50)
Praw=hypotesting(out['agg'], outNull['agg'], showfig=0, bound='up')['Praw']
model=distfit(outNull['agg'], showfig=1, alpha=0.05)[0]

# %%
