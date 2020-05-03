import caerus as caerus
print(caerus.__version__)

from caerus import caerus
print(dir(caerus))

# %% Import class
from caerus import caerus
cs = caerus()
X = cs.download_example(name='facebook')
cs = caerus()
cs.fit(X)
cs.plot()


# %% Import class
from caerus import caerus
cs = caerus()
X = cs.download_example(name='btc')
cs = caerus(minperc=1, window=100)
cs.fit(X)
cs.plot()


# %% Import class
from caerus import caerus
cs = caerus(minperc=1)
X = cs.download_example(name='btc')
cs.fit(X)
cs.plot()

# %% Gridsearch
from caerus import caerus
cs = caerus()
X = cs.download_example(name='facebook')
cs.gridsearch(X, window=np.arange(50,550,100), minperc=np.arange(1,20,5))
cs.plot()

# Take best results
cs = caerus(minperc=6)
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
