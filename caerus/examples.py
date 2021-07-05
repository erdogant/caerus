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
cs = caerus(minperc=5, window=10)
cs.fit(X)
cs.plot()


# %% Import class
from caerus import caerus
cs = caerus(minperc=1)
X = cs.download_example(name='btc')
cs.fit(X)
cs.plot()

# %%
from caerus import caerus
cs = caerus(minperc=4)
X = cs.download_example(name='btc')
cs.fit(X)
cs.plot()


# %% Gridsearch
from caerus import caerus
cs = caerus()
X = cs.download_example(name='facebook')
cs.gridsearch(X)
# cs.gridsearch(X, window=np.arange(50,550,100), minperc=np.arange(1,20,5))
cs.plot()

# Take best results
cs = caerus(minperc=6)
cs.fit(X)
cs.plot()

# %%
