# import caerus as caerus
# print(caerus.__version__)

# from caerus import caerus
# print(dir(caerus))

# %% Import class
from caerus import caerus
cs = caerus(verbose='info')
X = cs.download_example(name='facebook')

cs = caerus(verbose='info')
results = cs.fit(X)
fig, axs = cs.plot(visible=False)
fig, axs = cs.plot(visible=True)


# %% Import class
from caerus import caerus
cs = caerus()
X = cs.download_example(name='bitcoin')
# cs = caerus()
cs = caerus(window=200)
results = cs.fit(X[-300:])
fig, axs = cs.plot()


# %% Import class
from caerus import caerus
cs = caerus(minperc=1)
X = cs.download_example(name='bitcoin')
results = cs.fit(X)
fig, axs = cs.plot()

# %%
from caerus import caerus
cs = caerus(minperc=4)
X = cs.download_example(name='bitcoin')
results = cs.fit(X)
fig, axs = cs.plot()


# %% Gridsearch
from caerus import caerus
cs = caerus(verbose='info')
X = cs.download_example(name='facebook')
cs.gridsearch(X)
# cs.gridsearch(X, window=np.arange(50,550,100), minperc=np.arange(1,20,5))
fig, axs = cs.plot(visible=False)
fig, axs = cs.plot(visible=True)

# Take best results
cs = caerus(minperc=6)
results = cs.fit(X)
fig, axs = cs.plot()

# %%
