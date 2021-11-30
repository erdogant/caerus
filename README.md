# caerus

[![Python](https://img.shields.io/pypi/pyversions/caerus)](https://img.shields.io/pypi/pyversions/caerus)
[![PyPI Version](https://img.shields.io/pypi/v/caerus)](https://pypi.org/project/caerus/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/caerus/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/caerus)](https://pepy.tech/project/caerus)
[![Downloads](https://pepy.tech/badge/caerus/month)](https://pepy.tech/project/caerus/month)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![DOI](https://zenodo.org/badge/230082801.svg)](https://zenodo.org/badge/latestdoi/230082801)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments.
He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

**caerus** is a python package providing that determines the local-minima with the corresponding local-maxima within the given time-frame. The method is build
using a forward rolling window to iteratively evaluate thousands of windows. For each window a score of percentages is computed from the start-to-stop
position. The resulting matrix is a [window x length dataframe] for which only the high scoring percentages, e.g. those above a certain value (minperc) are
used. The best scoring percentages is then aggregated by sum per time-point followed by a cut using the threshold. The resulting regions are subsequently
detected, and represent the starting-locations of the trade. The stop-locations are determined based on the distance and percentage of te start-locations.
As an example, if you want to have best regions, use threshold=1, minperc=high and nlargest=1 (small).

Here are just a few of the things that caerus does well:
    - Ouput contains detected start-stop regions of local minima and maxima.
    - Figures are created.
    - Parameter gridsearch.
    - Designed for the detection of complex trend movements.
    
    
### Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install caerus from PyPI (recommended). caerus is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```python
conda create -n env_caerus python=3.7
conda activate env_caerus
```

```bash
pip install caerus
```

* Alternatively, install caerus from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/caerus.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/caerus#egg=master
```  

#### Import caerus package
```python
from caerus import caerus
```

#### Example 1:
```python
cs = caerus()
X = cs.download_example()
cs.fit(X)
cs.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/caerus/blob/master/docs/figs/figure_btc.png" width="600" />
  <img src="https://github.com/erdogant/caerus/blob/master/docs/figs/figure_btc_last_300days.png" width="600" />
</p>

#### Example 2:
```python
cs = caerus()
X = cs.download_example(name='facebook')
cs.fit(X)
cs.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/caerus/blob/master/docs/figs/figure_fb.png" width="600" />
</p>

#### Example gridsearch:
```python
cs = caerus()
X = cs.download_example(name='facebook')
cs.gridsearch(X)
cs.plot()

# Change some gridsearch parameters
cs.gridsearch(X, window=np.arange(50,550,100), minperc=np.arange(1,20,5))
cs.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/caerus/blob/master/docs/figs/figure_gridsearch_btc.png" width="600" />
</p>


### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant/caerus)
	Please cite in your publications if this is useful for your research (see citation).
	Contributions are welcome.
	This work is created and maintained in my free time. If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated.

#### References
* https://github.com/erdogant/caerus
