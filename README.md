# CAERUS
[![Build Status](https://travis-ci.org/erdoganta/caerus.svg?branch=master)](https://travis-ci.org/erdoganta/caerus)
[![PyPI Version](https://img.shields.io/pypi/v/caerus)](https://pypi.org/project/caerus/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdoganta/caerus/blob/master/LICENSE)

Detection of local minima with the corresponding local maxima within the given time-frame.

Caerus is a python package that detects points in time that are most probable local minima and local maxima. As an example is shown the last 300 days of BTC. Green are the local minima and red are the local maxima.
<p align="center">
  <img src="docs/figs/figure_btc_last_300days.png" width="900" />
</p>

This package determines the local-minima with the corresponding local-maxima within a given time-frame and can therefore also easily detect differences across time without any normalization procedures. The method is as following; in a forward rolling window, thousands of windows are iteratively created and for each window a percentage score is computed from the start-to-stop position. For resulting matrix [window x length dataframe], only the high scoring percentages, e.g. those above a certain value (minperc) are used. The cleaned matrix is then aggregated by sum per time-point followed by a cut using the threshold. The resulting regions are subsequently detected, and represent the starting-locations of the trade. The stop-locations are determined based on the distance and percentage of te start-locations. As an example, if you want to have best regions, use threshold=1, minperc=high and nlargest=1 (small).

### About the name
In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments. 
He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

## Contents
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install caerus from PyPI (recommended). caerus is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
It is distributed under the Apache 2.0 license.

```
pip install caerus
```
* Alternatively, install caerus from the GitHub source:

```bash
git clone https://github.com/erdoganta/caerus.git
cd caerus
python setup.py install
```  

## Quick Start
- Import caerus method

```python
import caerus as cs
```

- Simple example with constant window, minimum percentage and threshold
```python
df=pd.read_csv('https://github.com/erdoganta/caerus/blob/master/data/fb.csv')['close']
out = cs.fit(df)
fig = cs.makefig(out)
```
The input is a pandas dataframe or series and looks like this:
<p align="left">
  <img src="docs/figs/input_example.png" width="110" />
</p>
The output looks as below:
<p align="center">
  <img src="docs/figs/figure_fb.png" width="900" />
</p>


```python
df=pd.read_csv('https://github.com/erdoganta/caerus/blob/master/data/btc.csv')['Close']
out = cs.fit(df)
fig = cs.makefig(out)
```
The output looks as below:
<p align="center">
  <img src="docs/figs/figure_btc.png" width="900" />
</p>


- Gridsearch walks over the windows and over the percentages to determine optimal window, minimum percentage and the threshold.
```python
df = pd.read_csv('https://github.com/erdoganta/caerus/blob/master/data/btc.csv')['close']
out = cs.gridsearch(df)
```

The output looks as below:
<p align="center">
  <img src="docs/figs/figure_gridsearch_btc.png" width="900" />
</p>


## Citation
Please cite caerus in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdoganta2019caerus,
  title={caerus},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdoganta/caerus}},
}
```

## Maintainers
* Erdogan Taskesen, github: [erdoganta](https://github.com/erdoganta)

## © Copyright
See [LICENSE](LICENSE) for details.
