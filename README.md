# CAERUS
Detection of local minima with the corresponding local maxima within the given time-frame

[![Build Status](https://travis-ci.org/erdoganta/caerus.svg?branch=master)](https://travis-ci.org/erdoganta/caerus)
[![PyPI Version](https://img.shields.io/pypi/v/caerus)](https://pypi.org/project/caerus/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdoganta/caerus/blob/master/LICENSE)

caerus is a python package that detects points in time that are most probable local minima and local maxima.

<p align="center">
  <img src="examples/figure_1.png" width="600" />
</p>

About the name
In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments. 
He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

This package determines the local-minima with the corresponding local-maxima within the given time-frame.
The method is as following; in a forward rolling window, thousands of windows are iteratively created and for each window a percentage score is computed from the start-to-stop position. For resulting matrix [window x length dataframe], only the high scoring percentages, e.g. those above a certain value (minperc) are used. The cleaned matrix is then aggregated by sum per time-point followed by a cut using the threshold. The resulting regions are subsequently detected, and represent the starting-locations of the trade. The stop-locations are determined based on the distance and percentage of te start-locations. As an example, if you want to have best regions, use threshold=1, minperc=high and nlargest=1 (small).

caerus is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
It is distributed under the Apache 2.0 license.

## Contents
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install caerus from PyPI (recommended):
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
from caerus import caerus
```

- Simple example with constant window, minimum percentage and threshold
```python
df = picklefast.load('../DATA/STOCK/btcyears.pkl')['Close']
out = caerus.caerus(df, window=50, minperc=3, threshold=0.25, nlargest=10)
```

The output looks as below:

<p align="center">
  <img src="examples/karateclub/d3graph_1.png" width="300" />
  <img src="examples/karateclub/d3graph_2.png" width="300" />
</p>


- Gridsearch to determine optimal window, minimum percentage and threshold
```python
df = picklefast.load('../DATA/STOCK/btcyears.pkl')['Close']
out = caerus.gridsearch(df)
```

The output looks as below:

<p align="center">
  <img src="examples/karateclub/d3graph_1.png" width="300" />
  <img src="examples/karateclub/d3graph_2.png" width="300" />
</p>



## Contribute
We welcome all kinds of contributions.
See the [Contribution](CONTRIBUTING.md) guide for more details.

## Citation
Please cite d3graph in your publications if this is useful for your research. Here is an example BibTeX entry:
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
