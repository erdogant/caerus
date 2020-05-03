# caerus

[![Python](https://img.shields.io/pypi/pyversions/caerus)](https://img.shields.io/pypi/pyversions/caerus)
[![PyPI Version](https://img.shields.io/pypi/v/caerus)](https://pypi.org/project/caerus/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/caerus/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/caerus/month)](https://pepy.tech/project/caerus/month)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* caerus is Python package

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

#### Example:
```python
from caerus import caerus
cs = caerus()
X = cs.download_example()
cs.fit(X)
cs.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/caerus/blob/master/docs/figs/fig1.png" width="600" />
</p>

#### Example gridsearch:
```python
from caerus import caerus
cs = caerus()
X = cs.download_example(name='facebook')
cs.gridsearch(X)
cs.plot()

# Change some gridsearch parameters
cs.gridsearch(X, window=np.arange(50,550,100), minperc=np.arange(1,20,5))
cs.plot()
```



#### Citation
Please cite caerus in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020caerus,
  title={caerus},
  author={Erdogan Taskesen},
  year={2020},
  howpublished={\url{https://github.com/erdogant/caerus}},
}
```

#### References
* https://github.com/erdogant/caerus

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
	See [LICENSE](LICENSE) for details.
	This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
