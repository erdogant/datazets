# datazets

[![Python](https://img.shields.io/pypi/pyversions/datazets)](https://img.shields.io/pypi/pyversions/datazets)
[![Pypi](https://img.shields.io/pypi/v/datazets)](https://pypi.org/project/datazets/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/datazets/)
[![LOC](https://sloc.xyz/github/erdogant/datazets/?category=code)](https://github.com/erdogant/datazets/)
[![Downloads](https://static.pepy.tech/personalized-badge/datazets?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/datazets)
[![Downloads](https://static.pepy.tech/personalized-badge/datazets?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/datazets)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/datazets/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/datazets.svg)](https://github.com/erdogant/datazets/network)
[![Issues](https://img.shields.io/github/issues/erdogant/datazets.svg)](https://github.com/erdogant/datazets/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/datazets/pages/html/Documentation.html#colab-notebook)
![GitHub Repo stars](https://img.shields.io/github/stars/erdogant/datazets)
![GitHub repo size](https://img.shields.io/github/repo-size/erdogant/datazets)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/datazets/pages/html/Documentation.html#)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->





<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* ``datazets`` is Python package

# 
**Star this repo if you like it! ⭐️**
#

### Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install datazets from PyPI (recommended). datazets is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```bash
conda create -n env_datazets python=3.8
conda activate env_datazets
```

```bash
pip install datazets            # normal install
pip install --upgrade datazets # or update if needed
```

* Alternatively, you can install from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/datazets.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/datazets#egg=master
pip install git+https://github.com/erdogant/datazets

# By cloning
git clone https://github.com/erdogant/datazets.git
cd datazets
pip install -U .
```  

#### Import datazets package
```python
import datazets as datazets
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/datazets/data/example_data.csv')
model = datazets.fit(df)
G = datazets.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/datazets/blob/master/docs/figs/fig1.png" width="600" />
  
</p>


#### References
* https://github.com/erdogant/datazets

#### Citation
Please cite in your publications if this is useful for your research (see citation).
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.
