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
![GitHub Repo stars](https://img.shields.io/github/stars/erdogant/datazets)
![GitHub repo size](https://img.shields.io/github/repo-size/erdogant/datazets)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/datazets/pages/html/Documentation.html#)

* ``datazets`` is Python package

# 
**Star this repo if you like it! ⭐️**
#


```bash
pip install datazets
```

#### Import datazets
```python
# Import library
import datazets as dz
# Import data set
df = dz.get('titanic')

```

#### Data sets:


| Dataset Name           | Shape Size           | Type                | Description                                                                                   |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| meta                   | (1472, 20)           | Continuous | time   | Stock price of Meta                                                                           |
| bitcoin                | (2522, 2)            | Continuous | time   | Bitcoin price history data for time series and price prediction                               |
| iris                   | (150, 3)             | Continuous          | Classic flower classification dataset with iris species measurements with coordinates         |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| gas_prices             | (6556, 2)            | Mixed | time        | Historical gas prices by region for trend analysis                                            |
| ads                    | (10000, 10)          | Discrete            | Data on online ads, covering click-through rates and targeting information                    |
| sprinkler              | (1000, 4)            | Discrete            | Synthetic dataset with binary variables for rain and sprinkler probability illustration       |
| random_discrete        | (1000, 5)            | Discrete            | Synthetic dataset with random discrete variables, useful for probability modeling             |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| malicious_urls         | (387588, 2)          | Text                | URLs labeled as malicious or benign, useful in cybersecurity                                  |
| malicious_phish        | (651191, 4)          | Text                | URLs labeled as malicious or benign, defacement, phishing, malware (cybersecurity)            |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| stormofswords          | (352, 3)             | Network             | Character data from *A Storm of Swords*, with relationships, traits, and alliance info        |
| bigbang                | (9, 3)               | Network             | Data on *The Big Bang Theory* episodes and characters                                         |
| energy                 | (68, 3)              | Network             | Data on building energy consumption                                                           |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| auto_mpg               | (392, 8)             | Mixed               | Data on cars with features for predicting miles per gallon                                    |
| breast_cancer          | (569, 30)            | Mixed               | Dataset for breast cancer diagnosis prediction using tumor cell features                      |
| cancer                 | (4674, 9)            | Mixed               | Cancer patient data for classification and prediction of diagnosis outcome with Coordinates   |
| census_income          | (32561, 15)          | Mixed               | US Census data with various demographic and economic factors for income prediction            |
| elections_rus          | (94487, 23)          | Mixed               | Russian election data with demographic and political attributes                               |
| elections_usa          | (24611, 8)           | Mixed               | US election data with demographic and political attributes                                    |
| fifa                   | (128, 27)            | Mixed               | FIFA player stats including attributes like skill, position, country, and performance         |
| marketing_retail       | (999, 8)             | Mixed               | Retail customer data for behavior and segmentation analysis                                   |
| predictive_maintenance | (10000, 14)          | Mixed               | Industrial equipment data for predictive maintenance                                          |
| student                | (649, 33)            | Mixed               | Data on student performance with socio-demographic and academic factors                       |
| surfspots              | (9413, 4)            | Mixed | latlon      | Information on global surf spots, with details on location and wave characteristics           |
| tips                   | (244, 7)             | Mixed               | Restaurant tipping data with variables on meal size, day, and tip amount                      |
| titanic                | (891, 12)            | Mixed               | Titanic passenger data with demographic, class, and survival information                      |
| waterpump              | (59400, 41)          | Mixed               | Water pump data with features for predicting functionality and maintenance needs              |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| cat_and_dog            | None                 | Image               | Images of cats and dogs for classification and object recognition                             |
| digits                 | (1083, 65)           | Image               | Handwritten digit images (8x8 pixels) for recognition and classification                      |
| faces                  | (400, 4097)          | Image               | Images of faces used in facial recognition and feature analysis                               |
| flowers                | None                 | Image               | Various flower images for classification and image recognition                                |
| img_peaks1             | (930, 930, 3)        | Image               | Synthetic peak images for image processing and analysis                                       |
| img_peaks2             | (125, 496, 3)        | Image               | Additional synthetic peak images for image processing                                         |
| mnist                  | (1797, 65)           | Image               | MNIST handwritten digit images (28x28 pixels) for classification tasks                        |
| scenes                 | None                 | Image               | Scene images for scene classification tasks                                                   |
| southern_nebula        | None                 | Image               | Images of the Southern Nebula, suitable for astronomical analysis                             |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
| blobs                  | Custom               | Continuous          | Synthetic data of datapoints in blob shape                                                    |
| moons                  | Custom               | Continuous          | Synthetic data of datapoints in moon shape                                                    |
| circles                | Custom               | Continuous          | Synthetic data of datapoints in circle shape                                                  |
| anisotropic            | Custom               | Continuous          | Synthetic data of datapoints with anisotropic shape                                           |
| globular               | Custom               | Continuous          | Synthetic data of datapoints with globular shape                                              |
| uniform                | Custom               | Continuous          | Synthetic data with uniform shape                                                             |
| densities              | Custom               | Continuous          | Synthetic data with different densities                                                       |
|------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|



#### Example:

```python

import datazets as dz
df = dz.get(data='titanic')

```


```python

import datazets as dz

# Import from url
url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = dz.get(url=url, sep=',')

```

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.
