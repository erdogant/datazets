# %%
import datazets as dz
X = dz.get(data='img_peaks1', overwrite=True)

# %%
import datazets as dz
df = dz.get(data='surfspots', overwrite=True)

# %%
import datazets as dz
url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = dz.get(url=url, sep=',')
df.shape

# %%
import datazets as dz
df = dz.get(data='meta')
df = dz.get(data='elections_usa')
df = dz.get(data='elections_rus')
# df.shape

# %% New
import datazets as dz
IMAGES = ['faces', 'mnist', 'southern_nebula', 'flowers', 'scenes', 'cat_and_dog', 'img_peaks1', 'img_peaks2']

datasets = ['census_income',
            'stormofswords',
            'sprinkler',
            'titanic',
            'student',
            'fifa',
            'cancer',
            'marketing_retail',
            'auto_mpg',
            'random_discrete',
            'ads',
            'breast_cancer',
            'bitcoin',
            'digits',
            'energy',
            'meta',
            'gas_prices',
            'iris',
            'malicious_urls',
            'malicious_phish',
            'waterpump',
            'elections_usa',
            'elections_rus',
            'tips',
            'predictive_maintenance',
            'bigbang',
            'surfspots',
            ]

datasets = datasets + IMAGES

# %%
for data in datasets:
    df = dz.get(data=data, verbose=0)
    if isinstance(df, list):
        print(f'{data} | None | Image')
    else:
        print(f'{data} | {df.shape} | ')
    # print(df)
    # input('prss enter')

# %%

# | Dataset Name           | Shape Size           | Type                | Description                                                                                   |
# |------------------------|----------------------|---------------------|-----------------------------------------------------------------------------------------------|
# | meta                   | (1472, 20)           | Continuous | time   | Stock price of Meta                                                                           |
# | bitcoin                | (2522, 2)            | Continuous | time   | Bitcoin price history data for time series and price prediction                               |
# | energy                 | (68, 3)              | Network             | Data on building energy consumption                                                           |
# | gas_prices             | (6556, 2)            | Mixed | time        | Historical gas prices by region for trend analysis                                            |
# | iris                   | (150, 3)             | Continuous          | Classic flower classification dataset with iris species measurements with coordinates         |
# | ads                    | (10000, 10)          | Discrete            | Data on online ads, covering click-through rates and targeting information                    |
# | bigbang                | (9, 3)               | Network             | Data on *The Big Bang Theory* episodes and characters                                         |
# | malicious_urls         | (387588, 2)          | Text                | URLs labeled as malicious or benign, useful in cybersecurity                                  |
# | malicious_phish        | (651191, 4)          | Text                | URLs labeled as malicious or benign, defacement, phishing, malware (cybersecurity)            |
# | random_discrete        | (1000, 5)            | Discrete            | Synthetic dataset with random discrete variables, useful for probability modeling             |
# | stormofswords          | (352, 3)             | Network             | Character data from *A Storm of Swords*, with relationships, traits, and alliance info        |
# | sprinkler              | (1000, 4)            | Discrete            | Synthetic dataset with binary variables for rain and sprinkler probability illustration       |
# | auto_mpg               | (392, 8)             | Mixed               | Data on cars with features for predicting miles per gallon                                    |
# | breast_cancer          | (569, 30)            | Mixed               | Dataset for breast cancer diagnosis prediction using tumor cell features                      |
# | cancer                 | (4674, 9)            | Mixed               | Cancer patient data for classification and prediction of diagnosis outcome with Coordinates   |
# | census_income          | (32561, 15)          | Mixed               | US Census data with various demographic and economic factors for income prediction            |
# | elections_rus          | (94487, 23)          | Mixed               | Russian election data with demographic and political attributes                               |
# | elections_usa          | (24611, 8)           | Mixed               | US election data with demographic and political attributes                                    |
# | fifa                   | (128, 27)            | Mixed               | FIFA player stats including attributes like skill, position, country, and performance         |
# | marketing_retail       | (999, 8)             | Mixed               | Retail customer data for behavior and segmentation analysis                                   |
# | predictive_maintenance | (10000, 14)          | Mixed               | Industrial equipment data for predictive maintenance                                          |
# | student                | (649, 33)            | Mixed               | Data on student performance with socio-demographic and academic factors                       |
# | surfspots              | (9413, 4)            | Mixed | latlon      | Information on global surf spots, with details on location and wave characteristics           |
# | tips                   | (244, 7)             | Mixed               | Restaurant tipping data with variables on meal size, day, and tip amount                      |
# | titanic                | (891, 12)            | Mixed               | Titanic passenger data with demographic, class, and survival information                      |
# | waterpump              | (59400, 41)          | Mixed               | Water pump data with features for predicting functionality and maintenance needs              |
# | cat_and_dog            | None                 | Image               | Images of cats and dogs for classification and object recognition                             |
# | digits                 | (1083, 65)           | Image               | Handwritten digit images (8x8 pixels) for recognition and classification                      |
# | faces                  | (400, 4097)          | Image               | Images of faces used in facial recognition and feature analysis                               |
# | flowers                | None                 | Image               | Various flower images for classification and image recognition                                |
# | img_peaks1             | (930, 930, 3)        | Image               | Synthetic peak images for image processing and analysis                                       |
# | img_peaks2             | (125, 496, 3)        | Image               | Additional synthetic peak images for image processing                                         |
# | mnist                  | (1797, 65)           | Image               | MNIST handwritten digit images (28x28 pixels) for classification tasks                        |
# | scenes                 | None                 | Image               | Scene images for scene classification tasks                                                   |
# | southern_nebula        | None                 | Image               | Images of the Southern Nebula, suitable for astronomical analysis                             |
