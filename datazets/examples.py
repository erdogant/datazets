# %%
import datazets as dz
df = dz.get(data='bigbang', overwrite=True)


# %%
import datazets as dz
url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = dz.get(url=url, sep=',')
df.shape

# %%
import datazets as dz
df = dz.get(data='auto_mpg')
# df.shape

# %% New
import datazets as dz
IMAGES = ['faces', 'mnist', 'southern_nebula', 'flowers', 'scenes', 'cat_and_dog']

# %% test
datasets = ['census_income',
            'stormofswords',
            'sprinkler',
            'titanic',
            'student',
            'fifa',
            'cancer',
            'auto_mpg',
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
            'waterpump',
            'elections',
            'tips',
            'predictive_maintenance',
            'bigbang',
            ]

datasets = datasets + IMAGES

# %%
for data in datasets:
    df = dz.get(data=data)
    print(len(df))

# %%
