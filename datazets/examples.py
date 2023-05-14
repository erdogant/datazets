# %%
# import datazets as dz
# url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
# df = dz.get(url=url, sep=',')

# %%
import datazets as dz
df = dz.get(data='USA_elections')
df.shape
# %% New

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
            'USA_elections',
            'tips',
            'predictive_maintenance',
            ]

for data in datasets:
    df = dz.get(data=data)
    print(df.shape)
