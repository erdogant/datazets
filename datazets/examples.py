# %%
import datazets as ds
df = ds.get(data='mnist')

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
            'retail',
            'auto_mpg',
            'random_discrete',
            'ads',
            'breast_cancer',
            'bitcoin',
            'digits',
            'energy',
            'meta',
            'flowers',
            'faces',
            'mnist',
            'scenes',
            'digits',
            ]

for data in datasets:
    df = ds.get(data=data)
    # print(df.shape)
