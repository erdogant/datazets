import unittest
import datazets as dz

class Testdatazets(unittest.TestCase):

    def test_import_example(self):
        df = dz.get(data='random_discrete', n=1000)
        assert df.shape==(1000, 5)
        df = dz.get(data='random_discrete', n=10)
        assert df.shape==(10, 5)

    def test_image_datasets(self):
        img = dz.get(data='flowers', verbose=0)
        assert len(img)==214
        img = dz.get(data='faces', verbose=0)
        assert img.shape==(400, 4097)
        img = dz.get(data='mnist', verbose=0)
        assert img.shape==(1797, 65)
        img = dz.get(data='scenes', verbose=0)
        assert len(img)==238

    def test_image_various(self):
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
        shapes = [(32561, 15),
         (352, 3),
         (1000, 4),
         (891, 12),
         (649, 33),
         (128, 27),
         (4674, 9),
         (392, 8),
         (4674, 9),
         (999, 8),
         (392, 8),
         (1000, 5),
         (10000, 10),
         (569, 30),
         (2522, 2),
         (68, 3),
         (1472, 20),
         (6556, 2),
         (150, 3),
         (387588, 2),
         (59400, 41),
         (24611, 8),
         (244, 7),
         (10000, 14),
         ]

        for i, data in enumerate(datasets):
            df = dz.get(data=data)
            assert df.shape==shapes[i]
