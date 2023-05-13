import unittest
import datazets as ds

class Testdatazets(unittest.TestCase):

    def test_import_example(self):
        df = ds.get(data='random_discrete', n=1000)
        assert df.shape==(1000, 5)
        df = ds.get(data='random_discrete', n=10)
        assert df.shape==(10, 5)

    def test_image_datasets(self):
        img = ds.get(data='flowers', verbose=0)
        assert len(img)==214
        img = ds.get(data='faces', verbose=0)
        assert img.shape==(400, 4096)
        img = ds.get(data='mnist', verbose=0)
        assert img.shape==(1797, 64)
        img = ds.get(data='scenes', verbose=0)
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
                    'retail',
                    'auto_mpg',
                    'random_discrete',
                    'ads',
                    'breast_cancer',
                    'bitcoin',
                    'energy',
                    'meta',
                    ]
        shapes = [(32561, 15),
         (352, 3),
         (1000, 1),
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
         (1472, 20)]

        for i, data in enumerate(datasets):
            df = ds.get(data=data)
            assert df.shape==shapes[i]
