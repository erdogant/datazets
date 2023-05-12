import unittest
import datazets as datazets

class Testdatazets(unittest.TestCase):

    def test_import_example(self):
        cl = clusteval()
        sizes = [(1000, 4), (891, 12), (649, 33), (128, 27), (4674, 9), (14850, 40), (999, 8)]
        datasets = ['sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail']
        for data, size in zip(datasets, sizes):
            df = cl.import_example(data=data)
            assert df.shape==size

	def test_plot(self):
		pass
