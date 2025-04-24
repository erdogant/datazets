import logging
from datazets.datazets import (
    get,
    get_dataproperties,
    download_from_url,
    unzip,
    listdir,
    url2disk,
    check_logger,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.1'

# Setup root logger
_logger = logging.getLogger('datazets')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False

# module level doc-string
__doc__ = """
datazets
=====================================================================

Datazets is a python package to retrieve example data sets.

Example
-------
>>> # Import library
>>> import datazets as dz
>>> #
>>> # Import data set
>>> df = dz.get('titanic')
>>> #
>>> # Import from url
>>> url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
>>> df = dz.get(url=url, sep=',')

References
----------
https://github.com/erdogant/datazets

"""
