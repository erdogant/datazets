# --------------------------------------------------
# Name        : datazets.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/datazets
# Licence     : See licences
# --------------------------------------------------

import os
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse
import logging
import zipfile
import fnmatch

logger = logging.getLogger('')
for handler in logger.handlers[:]:  # get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[datazets] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger(__name__)


# %% Import example dataset from github.
def get(data=None, url=None, sep=',', verbose='info', **args):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Multicollinear data sets:
            * 'titanic'
            * 'student'
            * 'fifa'
            * 'waterpump'
            * 'DS_salaries'
            * 'energy'
        synthetic discrete
            * 'random_discrete'
        Discrete data sets:
            * 'sprinkler'
            * 'stormofswords'
            * 'census_income'
            * 'ads'
        Continous data sets:
            * 'breast_cancer'
            * 'cancer'
            * 'auto_mpg'
        Image data sets:
            * 'flowers'
            * 'faces'
            * 'mnist'
            * 'scenes'
            * 'digits'
        Text data sets:
            * 'retail'
        Time series:
            * 'bitcoin'
            * 'meta'
    url : str
        url link to to dataset.
	verbose : int, (default: 20)
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    Examples
    --------
    >>> # Import directly from url
    >>> url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    >>> df = import_example(url=url)
    >>>

    References
    ----------
    * Census Income. (1996). UCI Machine Learning Repository. https://doi.org/10.24432/C5S595.

    """
    set_logger(verbose=verbose)
    if data is None and url is None:
        logger.error('Input parameter <data> or <url> should be used.')
        return None

    # Get and Set data information
    file_data, url, datatype, sep = _set_names_and_url(data, url)
    logger.info('Import dataset [%s]' %(file_data))

    if url is None:
        logger.info('Nothing to download.')
        return None

    # Import dataset
    if datatype=='image':
        df = get_images(file_data, url)
    elif datatype=='synthetic':
        df = _generate_data(file_data, **args)
    else:
        PATH_TO_DATA = get_data(file_data, url)
        df = pd.read_csv(PATH_TO_DATA, sep=sep)

    # Return
    return df


# %%
def _generate_data(file_data, **args):
    df = None
    # Generate discrete random data
    if file_data=='random_discrete.zip':
        n = args.get('n', None) if args.get('n', None) is not None else 1000
        df = pd.DataFrame(np.random.randint(low=0, high=2, size=(n, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    return df


# %%
def get_images(data, url=None, curpath=None):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        * 'flowers'
        * 'faces'
        * 'mnist'
        * 'scenes'
    url : str
        url link to to dataset.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if data=='faces.zip':
        from sklearn.datasets import fetch_olivetti_faces
        X = fetch_olivetti_faces()
        return X['data']
    elif data=='mnist.zip':
        from sklearn.datasets import load_digits
        digits = load_digits(n_class=10)
        return digits.data

    # Collect data
    PATH_TO_DATA = get_data(data, url)
    # Unzip
    dirpath = unzip(PATH_TO_DATA)
    # Import local dataset
    image_files = listdir(dirpath)
    # Return
    return image_files


# %%
def _set_names_and_url(data, url):
    sep=';'
    datatype=None

    # Set names
    if data is not None:
        # Set datatype for imges
        if data=='flowers' or data=='faces' or data=='mnist' or data=='scenes':
            datatype='image'
        elif data=='random_discrete':
            datatype='synthetic'

        # Rename to correct filename
        if data=='flowers': data = 'flower_images'
        if data=='titanic': data, sep ='titanic_train', ','
        if data=='student': data, sep ='student_train', ','
        if data=='cancer': data, sep = 'cancer_dataset', ','
        if data=='fifa': data, sep = 'FIFA_2018', ','
        if data=='ads': data, sep = 'ads_data', ','
        if data=='bitcoin': data, sep = 'BTCUSDT', ','
        if data=='meta': data, sep = 'facebook_stocks', ','
        if data=='digits': sep = ','
        if data=='DS_salaries': sep = ','
        if data=='energy': data, sep = 'energy_source_target_value', ','
        if data=='breast_cancer': data = 'breast_cancer_dataset'
        if data=='retail': data = 'marketing_data_online_retail_small'
        if data=='auto_mpg': sep = ','
        if data=='stormofswords': sep = ','
        if data=='census_income': sep = ','

        # Set the url
        url = 'https://erdogant.github.io/datasets/' + data + '.zip'

    # Get filename from url
    file_data = wget.filename_from_url(url)

    return file_data, url, datatype, sep


# %% Extract basename from path
def basename(label):
    """Extract basename from path."""
    return os.path.basename(label)

# %% Recursively list files from directory
def listdir(dirpath, ext=['png', 'tiff', 'jpg'], black_list=None):
    """Collect recursive images from path.

    Parameters
    ----------
    dirpath : str
        Path to directory; "/tmp" or "c://temp/"
    ext : list, default: ['png','tiff','jpg']
        extentions to collect form directories.
    black_list : list, (default: ['undouble'])
        Exclude directory with all subdirectories from processing.

    Returns
    -------
    getfiles : list of str.
        Full pathnames to images.

    Example
    -------
    >>> import clustimage as cl
    >>> pathnames = cl.listdir('c://temp//flower_images')

    """
    if isinstance(ext, str): ext = [ext]
    if not isinstance('dirpath', str): raise Exception(print('Error: "dirpath" should be of type string.'))
    if not os.path.isdir(dirpath): raise Exception(print('Error: The directory can not be found: %s.' %dirpath))

    getfiles = []
    for root, _, filenames in os.walk(dirpath):
        # Check if the (sub)directory is black listed
        bl_found = np.isin(os.path.split(root)[1], black_list)
        if (black_list is None) or (not bl_found):
            for iext in ext:
                for filename in fnmatch.filter(filenames, '*.' +iext):
                    getfiles.append(os.path.join(root, filename))
        else:
            logger.info('Excluded: <%s>' %(root))
    logger.info('[%s] files are collected recursively from path: [%s]', len(getfiles), dirpath)
    return getfiles


# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url):
        """Return filename."""
        return os.path.basename(url)

    def download(url, writepath):
        """Download.

        Parameters
        ----------
        url : str.
            Internet source.
        writepath : str.
            Directory to write the file.

        Returns
        -------
        None.

        """
        r = requests.get(url, stream=True)
        with open(writepath, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)


# %% unzip
def unzip(path_to_zip):
    """Unzip files.

    Parameters
    ----------
    path_to_zip : str
        Path of the zip file.

    Returns
    -------
    getpath : str
        Path containing the unzipped files.

    Example
    -------
    >>> import clustimage as cl
    >>> dirpath = cl.unzip('c://temp//flower_images.zip')

    """
    getpath = None
    if path_to_zip[-4:]=='.zip':
        if not os.path.isdir(path_to_zip):
            logger.info('Extracting files..')
            pathname, _ = os.path.split(path_to_zip)
            # Unzip
            zip_ref = zipfile.ZipFile(path_to_zip, 'r')
            zip_ref.extractall(pathname)
            zip_ref.close()
            getpath = path_to_zip.replace('.zip', '')
            if not os.path.isdir(getpath):
                logger.error('Extraction failed.')
                getpath = None
    else:
        logger.warning('Input is not a zip file: [%s]', path_to_zip)
    # Return
    return getpath


# %%
def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %%
def get_data(file_data, url):
    # Create directory
    PATH_TO_DATA = _makedir(file_data, url)
    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading [%s] dataset from github source..' %(file_data))
        _download_data(url, PATH_TO_DATA)
    return PATH_TO_DATA


def _makedir(file_data, url):
    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # file_data = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, file_data)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)
    return PATH_TO_DATA

def _download_data(url, PATH_TO_DATA):
    wget.download(url, PATH_TO_DATA)

