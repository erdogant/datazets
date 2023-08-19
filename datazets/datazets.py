"""Datazets is a python package to import well known example data sets."""
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
import logging
import zipfile
import fnmatch
from io import BytesIO
from urllib.parse import urlparse

logger = logging.getLogger('')
[logger.removeHandler(handler) for handler in logger.handlers[:]]
console = logging.StreamHandler()
formatter = logging.Formatter('[datazets] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger(__name__)


# %% Import example dataset.
def get(data=None, url=None, sep=',', verbose='info', overwrite=False, **args):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Multicollinear data sets:
            * 'titanic'
            * 'student'
            * 'fifa'
            * 'ds_salaries'
            * 'waterpump'
            * 'elections'
            * 'tips'
            * 'predictive_maintenance'
        source-target
            * 'energy'
            * 'stormofswords'
            * 'gameofthrones'
            * 'bigbang'
        time-series
            * 'occupancy'
            * 'ps_data'
            * 'gas_prices'
        synthetic discrete
            * 'random_discrete'
        Discrete data sets:
            * 'sprinkler'
            * 'census_income'
            * 'ads'
        Continous data sets:
            * 'breast_cancer'
            * 'cancer'
            * 'auto_mpg'
            * 'iris'
        Images:
            * 'faces'
            * 'mnist'
        Images (files):
            * 'southern_nebula'
            * 'flowers'
            * 'scenes'
            * 'cat_and_dog'
        Text data sets:
            * 'marketing_retail'
            * 'malicious_urls'
            * 'grocery_products'
        Time series:
            * 'bitcoin'
            * 'meta'
    url : str
        url link to to dataset.
    n : int
        Number of samples to generate in case of 'random_discrete'.
        Number of classes in case of 'mnist'
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
    >>> # Import library
    >>> import datazets as dz
    >>> #
    >>> # Import data set
    >>> df = dz.get('titanic')
    >>> #
    >>> # Import data set
    >>> img = dz.get('faces')
    >>> #
    >>> # Import from url
    >>> url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    >>> df = dz.get(url=url, sep=',')

    References
    ----------
    * Census Income. (1996). UCI Machine Learning Repository. https://doi.org/10.24432/C5S595.
    * stormofswords: https://github.com/melaniewalsh/sample-social-network-datasets/
    * gameofthrones: https://github.com/mathbeveridge/gameofthrones (CC BY 4.0) license.
    * energy: https://www.gov.uk/guidance/2050-pathways-analysis

    """
    set_logger(verbose=verbose)
    if data is None and url is None:
        logger.error('Input parameter <data> or <url> should be used.')
        return None

    # Get and Set data information
    dataproperties = get_dataproperties(data, sep, url)
    logger.info('Import dataset [%s]' %(dataproperties['input']))

    if dataproperties['url'] is None:
        logger.info('Nothing to download.')
        return None

    if overwrite and os.path.isfile(os.path.join(dataproperties['curpath'], dataproperties['filename'])):
        logger.info('Removing [%s] from disk.' %(dataproperties['input']))
        os.remove(os.path.join(dataproperties['curpath'], dataproperties['filename']))

    # Import dataset
    if dataproperties['type']=='files':
        df = _extract_files(dataproperties)
    elif dataproperties['type']=='synthetic':
        df = _generate_data(dataproperties['filename'], **args)
    elif dataproperties['type']=='DAG':
        df = _extract_files(dataproperties, targetdir=os.path.join(dataproperties['curpath'], dataproperties['input']), ext='*')
        # PATH_TO_DATA = download_from_url(dataproperties['filename'], url=dataproperties['url'], dataproperties=dataproperties)
    else:
        PATH_TO_DATA = download_from_url(dataproperties['filename'], url=dataproperties['url'])
        df = pd.read_csv(PATH_TO_DATA, sep=dataproperties['sep'])

    df = _set_dtypes(df, dataproperties)
    # Return
    return df


# %%
def _set_dtypes(df, dataproperties):
    if dataproperties['input']=='auto_mpg':
        df['mpg'] = df['mpg'].astype('float64')
        df['cylinders'] = df['cylinders'].astype('int64')
        df['displacement'] = df['displacement'].astype('float64')
        df['horsepower'] = df['horsepower'].astype('float64')
        df['weight'] = df['weight'].astype('float64')
        df['acceleration'] = df['acceleration'].astype('float64')
        df['model_year'] = df['model_year'].astype('int64')
        df['origin'] = df['origin'].astype('int64')
    return df


# %%
def _generate_data(urlname, **args):
    df = None
    # Generate discrete random data
    if urlname=='random_discrete':
        n = args.get('n', None) if args.get('n', None) is not None else 1000
        df = pd.DataFrame(np.random.randint(low=0, high=2, size=(n, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    return df


# %%
def _extract_files(dataproperties, **args):
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
    targetdir = args.get('targetdir', None) if args.get('targetdir', None) is not None else None
    ext = args.get('ext', None) if args.get('ext', None) is not None else ['png', 'tiff', 'jpg']

    # Collect data
    PATH_TO_DATA = download_from_url(dataproperties['filename'], url=dataproperties['url'])
    # Unzip
    dirpath = unzip(PATH_TO_DATA, targetdir=targetdir)
    # Import local dataset
    image_files = listdir(dirpath, ext=ext)
    # Return
    return image_files


# %% Get the correct url name.
def get_dataproperties(data, sep=None, url=None):
    inputname, datatype = data, None
    if data is not None:
        data = data.lower()
        # Set datatype for imges
        if data=='flowers' or data=='scenes' or data=='southern_nebula' or data=='cat_and_dog':
            datatype='files'
            if data=='flowers': data = 'flower_images.zip'
            if data=='scenes': data = 'scenes.zip'
            if data=='cat_and_dog': data = 'cat_and_dog.zip'
            if data=='southern_nebula': data = 'southern_nebula.zip'
        elif data=='alarm' or data=='andes' or data=='asia' or data=='sachs' or data=='water':
            datatype='DAG'
            if os.path.splitext(data)[1]=='':
                sep=','
                data = data + '.zip'
        elif data=='random_discrete':
            datatype='synthetic'
        else:
            datatype='various'
            # Rename to correct filename
            if data=='titanic': data, sep ='titanic_train.zip', ','
            if data=='ds_salaries': data, sep ='ds_salaries.zip', ','
            if data=='student': data, sep ='student_train.zip', ','
            if data=='cancer': data, sep = 'cancer_dataset.zip', ','
            if data=='fifa': data, sep = 'FIFA_2018.zip', ','
            if data=='ads': data, sep = 'ads_data.zip', ','
            if data=='bitcoin': data, sep = 'BTCUSDT.zip', ','
            if data=='meta': data, sep = 'facebook_stocks.zip', ','
            if data=='energy': data, sep = 'energy.zip', ','
            if data=='breast_cancer': data, sep = 'breast_cancer_dataset.zip', ';'
            if data=='marketing_retail': data, sep = 'marketing_data_online_retail_small.zip', ';'
            if data=='waterpump': data, sep = 'waterpump.zip', ';'
            if data=='elections': data, sep = 'USA_2016_elections.zip', ','
            if data=='occupancy': data, sep = 'UCI_Occupancy_Detection.zip', ','
            if data=='predictive_maintenance': data, sep = 'predictive_maintenance_ai4i2020.zip', ','
            if data=='iris': data, sep = 'iris_dataset.zip', ';'
            if data=='gas_prices': data, sep = 'Henry_Hub_Natural_Gas_Spot_Price.zip', ','
            if data=='grocery_products': data, sep = 'grocery_products_purchase.zip', ','
            if data=='bigbang': data, sep = 'bigbang.zip', ','
            # images
            if data=='faces': data, sep = 'olivetti_faces.zip', ';'
            if data=='mnist': data, sep = 'mnist_images.zip', ';'

            if os.path.splitext(data)[1]=='':
                sep=','
                data = data + '.zip'

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # Set the url
    dataproperties = {'input': inputname, 'filename': data, 'sep': sep, 'type': datatype, 'url': url, 'curpath': curpath}
    # Get filename from url
    if url is not None:
        dataproperties['filename'] = wget.filename_from_url(url, ext=False)
        dataproperties['input'] = dataproperties['filename']
    else:
        dataproperties['url'] = 'https://erdogant.github.io/datasets/' + dataproperties['filename']

    return dataproperties


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
                for filename in fnmatch.filter(filenames, '*.' + iext):
                    getfiles.append(os.path.join(root, filename))
        else:
            logger.info('Excluded: <%s>' %(root))
    logger.info('[%s] files are collected recursively from path: [%s]', len(getfiles), dirpath)
    return getfiles


# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url, ext=True):
        """Return filename."""
        urlname = os.path.basename(url)
        if not ext: _, ext = os.path.splitext(urlname)
        return urlname

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
def unzip(path_to_zip, targetdir=None):
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
            if targetdir is None:
                pathname, _ = os.path.split(path_to_zip)
            else:
                pathname = targetdir

            # Unzip
            zip_ref = zipfile.ZipFile(path_to_zip, 'r')
            zip_ref.extractall(pathname)
            zip_ref.close()
            getpath = path_to_zip.replace('.zip', '')
            if not os.path.isdir(getpath):
                logger.error('Extraction failed.')
                getpath = None
                raise Exception('Extraction failed')
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


# %% Extract basename from path
def basename(label):
    """Extract basename from path."""
    return os.path.basename(label)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


def download_from_url(filename, url, **args):
    """Url import function.

    Parameters
    ----------
    filename : string
        filename to create directory.
    url : string
        string.

    Returns
    -------
    PATH_TO_DATA : TYPE
        string.

    """
    # Create directory
    PATH_TO_DATA = _makedir(filename, url)
    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading [%s] dataset from github source..' %(filename))
        wget.download(url, PATH_TO_DATA)
    return PATH_TO_DATA


# %%
def _makedir(filename, url):
    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)
    return PATH_TO_DATA


# %% URL to disk
def url2disk(urls, save_dir):
    """Write url locations to disk.

    Images can also be imported from url locations.
    Each image is first downloaded and stored on a (specified) temp directory.
    In this example we will download 5 images from url locations. Note that url images and path locations can be combined.

    Parameters
    ----------
    urls : list
        list of url locations with image path.
    save_dir : str
        location to disk.

    Returns
    -------
    urls : list of str.
        list to url locations that are now stored on disk.

    Examples
    --------
    >>> # Init with default settings
    >>> import clustimage as cl
    >>>
    >>> # Importing the files files from disk, cleaning and pre-processing
    >>> url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png']
    >>>
    >>> # Import into model
    >>> results = cl.url2disk(url_to_images, r'c:/temp/out/')
    >>>

    """
    try:
        from PIL import Image
    except:
        logger.error('Could not import <pillow> library. Try to pip install first <pip install pillow>')

    if not isinstance(urls, list): urls = [urls]
    # Set filepath to the output of urls in case no url are used. Then the normal filepath is returned.
    filepath = urls.copy()
    idx_url = np.where(list(map(lambda x: x[0:4]=='http', filepath)))[0]
    if len(idx_url)>0:
        logger.info('[%.0d] urls are detected and stored on disk: [%s]' %(len(idx_url), save_dir))
    else:
        urls = None

    if not os.path.isdir(save_dir):
        logger.info('Create dir: [%s]' %(save_dir))
        os.mkdir(save_dir)

    for idx in idx_url:
        try:
            # Make connection to file
            response = requests.get(urls[idx])
            img = Image.open(BytesIO(response.content))
            # Get url
            url = urlparse(urls[idx])
            # Extract filename from url
            url_filename = os.path.basename(url.path)
            path_to_file = os.path.join(save_dir, url_filename)
            if os.path.isfile(path_to_file):
                logger.info('File already exists and is overwritten: [%s]' %(url.path))
            else:
                logger.info('Downloading [%s]' %(urls[idx]))
            # save a image using extension
            img.save(path_to_file)
            # Store new location
            filepath[idx] = path_to_file
        except:
            logger.warning('error downloading file from [%s]' %(urls[idx]))

    # Make dictionary output
    out = {'url': urls, 'pathnames': filepath}

    # Return
    return out
