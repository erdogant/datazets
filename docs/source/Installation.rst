Installation
################

Create environment
**********************

If desired, install ``datazets`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_datazets python=3.8
    conda activate env_datazets


Pypi
**********************

.. code-block:: console

    # Install from Pypi:
    pip install datazets

    # Force update to latest version
    pip install -U datazets


Github source
************************************

.. code-block:: console

    # Install directly from github
    pip install git+https://github.com/erdogant/datazets


Uninstalling
################

Remove environment
**********************

.. code-block:: console

   # List all the active environments. datazets should be listed.
   conda env list

   # Remove the datazets environment
   conda env remove --name datazets

   # List all the active environments. datazets should be absent.
   conda env list


Remove installation
**********************

Note that the removal of the environment will also remove the ``datazets`` installation.

.. code-block:: console

    # Install from Pypi:
    pip uninstall datazets

