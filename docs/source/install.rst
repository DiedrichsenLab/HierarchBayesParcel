Installation
============

Dependencies
------------

The required dependencies to use the software are:

* python >= 3.6,
* setuptools
* numpy >= 1.16
* nibabel >= 2.5
* pandas >= 0.24
* matplotlib >= 1.5.1

Install over pip
----------------

This project is currently not support pip install

Install for developers
----------------------

Alternatively you can also fork or clone the repository at https://github.com/DiedrichsenLab/HierarchBayesParcel to a desired location (DIR). Simply include the lines::

    PYTHONPATH=/DIR/SUITPy:${PYTHONPATH}
    export PYTHONPATH

To your ``.bash.profile`` or other shell startup file.