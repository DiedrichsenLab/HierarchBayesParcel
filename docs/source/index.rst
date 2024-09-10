.. HierarchBayesParcel documentation master file, created by
   sphinx-quickstart on Mon Jan  8 14:54:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HierarchBayesParcel's documentation
===================================

This repository implements the hierarchical Bayesian framework for learning brain parcellations across task-based and resting-state fMRI datasets.
The technical details are described in the following `paper <https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1>`_, and we have applied the framework to generate a new probabilistic `atlas of the human cerebellum <https://www.biorxiv.org/content/10.1101/2023.09.14.557689v2>`_.

The code for this framework is openly available. You can use this repository to:

* Learn new probabilistic brain parcellations across multiple fMRI datasets using other datasets for different brain structures.
* Use existing probabilistic atlases to obtain individualized brain parcellations for new subjects through the optimal integration of individual localizer data and the group atlas.



Diedrichsen Lab, Western University


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   overview.rst
   atlas_training_example
   indiv_parcel.rst
   math.rst
   gpu_acceleration.rst
   reference.rst
   literature.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Repository
----------
GitHub repository link: https://github.com/DiedrichsenLab/HierarchBayesParcel

License
-------
Please find out our development license (MIT) in ``LICENSE`` file.

Bug reports
-----------
For any problems and questions, please use the issues page on this repository. https://github.com/DiedrichsenLab/HierarchBayesParcel/issues.  We will endeavour to answer quickly.
