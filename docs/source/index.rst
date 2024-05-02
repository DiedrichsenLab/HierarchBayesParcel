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
   training.rst
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


References
==========

* Zhi, D., Shahshahani, L., Nettekoven, C., Pinho, A. L. Bzdok, D., Diedrichsen, J., (2023). A hierarchical Bayesian brain parcellation framework for fusion of functional imaging datasets. BioRxiv. `link <https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1>`_
* Nettekoven, C., Zhi, D., Ladan, S., Pinho, A., Saadon, N., Buckner, R., Diedrichsen, J. (2023). A hierarchical atlas of the human cerebellum for functional precision mapping. BioRviv. `link <https://www.biorxiv.org/content/10.1101/2023.09.14.557689v2>`_


GitHub repository link: https://github.com/DiedrichsenLab/HierarchBayesParcel