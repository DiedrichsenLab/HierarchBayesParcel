.. HierarchBayesParcel documentation master file, created by
   sphinx-quickstart on Mon Jan  8 14:54:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HierarchBayesParcel's documentation
===============================================

HierarchBayesParcel is a Python project that hosts the computation models and implementatoin of a 
hierarchical Bayesian framework for learning brain organization across a number of different fMRI 
data that described in this [`paper <https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1>`_].

The framework is partitioned into a spatial *arrangement model*, :math:`p(\mathbf{U}|\theta_A)`, the 
probability of how likely a parcel assignment is within the studied population, and a collection of 
dataset-specific *emission models*, :math:`p(\mathbf{Y}^{s,n}| \mathbf{U}^s;\theta_{En})`, the probability 
of each observed dataset given the individual brain parcellation. This distributed structure allows 
the parameters of the model to be estimated using a message-passing algorithm between the different 
model components. For more details, please checkout the preprint.

Diedrichsen Lab, Western University


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   math.rst
   training.rst
   indiv_parcel.rst
   references.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========

* Zhi, D., Shahshahani, L., Nettekovena, C., Pinho, A. L. Bzdok, D., Diedrichsen, J., (2023). "A hierarchical Bayesian brain parcellation framework for fusion of functional imaging datasets". BioRxiv. [`link <https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1>`_]

* GitHub repository link: https://github.com/DiedrichsenLab/HierarchBayesParcel