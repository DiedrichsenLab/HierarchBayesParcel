Overview
========

.. image:: _static/0_fusion_model.png
	:width: 680
	:height: 460
	:alt: fusion_model
	:align: center

The Hierarchical Bayesian Parcellation framework is designed to derive a probabilistic brain parcellation from multiple fMRI datasets. 
The central quantity that the framework is the brain parcellation for each individual subject, :math:`<\mathbf{U}^s>`. 

The framework is partitioned
into a spatial *arrangement model*, :math:`p(\mathbf{U}|\theta_A)`, the probability of how likely a parcel
assignment is within the studied population, and a collection of dataset-specific *emission models*, 
:math:`p(\mathbf{Y}^{s,n}| \mathbf{U}^s;\theta_{En})`, the probability of each observed dataset given the 
individual brain parcellation. The individual brain parcellations can be then estimated as the posterior probability of :math:`\mathbf{U}^s`, given the arrangement and the emission models. The integration, message passing, and parameter estimation are implemented in the ``FullMultiModel`` class. 

The framework can be used for two main purposes: 

* To learn a new probabilistic brain parcellations across multiple fMRI datasets. For this purpose, you will build a ``FullMultiModel`` from one arrangement model and multiple emission models. The parameters of both arrangement model and emission models will be simultaneously estimated using the EM-algorithm. For more information, see the :ref:`training_example`. We show that the integration of multiple datasets can dramatically improve the quality of brain parcellations.
* To use an existing probabilistic atlases to obtain individualized brain parcellations for new subjects through the optimal integration of individual localizer data and the group atlas. For this purpose, you will build a ``FullMultiModel`` from one arrangement model and (typically) a single emission model. The parameters of the emission model will be estimated using the EM-algorithm, while the arrangement model will be frozen. For more information, see the :ref:`individual_parcellation`. In Zhi et al. (2023), we show that the integration of as little as 10 min of individual data substantially improves the predictive power of the brain parcellation. 


Arrangement Model
-----------------

**arrangement_model.py** contains the implementation of different arrangement model classes that being used in the framework along with the helper functions. The main active spatial arrangement models are:

* ``ArrangementModel``: The base class for all arrangement models, which inherits from the ``Model`` class.

* ``ArrangeIndependent``: The Independent arrangement model, which assumes that the brain locations are spatially independent. The current spatial arrangement model being used in the Zhi et al. (2023).

* ``ArrangeIndependentSymmetric``: The Independent symmetric arrangement model, which assumes that corresponding brain locations in the left and right hemisphere are assigned to corresponding parcels. While the boundaries are forced to be symmetric, the functional profiles for the left and right parcel are being estimated separately, which allows the study of functional lateralization (Nettekoven et al., 2024).

* ``ArrangeIndependentSeparateHem``: An spatially indpendent arrangement model, which also constraints that there are matched pairs of parcels, which seperately model regions in the left and right hemisphere. In contrast to the symmetric model, the boundaries are not constrained to be the same across the hemispheres. Used for the asymmetric version of our cerebellar atlas (Nettekoven et al., 2024). 
   
* ``PottsModel``: A potts model (Markov random field on multinomial variable) with K possible states. In our framework, we use this arrangement model only to simulate realistically looking brain parcellation maps. Due to computational requirement of Gibbs sampling, it is currently not used for inference. 

* ``cmpRBM``: A convolutional multinomial (categorial) restricted Boltzman machine for learning of brain parcellations for probabilistic input. It uses variational stochastic maximum likelihood for the learning. Described in the 4th chapter of the Da Zhi's dissertation (2023).


Emission Model
--------------

**emission_model.py** contains the implementation of different emission model classes to calculate the data likelihood given the individual brain parcellation. The main active emission models are:

* ``EmissionModel``: The base class for all emission models, which inherits from the ``Model`` class.

* ``MultiNomial``: An emission model which can integrated discrete data, such as a winner-take-all map of some characteristic of an individual subject. 

* ``MixGaussian``: The Gaussian mixture emission model with isotropic noise.

* ``MixVMF``: The von Mises-Fisher mixture emission model, which assumes that the data is projected on the sphere with unit length. This is the recommended emission model both for task-based and resting-state fMRI data.

Full Model
----------

**full_model.py** contains the implementation of the ``FullMultiModel`` class that combines the arrangement and emission models. The class have the learning and inference details for different arrangment and emission models combination. The main active learning methods are:

* ``fit_em_ninits()``: which runs the EM-algorithm on a full model starting with ``n_inits`` multiple random initialization values and escape from local maxima by selecting the model with the highest likelihood after first few iterations. Check the paper for more mathmatical inference details.

* ``fit_sml()``: which runs a stochastic maximum likelihood algorithm on a full model when the posterior given the data likelihood and arrangment model parameters is intracted. The emission model is still assumed to have E-step and Mstep. The arrangement model is has a postive and negative phase estep, and a gradient M-step to perform the contrastive divergence algorithm.

Scope and related repositories
------------------------------
This repository implements the computational side of the hierarchical Bayesian 
parcellation framework. In the interest of making this toolbox as modular as possible, we do not provide the 
tools to extract the individual subject data in a group atlas space or to project the parcellations back into the volume of the surface. 

For the illustrative examples, we are using the 
[Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion)
repository to import the preprocessed data as the input to the framework.

The analyzes and simulations reported in Zhi et al. (2023), can be replicated using the [FusionModel](https://github.com/DiedrichsenLab/FusionModel) repository. 


License
-------
Please find out our development license (MIT) in ``LICENSE`` file.

Bug reports
-----------
For any problems and questions, please use the issues page on this repository. We will endeavour to answer quickly. 
