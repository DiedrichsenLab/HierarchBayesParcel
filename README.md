HierarchBayesParcel: A Hierarchical Bayesian Brain Parcellation Framework for fusion of functional imaging datasets
====
Diedrichsen Lab, Western University

This repository hosts the computation models and implementatoin of a hierarchical Bayesian framework for learning
brain organization across a number of different fMRI data that described in this [paper](https://www.diedrichsenlab.org/). 
The framework is partitioned into a spatial arrangement model, $p(\mathbf{U}|\theta_A)$, 
the probability of how likely a parcel assignment is within the studied population, and a collection of dataset-specific
\textit{emission models}, $p(\mathbf{Y}^{s,n}| \boldsymbol{\theta}_{En})$, the probability of each observed dataset given
the individual brain parcellation. This distributed structure allows the parameters of the model, 
$\left( \boldsymbol{\theta}_A,\boldsymbol{\theta}_{E1},..\right)$ to be estimated using a message-passing algorithm 
between the different model components. For more details, please checkout the preprint.

Reference
------
* Zhi, D., Shahshahani, L., Nettekovena, C., Pinho, A. L. Bzdok, D., Diedrichsen, J., (2023). "A hierarchical Bayesian 
brain parcellation framework for fusion of functional imaging datasets". BioRxiv. 
[[link]](https://www.diedrichsenlab.org/)

Dependencies
------
### Packages
This project depends on several third party libraries, including:

[numpy](https://numpy.org/) (version>=1.22.2)\
[PyTorch](https://pytorch.org/) (version>=1.10.1 + CUDA enabled)\
[nilearn](https://nilearn.github.io/stable/index.html) (version>=0.9.0)\
[nibabel](https://nipy.org/nibabel/) (version>=3.2.0)\
[nitools](https://nitools.readthedocs.io/en/latest/) (version=1.0.0)

### Installations
```
pip install numpy nilearn nibabel neuroimagingtools nilearn
```

Or you can install the package manually from their original binary source as above links.

Once you clone the functional fusion repository, you need to add it to your PYTHONPATH, so you can
import the functionality. Add these lines to your .bash_profile, .bash_rc .zsh_profile file... 

```
PYTHONPATH=<your_repo_absolute_path>:${PYTHONPATH}
export PYTHONPATH
```

Class descriptions
------
### Overview

[comment]: <> (![ScreenShot]&#40;doc/fusion_model.png&#41;)

<img src="doc/fusion_model.png" alt="ScreenShot" width=60%>

One important barrier in the development of complex models of human brain organization is the lack of a large and comprehensive task-based neuro-imaging dataset.
Therefore, current atlases of functional brain organization are mainly based on single and homogeneous resting-state datasets. Here, we propose a hierarchical Bayesian
framework that can learn a probabilistically defined brain parcellation across numerous task-based and resting-state datasets, exploiting their combined strengths. The
framework is partitioned into a spatial arrangement model that defines the probability of a specific individual brain parcellation, and a set of dataset-specific emission models
that defines the probability of the observed data given the individual brain organization. We show that the framework optimally combines information from different
datasets to achieve a new population-based atlas of the human cerebellum. Furthermore, we demonstrate that, using only 10 min of individual data, the framework is
able to generate individual brain parcellations that outperform group atlases.


### Arrangement Model Class

### Emission Model Class

### Full Model Class

