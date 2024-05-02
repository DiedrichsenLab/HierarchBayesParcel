HierarchBayesParcel: A Hierarchical Bayesian Brain Parcellation Framework for fusion of functional imaging datasets
====
Diedrichsen Lab, Western University

This repository implements the hierarchical Bayesian framework for learning brain parcellations across task-based and resting-state fMRI datasets. 
The technical details are described in the following 
[paper](https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1), and we have applied the framework to generate a new probabilistic [atlas of the human cerebellum](https://www.biorxiv.org/content/10.1101/2023.09.14.557689v2). 

The code for this framework is openly available. You can use this repository to 
* Learn new probabilistic brain parcellations across multiple fMRI datasets using other datasets for different brain structures. 
* Use existing probabilistic atlases to obtain individualized brain parcellations for new subjects through the optimal integration of individual localizer data and the group atlas. 

Reference
------
* Zhi, D., Shahshahani, L., Nettekoven, C., Pinho, A. L. Bzdok, D., Diedrichsen, J., (2023). 
"A hierarchical Bayesian brain parcellation framework for fusion of functional imaging datasets". 
BioRxiv. [[link]](https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1)
* Nettekoven, C., Zhi, D., Ladan, S., Pinho, A., Saadon, N., Buckner, R., Diedrichsen, J. (2023). A hierarchical atlas of the human cerebellum for functional precision mapping. BioRviv. [[link]](https://www.biorxiv.org/content/10.1101/2023.09.14.557689v2)

Dependencies
------------
### Packages
This project depends on several third party libraries, including:

[numpy](https://numpy.org/) (version>=1.22.2)\
[PyTorch](https://pytorch.org/) (version>=1.10.1)\
[nilearn](https://nilearn.github.io/stable/index.html) (version>=0.9.0)\
[nibabel](https://nipy.org/nibabel/) (version>=3.2.0)\
[nitools](https://nitools.readthedocs.io/en/latest/) (version>=1.0.0)

Installation
------------
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

Scope and related repositories
------------------------------
This repository implements the computational side of the hierarchical Bayesian 
parcellation framework. In the interest of making this toolbox as modular as possible, we do not provide the 
tools to extract the individual subject data in a group atlas space or to project the parcellations back into the volume of the surface. 

For the illustrative examples, we are using the 
[Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion)
repository to import the preprocessed data as the input to the framework.

The analyzes and simulations reported in Zhi et al. (2023), can be replicated using the [FusionModel](https://github.com/DiedrichsenLab/FusionModel) repository. 


Documentation
------
For detailed documentation see: [https://hierarchbayesparcel.readthedocs.io/](https://hierarchbayesparcel.readthedocs.io/)

License
------
Please find out our development license (MIT) in `LICENSE` file.

Bug reports
------
For any problems and questions, please use the issues page on this repository. We will endeavour to answer quickly. 
