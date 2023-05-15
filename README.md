HierarchBayesParcel: A Hierarchical Bayesian Brain Parcellation Framework for fusion of functional imaging datasets
====
Diedrichsen Lab, Western University

A Bayesian framework for individual brain organization across a number of different data. 
The Model is partitioned into a model that determines the probability of the spatial arrangement of 
regions in each subject s, $p(\mathbf{U}^{(s)};\theta_A)$ and the probability of observing a set of 
data at each given brain location. We introduce the Markov property that the observations are mutually 
independent, given the spatial arrangement.

Dependencies
------
### Packages
This project depends on several third party libraries, including:

[numpy](https://numpy.org/) (version>=1.22.2)\
[PyTorch](https://pytorch.org/) (version>=1.10.1 + CUDA enabled)\
[nilearn](https://nilearn.github.io/stable/index.html) (version>=0.9.0)\
[nibabel](https://nipy.org/nibabel/) (version>=3.2.0)\
[nitools](https://nitools.readthedocs.io/en/latest/)

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

Usage
------
### Overview

### Arrangement Model Class

### Emission Model Class

### Full Model Class

