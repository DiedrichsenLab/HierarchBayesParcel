Installation
============

Dependencies
------------

The required dependencies to use the software are:

* python >= 3.6,
* numpy >= 1.16
* nibabel >= 2.5
* nitools >= 1.0.0
* pandas
* matplotlib >= 1.5.1
* torch 

```
pip install numpy nibabel neuroimagingtools matplotlib pandas torch
```

To run some of the example scripts, you may also need to clone  
* [Functional_Fusion repository](https://github.com/DiedrichsenLab/Functional_Fusion) 

Installation repository
-----------------------
Fork or clone the repository at https://github.com/DiedrichsenLab/HierarchBayesParcel to a desired location (DIR). Simply include the lines::

    PYTHONPATH=/DIR/HierarchBayesParcel:${PYTHONPATH}
    export PYTHONPATH

To your ``.bash.profile`` or other shell startup file.

Installation over pip is not yet available. 