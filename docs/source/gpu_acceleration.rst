GPU Acceleration 
================

The toolbox is programmed in `Pytorch <https://pytorch.org>`_ and can be run on a GPU, which is especially useful for model training. 
With the new version of Pytorch, the framework can utilize both the NVIDIA and Apple M1/M2 chip GPU.

.. sourcecode:: python

	pt.set_default_tensor_type(pt.cuda.FloatTensor if pt.cuda.is_available()
							   else pt.FloatTensor)

