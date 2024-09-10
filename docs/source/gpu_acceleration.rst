GPU Acceleration 
================

The toolbox is programmed in `Pytorch <https://pytorch.org>`_ and can be run on a GPU, which is especially useful for model training. 
With the new version of Pytorch, the framework can utilize both the NVIDIA and Apple M1/M2 chip GPU.


To utilize GPU, you need to set the default device and default tensor type in your code.

.. sourcecode:: python

	pt.set_default_tensor_type(pt.cuda.FloatTensor if pt.cuda.is_available()
							   else pt.FloatTensor)

