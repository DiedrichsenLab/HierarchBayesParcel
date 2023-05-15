#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Base model class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
from copy import copy, deepcopy

class Model:
    """Abstract class for models
    Implements two behaviors:
        - param_list: These is the list of free parameters.
            the class allows vectorization of free parameters, sets
            param_size and nparams automatically, and tells you
            where to find the parameters in the vector
        - tmp_list: List of data-specific attributes that should not be copied
            or saved. when deep_copy is called on a model, these items will only be copied by reference (id), but not copied in memory.
            When clear is call, the reference to these items will be deleted (but not the data itself, if it is referenced from somewhere else).
    """
    def __deepcopy__(self, memo):
        """ Overwrites deepcopy behavior such that members of tmp_list are not deepcopied, but only shallow copied (by reference). One important example is the data attached to emission models. This saves memory

        Args:
            memo (dictionary): already copied objects to avoid recursion

        Returns:
            _type_: _description_
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self.tmp_list:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def clear(self):
        """Removes any member of tmp_list from the model
        This is important when saving model fits.
        """
        for att in self.tmp_list:
            if hasattr(self,att):
                delattr(self,att)

    def move_to(self, device='cpu'):
        """move all torch.Tensor object in Model
           class to the targe device
       Args:
        M: a FullMultiModel
           object
        device: the target device to store the tensor
                default - 'cpu'
        Returns:
            None
        Notes:
            This function works generally for all Models but is not recursive
       """
        for attr, value in self.__dict__.items():
            if isinstance(value, pt.Tensor):
                vars(self)[attr] = value.to(device)

    def set_param_list(self, param_list=[]):
        """Initializes the parameter list for a model

        Args:
            param_list (list, optional): Names of parameters
        """
        self.param_list = param_list
        self.param_size = []
        self.param_offset = [0, ]
        self.nparams = 0
        for s in param_list:
            if isinstance(vars(self)[s], np.ndarray) or isinstance(vars(self)[s], pt.Tensor):
                isTensor = isinstance(vars(self)[s], pt.Tensor)
                if (isTensor and vars(self)[s].numel() > 1) or (not isTensor and vars(self)[s].size > 1):  # vector
                    self.param_size.append(vars(self)[s].shape)
                    self.nparams = self.nparams + np.int(np.prod(vars(self)[s].shape))
                else:
                    self.param_size.append(1)  # Tensor Scalar
                    self.nparams = self.nparams + 1
                self.param_offset.append(self.nparams)
            elif np.isscalar(vars(self)[s]):
                self.param_size.append(1)  # numpy scalar
                self.nparams = self.nparams + 1
                self.param_offset.append(self.nparams)
            else:
                raise ValueError("The initialized model parameters must be a numpy.array or torch.tensor!")

    def get_params(self):
        """Returns the vectorized version of the parameters
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        theta = pt.empty((self.nparams,))
        for i, s in enumerate(self.param_list):
            if type(self.param_size[i]) is tuple:  # ndarray with more than 1 element
                if isinstance(vars(self)[s], np.ndarray):
                    theta[self.param_offset[i]:self.param_offset[i + 1]] = pt.tensor(vars(self)[s].flatten(), dtype=pt.get_default_dtype())
                else:
                    theta[self.param_offset[i]:self.param_offset[i + 1]] = vars(self)[s].flatten()
            elif type(self.param_size[i]) is pt.Size:  # tensor with more than 1 element
                theta[self.param_offset[i]:self.param_offset[i + 1]] = vars(self)[s].flatten()
            else:
                theta[self.param_offset[i]] = vars(self)[s]  # Scalar
        return theta

    def set_params(self, theta):
        """ Sets the parameters from a vector
        Args:
            theta (numpy.ndarray or torch.tensor): Input parameters as vector.
        """
        if type(theta) is np.ndarray:  # Convert input theta to tensor if it is ndarray
            theta = pt.tensor(theta, dtype=pt.get_default_dtype())

        for i, s in enumerate(self.param_list):
            if (type(self.param_size[i]) is tuple) or (type(self.param_size[i]) is pt.Size):
                vars(self)[s] = theta[self.param_offset[i]:self.param_offset[i + 1]].reshape(self.param_size[i])
            else:
                vars(self)[s] = theta[self.param_offset[i]]  # Scalar
        pass

    def get_param_indices(self, name):
        """Returns the indices for specific set of parameters
        Args:
            names (str): parameter names to returns indices for
        Returns:
            indices (nparray)
        """
        if name not in self.param_list:
            raise NameError(f'Parameter {name} not in param list')
        ind = self.param_list.index(name)
        return np.arange(self.param_offset[ind], self.param_offset[ind+1])
