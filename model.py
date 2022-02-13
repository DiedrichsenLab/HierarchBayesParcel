# Example Models
import numpy as np

class Model:
    """Abstract model class for vectorization of free parameters
    Automatically sets param_size and nparams
    """
    def set_param_list(self,param_list=[]):
        self.param_list=param_list
        self.param_size=[]
        self.param_offset=[0,]
        self.nparams = 0
        for s in param_list:
            if isinstance(vars(self)[s],np.ndarray):
                self.param_size.append(vars(self)[s].shape)
                self.nparams= self.nparams + np.prod(vars(self)[s].shape)
            else:
                self.param_size.append(1) # Scalar
                self.nparams = self.nparams + 1
            self.param_offset.append(self.nparams)

    def get_params(self):
        """Returns the vectorized version of the parameters
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        theta = np.empty((self.nparams,))
        for i,s in enumerate(self.param_list):
            if type(self.param_size[i]) is tuple:
                theta[self.param_offset[i]:self.param_offset[i+1]] = vars(self)[s].flatten()
            else:
                theta[self.param_offset[i]]= vars(self)[s] # Scalar
        return theta

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        for i,s in enumerate(self.param_list):
            if type(self.param_size[i]) is tuple:
                vars(self)[s]=theta[self.param_offset[i]:self.param_offset[i+1]].reshape(self.param_size[i])
            else:
                vars(self)[s] = theta[self.param_offset[i]]  # Scalar
        pass

    def get_param_indices(self,name):
        """Returns the indices for specific set of parameters 
        Args:
            names (str): parameter names to returns indices for 
        Returns: 
            indices (nparray)
        """
        if name not in self.param_list:
            raise NameError(f'Parameter {name} not in param list')
        ind = self.param_list.index(name)
        return np.arange(self.param_offset[ind],self.param_offset[ind+1])

"""
Testing: 
if __name__ == '__main__':
    M = Model()
    M.logpi = np.arange(0,12).reshape(4,3)
    M.mean = np.arange(0,5)
    M.std = 1
    M.set_param_list(['logpi','mean','std'])
    pass 
"""