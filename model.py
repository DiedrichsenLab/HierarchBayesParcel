# Example Models
import numpy as np

class Model:
    """Abstract model class for vectorization of free parameters
    Automatically sets param_size and nparams
    """
    def set_param_list(self,param_list=[]):
        self.param_list=param_list
        self.param_size=[]
        self.nparams = 0
        for s in param_list:
            if isinstance(self[s],np.ndarray):
                self.param_size.append(self[s].shape)
                self.nparams= self.nparams + np.prod(self[s].shape)
            else:
                self.param_size.append(1) # Scalar
                self.nparams = self.nparams + 1

    def get_params(self):
        """Returns the vectorized version of the parameters
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        theta = np.empty((self.nparams,))
        start = 0 # Current index
        for s,t in zip(self.param_list,self.param_size):
            end = start + np.prod(t)
            if type(t) is tuple:
                theta[start:end] = self[s].flatten()
                start = end
            else:
                theta[start]= self[s] # Scalar
                start = start+1
        return theta

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        start = 0 # Current index
        for s,t in zip(self.param_list,self.param_size):
            end = start + np.prod(t)
            if type(t) is tuple:
                self[s]=theta[start:end].reshape(t)
                start = end
            else:
                self[s] = theta[start] # Scalar
                start = start+1
        pass

    def get_param_names(self):
        names=[]
        for s,t in zip(self.param_list,self.param_size):
            end = start + np.prod(t)
            if type(t) is tuple:
                self[s]=theta[start:end].reshape(t)
                start = end
            else:
                self[s] = theta[start] # Scalar
                start = start+1
        pass

    def get_param_indices(self):

