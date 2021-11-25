from arrangements import ArrangeIndependent
from emissions import MixGaussianExponential
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
import sys
sys.path.insert(0, "D:/python_workspace/")


class FullModel:
    def __init__(self, arrange, emission):
        self.arrange = arrange
        self.emission = emission
    
    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y
    
    def fit_em(self, iter, tol):
        """ Do the real EM algorithm for the complete log likelihood for the
        combination of the arrangement model and emission model

        :param Y: the data to passing
        :param iter: the maximum iteration number
        :param tol: the delta

        :return: the resulting parameters theta and the log likelihood
        """
        # Initialize the tracking 
        ll = []
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        for i in range(iter): 
            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood 
            ll.append(np.sum(Uhat * emloglik) + np.sum(ll_A))
            # Updates the parameters 
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(Uhat)
            theta[i, :] = np.concatenate([self.emission.get_params(), self.arrange.get_params()])
        print('debug here')
        return self, ll


def _fit_full(Y):
    pass


def _simulate_full():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionT = MixGaussianExponential(K=5, N=40, P=100)

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixGaussianExponential(K=5, N=40, P=100, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    theta = FullModel(arrangeM, emissionM).fit_em(iter=100, tol=1e-6)


if __name__ == '__main__':
    _simulate_full()

