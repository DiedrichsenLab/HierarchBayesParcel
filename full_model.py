from arrangements import ArrangeIndependent
from emissions import MixGaussian, MixGaussianGamma
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
import sys
# sys.path.insert(0, "D:/python_workspace/")

class FullModel:
    def __init__(self, arrange, emission):
        self.arrange = arrange
        self.emission = emission

    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y

    def fit_em(self, Y, iter, tol):
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
        self.emission.initialize(Y)
        for i in range(iter):
            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood
            this_ll = np.sum(Uhat * emloglik) + np.sum(ll_A)
            if (i > 1) and (this_ll - ll[-1] < tol):  # convergence
                break
            else:
                ll.append(this_ll)

            # Updates the parameters
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(Uhat)
            theta[i, :] = np.concatenate([self.emission.get_params(), self.arrange.get_params()])

        return np.asarray(ll), theta[0:len(ll),:]

def _fit_full(Y):
    pass


def _plot_loglike(loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)


def _simulate_full():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionT = MixGaussianGamma(K=5, N=40, P=100)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixGaussianGamma(K=5, N=40, P=100, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=1000, tol=0.001)
    _plot_loglike(ll, color='b')
    print(theta)


if __name__ == '__main__':
    _simulate_full()
