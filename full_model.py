from numpy.typing import _16Bit
from arrangements import ArrangeIndependent
from emissions import MixGaussian, MixGaussianExp, MixVMF
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
import sys
# sys.path.insert(0, "D:/python_workspace/")

class FullModel:
    def __init__(self, arrange,emission):
        self.arrange = arrange
        self.emission = emission
        self.nparams = self.arrange.nparams + self.emission.nparams 

    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y

    def fit_em(self, Y, iter=30, tol=0.01, seperate_ll=False):
        """ Run the EM-algorithm on a full model
        this demands that both the Emission and Arrangement model
        have a full Estep and Mstep and can calculate the likelihood, including the partition function

        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector

        """
        # Initialize the tracking
        ll = np.zeros((iter,2))
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = np.concatenate([self.arrange.get_params(),self.emission.get_params()])

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood
            ll_E = np.sum(Uhat * emloglik, axis=(1, 2))
            ll[i,0]=np.sum(ll_A)
            ll[i,1]=np.sum(ll_E)
            # Check convergence
            if i==iter-1 or ((i > 1) and (ll[i,:].sum() - ll[i-1,:].sum() < tol)):
                break

            # Updates the parameters
            self.emission.Mstep(Uhat)
            self.arrange.Mstep()

        if seperate_ll:
            return self, ll[:i+1, :], theta[:i+1, :]
        else:
            return self, ll[:i+1, :].sum(axis=1), theta[:i+1, :]

    def fit_sml(self, Y, iter=60, stepsize= 0.8, seperate_ll=False):
        """ Runs a Stochastic Maximum likelihood algorithm on a full model.
        The emission model is still assumed to have E-step and Mstep. 
        The arrangement model is has a postive and negative phase estep, 
        and a gradient M-step. The arrangement likelihood is not necessarily 
        FUTURE EXTENSIONS: 
        * Sampling of subjects from training set 
        * initialization of parameters
        * adaptitive stopping criteria
        * Adaptive stepsizes 
        * Gradient acceleration methods
        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations
            stepsize (double): Fixed step size for MStep 
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E 
            theta (ndarray): History of the parameter vector
        """
        # Initialize the tracking
        ll = np.zeros((iter,2))
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = np.concatenate([self.arrange.get_params(),self.emission.get_params()])

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat,ll_A = self.arrange.epos_sample(emloglik)
            # Compute the expected complete logliklihood
            ll_E = np.sum(Uhat * emloglik,axis=(1,2))
            ll[i,0]=np.sum(ll_A)
            ll[i,1]=np.sum(ll_E)

            if i==iter-1:
                break

            # Run the negative phase 
            self.arrange.eneg_sample()
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(stepsize)

        if seperate_ll: 
            return self,ll[:i+1,:], theta[:i+1,:]
        else:
            return self,ll[:i+1,:].sum(axis=1), theta[:i+1,:]

    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model 
        Returns: 
            theta (ndarrap) 
        """
        return np.concatenate([self.arrange.get_params(),self.emission.get_params()])
    
    def get_param_indices(self,name):
        """Return the indices for the full model theta vector

        Args:
            name (str): Parameter name in the format of 'arrange.logpi' 
                        or 'emission.V'
        Returns: 
            indices (np.ndarray): 1-d numpy array of indices into the theta vector 
        """
        names = name.split(".")
        if len(names)==2:
            ind=vars(self)[names[0]].get_param_indices(names[1])
            if names[0]=='emission':
                ind=ind+self.arrange.nparams
            return ind
        else: 
            raise NameError('Parameter name needs to be model.param')


def _fit_full(Y):
    pass


def _plot_loglike(loglike, true_loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)
    plt.axhline(y=true_loglike, color='r', linestyle=':')


def _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussian(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    # emll_true = emissionT._loglikelihood(Y)
    # Uhat, ll_a = arrangeT.Estep(emll_true)
    # loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    # print(theta_true)
    emissionT.initialize(Y)
    emll_true = emissionT.Estep()
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixGaussian(K=K, N=N, P=P)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta = M.fit_em(Y, iter=max_iter, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    print('Done.')


def _simulate_full_GME():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=5, N=40, P=100)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y, signal = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([arrangeT.get_params(), emissionT.get_params()])
    emissionT.initialize(Y)
    emissionT.s = signal
    emll_true = emissionT.Estep_grid()
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixGaussianExp(K=5, N=40, P=100)
    # emissionM.set_params([emissionT.V, emissionT.sigma2, emissionT.alpha, emissionM.beta])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(Y, iter=50, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    print('Done.')


def _simulate_full_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixVMF(K=K, N=N, P=P, uniform=True)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    # emissionT.initialize(Y)
    # emll_true = emissionT.Estep()
    # Uhat, ll_a = arrangeT.Estep(emll_true)
    # loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    # print(theta_true)
    T = FullModel(arrangeT, emissionT)
    T, ll, theta = T.fit_em(Y=Y, iter=1, tol=0.00001)
    loglike_true = ll

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixVMF(K=K, N=N, P=P, uniform=False)
    # emissionM.set_params([emissionM.V, emissionM.kappa])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta = M.fit_em(Y=Y, iter=max_iter, tol=0.00001)
    _plot_loglike(ll, loglike_true, color='b')
    print('Done.')


if __name__ == '__main__':
    _simulate_full_VMF(K=5, P=1000, N=40, num_sub=10, max_iter=50)
    # _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=100)
    # _simulate_full_GME()
