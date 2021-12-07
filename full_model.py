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
            this_ll = np.sum(Uhat * emloglik) + np.sum(ll_A)
            if (i > 1) and (np.abs(this_ll - ll[-1]) < tol):  # convergence
                break
            else:
                ll.append(this_ll)

            # Updates the parameters
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(Uhat)
            theta[i, :] = np.concatenate([self.emission.get_params(), self.arrange.get_params()])

        return np.asarray(ll), theta[0:len(ll), :]


def _fit_full(Y):
    pass


def _plot_loglike(loglike, true_loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)
    plt.axhline(y=true_loglike, color='r', linestyle=':')


def _plot_v_diff(theta_true, theta, K):
    """ Plot the model parameters differences.

    Args:
        theta_true: the params from the true model
        theta: the estimated params
        color: the line color for the differences

    Returns: a matplotlib object to be plot

    """
    iter = theta.shape[0]
    diff = np.empty((iter, K))
    Y = np.split(theta_true, K)
    for i in range(iter):
        x = np.split(theta[i], K)
        for j in range(len(x)):
            dist = np.linalg.norm(x[j] - Y[j])
            diff[i, j] = dist
    plt.figure()
    plt.plot(diff)
    plt.title('the differences: true Vs, estimated Vs')


def _plt_single_param_diff(theta_true, theta, name=None):
    plt.figure()
    if name is not None:
        plt.title('The difference: true %s vs estimated %s' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


def _simulate_full_GMM():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionT = MixGaussian(K=5, N=40, P=100)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixGaussian(K=5, N=40, P=100, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=100, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    _plot_v_diff(theta_true[0:200], theta[:, 0:200], K=5)
    _plt_single_param_diff(theta_true[202], theta[:, 202])
    print('Done.')


def _simulate_full_GME():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionT = MixGaussianExp(K=5, N=40, P=100)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y, signal = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y, signal)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixGaussianExp(K=5, N=40, P=100, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=50, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    _plot_v_diff(theta_true[0:200], theta[:, 0:200], K=5)
    _plt_single_param_diff(theta_true[202], theta[:, 202])
    print('Done.')


def _simulate_full_VMF():
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionT = MixVMF(K=5, N=40, P=100)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=5, P=100, spatial_specific=False)
    emissionM = MixVMF(K=5, N=40, P=100, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=50, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    _plot_v_diff(theta_true[0:200], theta[:, 0:200], K=5)
    _plt_single_param_diff(theta_true[202], theta[:, 202])
    print('Done.')


if __name__ == '__main__':
    _simulate_full_VMF()
