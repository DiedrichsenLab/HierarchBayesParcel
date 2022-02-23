#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/15/2022
The script the test different emission models standalone

Author: DZHI
"""
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from full_model import FullModel
from arrangements import ArrangeIndependent
from emissions import MixGaussianExp, MixGaussian, MixGaussianGamma, MixVMF
import time
import copy

def _plot_loglike(loglike, true_loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)
    plt.axhline(y=true_loglike, color='r', linestyle=':')
    plt.title('True log-likelihood (red) vs. estimated log-likelihood (blue)')


def _plot_diff(true_param, predicted_params, index=None, name='V'):
    """ Plot the model parameters differences.

    Args:
        true_param: the params from the true model
        predicted_params: the estimated params
        index: the matching index of parameters
        color: the line color for the differences
    Returns: a matplotlib object to be plot
    """
    # Convert input to tensor if ndarray
    if type(true_param) is np.ndarray:
        true_param = pt.tensor(true_param, dtype=pt.get_default_dtype())
    if type(predicted_params) is np.ndarray:
        predicted_params = pt.tensor(predicted_params, dtype=pt.get_default_dtype())

    N, K = true_param.shape
    diff = pt.empty(predicted_params.shape[0], K)

    for i in range(predicted_params.shape[0]):
        this_pred = predicted_params[i].reshape(N, K)
        for k in range(K):
            if index is not None:
                dist = pt.linalg.norm(this_pred[:, k] - true_param[:, index[i, k]])
            else:
                dist = pt.linalg.norm(this_pred[:, k] - true_param[:, k])
            diff[i, k] = dist
    plt.figure()
    plt.plot(diff)
    plt.title('the differences between true %ss and estimated %ss for each k' % (name, name))


def _plt_single_param_diff(theta_true, theta, name=None):
    plt.figure()
    if name is not None:
        plt.title('True %s (red) vs estimated %s (blue)' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


def generate_data(emission, k=2, dim=3, p=1000,
                  num_sub=10, beta=1, alpha=1, signal_type=0):
    model_name = ["GMM", "GMM_exp", "GMM_gamma", "VMF"]
    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False, remove_redundancy=False)
    U = arrangeT.sample(num_subj=num_sub)
    if signal_type == 0:
        signal = np.random.exponential(beta, (num_sub, p))
    elif signal_type == 1:
        signal = np.random.gamma(alpha, beta, (num_sub, p))
    else:
        raise ValueError("The value of signal strength must satisfy a distribution, 0 - exponential; 1 - gamma.")

    if emission == 0:  # GMM
        emissionT = MixGaussian(K=k, N=dim, P=p)
    elif emission == 1:  # GMM with exponential signal strength
        emissionT = MixGaussianExp(K=k, N=dim, P=p)
    elif emission == 2:  # GMM with gamma signal strength
        emissionT = MixGaussianGamma(K=k, N=dim, P=p)
    elif emission == 3:
        emissionT = MixVMF(K=k, N=dim, P=p)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), or 3(VMF).")

    if (emission == 1) or (emission == 2):
        data, signal = emissionT.sample(U, return_signal=True)
    elif emission == 3:
        data = emissionT.sample(U)
        signal = np.repeat(signal[:, np.newaxis, :], dim, axis=1)
        data = data * signal
    else:
        data = emissionT.sample(U)

    return data, U


def evaluate_completion_emission(emissionM, data, k_fold=5, crit='u_abserr'):
    """ Evaluates an emission model on new dataset using cross-validation and
        given criterion
    Args:
        emissionM:
        data:
        k_fold:
        crit:
    Returns:
        evaluation results
    """
    if type(data) is np.ndarray:
        data = pt.tensor(data, dtype=pt.get_default_dtype())



def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def matching_params(true_param, predicted_params, once=False):
    """ Matching the estimated parameters with the true one, return indices

    Args:
        true_param: the true parameter, shape (N, K)
        predicted_params: the estimated parameter. Shape (iter, N*K)
        once: True - perform matching in every iteration. Otherwise only take
              matching once using the estimated param of first iteration

    Returns: The matching index

    """
    # Convert input to tensor if ndarray
    if type(true_param) is np.ndarray:
        true_param = pt.tensor(true_param, dtype=pt.get_default_dtype())
    if type(predicted_params) is np.ndarray:
        predicted_params = pt.tensor(predicted_params, dtype=pt.get_default_dtype())

    N, K = true_param.shape
    if once:
        index = pt.empty(K, )
        for k in range(K):
            tmp = pt.linalg.norm(predicted_params[0].reshape(N, K)[:, k] - true_param.transpose(0, 1), dim=1)
            index[k] = pt.argmin(tmp)
            true_param[:, pt.argmin(tmp)] = pt.tensor(float('inf'))
        index.expand(predicted_params.shape[0], K)
    else:
        index = pt.empty(predicted_params.shape[0], K)
        for i in range(index.shape[0]):
            this_pred = predicted_params[i].reshape(N, K)
            this_true = pt.clone(true_param).transpose(0, 1)
            for k in range(K):
                tmp = pt.linalg.norm(this_pred[:, k] - this_true, dim=1)
                index[i, k] = pt.argmin(tmp)
                this_true[pt.argmin(tmp), :] = pt.tensor(float('inf'))

    return index.int()


def _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=50,sigma2=1.0):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussian(K=K, N=N, P=P)
    emissionT.sigma2 = pt.tensor(sigma2)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the true log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y)
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussian(K=K, N=N, P=P)
    # new_params = emissionM.get_params()
    # new_params[emissionM.get_param_indices('sigma2')] = emissionT.get_params()[emissionT.get_param_indices('sigma2')]
    # emissionM.set_params(new_params)
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Plot fitting results
    _plot_loglike(ll, loglike_true, color='b')
    true_V = theta_true[M.get_param_indices('emission.V')].reshape(N, K)
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)

    _plot_diff(true_V, predicted_V, index=idx, name='V')
    _plt_single_param_diff(theta_true[M.get_param_indices('emission.sigma2')],
                           theta[:, M.get_param_indices('emission.sigma2')], name='sigma2')

    plt.show()
    print('Done simulation GMM.')


def _simulate_full_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100,
        sigma2=1.0,beta =1.0):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P)
    emissionT.sigma2 = pt.tensor(sigma2)
    emissionT.beta = pt.tensor(beta)
    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U, return_signal=True)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussianExp(K=K, N=N, P=P)
    emissionM.std_V = True # Set to False if you don't want to standardize V....
    # new_params = emissionM.get_params()
    # new_params[emissionM.get_param_indices('sigma2')] = emissionT.get_params()[emissionT.get_param_indices('sigma2')]
    # new_params[emissionM.get_param_indices('beta')] = emissionT.get_params()[emissionT.get_param_indices('beta')]
    # emissionM.set_params(new_params)
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

    # Plotfitting results
    _plot_loglike(ll, loglike_true, color='b')
    ind = M.get_param_indices('emission.V')
    true_V = theta_true[ind].reshape(N, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(np.log(theta_true[ind]),np.log(theta[:, ind]), name='log sigma2')

    ind = M.get_param_indices('emission.beta')
    _plt_single_param_diff(theta_true[ind],theta[:, ind], name='beta')
    # SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=True, soft_assign=False)

    plt.show()
    print('Done simulation GME.')


def _simulate_full_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50, uniform_kappa=True):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixVMF(K=K, N=N, P=P, uniform_kappa=uniform_kappa)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)  # Run only Estep!
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixVMF(K=K, N=N, P=P, uniform_kappa=uniform_kappa)
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Plotfitting results
    _plot_loglike(ll, loglike_true, color='b')
    true_V = theta_true[M.get_param_indices('emission.V')].reshape(N, K)
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)

    _plot_diff(true_V, predicted_V, index=idx, name='V')

    if uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(theta_true[M.get_param_indices('emission.kappa')],
                               theta[:, M.get_param_indices('emission.kappa')], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[M.get_param_indices('emission.kappa')].reshape(1, K),
                              theta[:, M.get_param_indices('emission.kappa')], once=False)
        _plot_diff(theta_true[M.get_param_indices('emission.kappa')].reshape(1, K),
                   theta[:, M.get_param_indices('emission.kappa')], index=idx, name='kappa')

    plt.show()
    print('Done simulation VMF.')

def _test_GME_Estep(K=5, P=200, N=8, num_sub=10, max_iter=100,
        sigma2=1.0,beta =1.0):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P)
    emissionT.sigma2 = pt.tensor(sigma2)
    emissionT.beta = pt.tensor(beta)
    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U, return_signal=True)

    em1 = copy.deepcopy(emissionT)
    em2 = copy.deepcopy(emissionT)

    # Check old and new Estep
    t = time.time()
    LL1 = em1.Estep(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    t = time.time()
    LL2 = em2.Estep_old(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    pass


if __name__ == '__main__':
    # _simulate_full_VMF(K=5, P=1000, N=20, num_sub=10, max_iter=100, uniform_kappa=False)
    # _simulate_full_GMM(K=5, P=1000, N=20, num_sub=10, max_iter=100)
    _simulate_full_GME(K=7, P=200, N=20, num_sub=10, max_iter=50,sigma2=1.0,beta=1.0)
    pass
    # _test_GME_Estep(P=500)