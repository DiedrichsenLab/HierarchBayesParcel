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
from full_model import *


def _plot_loglike(loglike, true_loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)
    plt.axhline(y=true_loglike, color='r', linestyle=':')


def _plot_diff(theta_true, theta, K, name='V'):
    """ Plot the model parameters differences.

    Args:
        theta_true: the params from the true model
        theta: the estimated params
        color: the line color for the differences
    Returns: a matplotlib object to be plot
    """
    theta = theta[~np.all(theta == 0, axis=1)]
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
    plt.title('the differences: true %ss, estimated %ss' % (name, name))


def _plt_single_param_diff(theta_true, theta, name=None):
    plt.figure()
    if name is not None:
        plt.title('The difference: true %s vs estimated %s' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


def generate_data(emission, k=2, dim=3, p=1000,
                  num_sub=10, beta=1, alpha=1, signal_type=0):
    model_name = ["GMM", "GMM_exp", "GMM_gamma", "VMF"]
    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False)
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
        data = emissionT.sample(U, signal)
    elif emission == 3:
        data = emissionT.sample(U)
        signal = np.repeat(signal[:, np.newaxis, :], dim, axis=1)
        data = data * signal
    else:
        data = emissionT.sample(U)

    return data, U


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


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
    emloglik = emissionT.Estep(Y)
    Uhat, ll_a = arrangeT.Estep(emloglik)
    loglike_true = pt.sum(Uhat * emloglik) + pt.sum(ll_a)
    # # print(theta_true)
    T = FullModel(arrangeT, emissionT)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussian(K=K, N=N, P=P)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, 
                tol=0.00001, fit_arrangement =False)
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    _plot_diff(theta_true[0:N*K], theta[:, 0:N*K], K, name='V')
    _plt_single_param_diff(theta_true[-1-K], np.trim_zeros(theta[:, -1-K], 'b'), name='sigma2')
    print('Done.')


def _simulate_full_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emissionT.initialize(Y)
    emll_true = emissionT.Estep(signal=signal)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = pt.sum(Uhat * emll_true) + pt.sum(ll_a)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussianExp(K=K, N=N, P=P)
    # emissionM.set_params(pt.cat((emissionM.V.flatten(), emissionT.sigma2.reshape(1), emissionM.alpha.reshape(1), emissionM.beta.reshape(1))))

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, U_hat = M.fit_em(Y=Y, iter=max_iter, tol=0.0001,fit_arrangement=False)
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    _plot_diff(theta_true[0:N*K], theta[:, 0:N*K], K, name='V')
    _plt_single_param_diff(theta_true[-3-K], np.trim_zeros(theta[:, -3-K], 'b'), name='sigma2')
    _plt_single_param_diff(theta_true[-1-K], np.trim_zeros(theta[:, -1-K], 'b'), name='beta')
    # SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=True, soft_assign=False)
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
    emissionT.initialize(Y)
    emll_true = emissionT.Estep()
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = pt.sum(Uhat * emll_true) + pt.sum(ll_a)
    print(theta_true)
    T = FullModel(arrangeT, emissionT)
    ## T, ll, theta, _ = T.fit_em(Y=Y, iter=1, tol=0.00001)
    ## loglike_true = ll
    

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixVMF(K=K, N=N, P=P, uniform=False)
    # emissionM.set_params([emissionM.V, emissionM.kappa])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001,fit_arrangement=False)
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    _plot_diff(theta_true[0:N * K], theta[:, 0:N * K], K, name='V')
    _plot_diff(theta_true[N*K: N*K+K], theta[:, N*K:N*K+K], K, name='Kappa')
    print('Done.')


if __name__ == '__main__':
    # _simulate_full_VMF(K=5, P=1000, N=40, num_sub=10, max_iter=100)
    _simulate_full_VMF(K=5, P=1000, N=20, num_sub=10, max_iter=100)
    # _simulate_full_GME(K=5, P=2000, N=20, num_sub=10, max_iter=100)