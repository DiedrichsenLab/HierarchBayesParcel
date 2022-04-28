#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script of simulating the generative model training when the data is incomplete,
and test the model recovery ability.

Created on 4/27/2022 at 2:51 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# for testing and evaluating models
from full_model import FullModel
import arrangements as ar
import emissions as em
import spatial as sp


def _plot_loglike(ax, loglike, true_loglike, color='b'):
    """Plot the log-likelihood curve and the true log-likelihood
    Args:
        loglike: The log-likelihood curve of the EM iterations
        true_loglike: The true log-likelihood from the true model
        color: the color of the log-likelihood curve

    Returns:
        The plot
    """
    ax.plot(loglike, color=color)
    ax.axhline(y=true_loglike, color='r', linestyle=':')
    ax.set_title('log-likelihood')
    return ax


def _plot_diff(ax, true_param, predicted_params, index=None, name='V', plot_single=False):
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

    if plot_single:
        diff = diff.sum(1)

    ax.plot(diff)
    ax.set_title('differences of true/estimated %ss' % name)
    return ax, diff


def _plt_single_param_diff(ax, theta_true, theta, name=None):
    """Plot the single estimated parameter array and the true parameter
    Args:
        theta_true: the true parameter
        theta: the estimated parameter
        name: the name of the plotted parameter

    Returns:
        The plot
    """
    if name is not None:
        ax.set_title('True/estimated %s' % name)

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    ax.plot(theta_true, linestyle='--', color='r')
    ax.plot(theta, color='b')
    return ax


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
            tmp = pt.linalg.norm(predicted_params[0].reshape(N, K)[:, k]
                                 - true_param.transpose(0, 1), dim=1)
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


def _simulate_missingData(K=5, width=30, height=30, N=40, num_sub=10, max_iter=50,sigma2=1.0,
                       missingdata=None):
    """Simulation function used for testing full model with a GMM emission
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of dimensions
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        sigma2: the sigma2 for GMM emission model
    Returns:
        Several evaluation plots.
    """
    # Ytrain, Ytest, Utrue, Mtrue = make_mrf_data(width=width, K=K, N=N, theta_mu=100, theta_w=20,
    #                                             sigma2=sigma2, do_plot=True)

    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.PottsModel(grid.W, K=K)
    emissionT = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
    emissionT.sigma2 = pt.tensor(sigma2)

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist, theta_mu=150)
    arrangeT.theta_w = pt.tensor(20)

    # Step 3: Plot the prior of the true mode
    plt.figure()
    grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, K])
    cluster = np.argmax(arrangeT.logpi, axis=0)
    grid.plot_maps(cluster, cmap='tab20')

    # Step 4: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = T.arrange.sample(num_subj=num_sub, burnin=30)
    Y_train = T.emission.sample(U)
    Y_test = T.emission.sample(U)

    plt.figure(figsize=(10, 4))
    grid.plot_maps(U[0:10], cmap='tab20', vmax=K, grid=[2, int(num_sub/2)])

    # Making incomplete data if needed
    if missingdata is not None:
        radius = np.sqrt(missingdata * grid.P/np.pi)
        centroid = np.random.choice(grid.P, (num_sub,))
        mask = pt.ones(num_sub, grid.P)
        mask[pt.where(grid.Dist[centroid] < radius)] = pt.nan
        Y_train = mask.unsqueeze(1) * Y_train
        # mask = pt.randint(0, grid.P-1, (num_sub, int(missingdata*grid.P)))
        # Y_train = pt.transpose(Y_train, 1, 2)
        # Y_train[pt.arange(Y_train.shape[0]).unsqueeze(-1), mask] = pt.nan
        # Y_train = pt.transpose(Y_train, 1, 2)
        grid.plot_maps(mask * U, cmap='tab20', vmax=K, grid=[2, int(num_sub / 2)])

    # Step 6: Generate new models for fitting
    # arrangeM = ar.PottsModel(grid.W, K=K)
    arrangeM = ar.ArrangeIndependent(K=K, P=grid.P, spatial_specific=True, remove_redundancy=False)
    emissionM = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, Uhat_fit = M.fit_em(Y=Y_train, iter=max_iter, tol=0.00001, fit_arrangement=True)

    # Step 7: Plot fitting results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # _plot_loglike(axs[0], ll, loglike_true, color='b')

    true_V = T.emission.V
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)
    _, diff = _plot_diff(axs[0], true_V, predicted_V, index=idx, name='V', plot_single=True)

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(axs[1], np.log(T.emission.sigma2),
                           np.log(theta[:, ind]), name='log sigma2')

    fig.suptitle('GMM fitting results')
    plt.tight_layout()
    # plt.show()
    print('Done simulation GMM.')
    return Uhat_fit, diff


if __name__ == '__main__':
    Uhat, diff = _simulate_missingData(K=5, N=20, num_sub=10, max_iter=100, sigma2=0.2,
                                       missingdata=0.1)
