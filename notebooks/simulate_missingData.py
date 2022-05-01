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
import evaluation as ev


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


def _compute_diff(true_param, predicted_params, index=None, compute_single=False):
    """ Compute the model parameters differences.

    Args:
        true_param: the params from the true model
        predicted_params: the estimated params
        index: the matching index of parameters
        compute_single: If True, compute all parameters difference.
                        Otherwise, cluster specific
    Returns: The parameters difference between true model and predicted model
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

    if compute_single:
        diff = diff.sum(1)

    return diff


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


def _simulate_missingData(K=5, width=30, height=30, N=40, num_sub=10, max_iter=50, sigma2=1.0,
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
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    emissionT = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
    emissionT.sigma2 = pt.tensor(sigma2)

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist, theta_mu=150)
    arrangeT.theta_w = pt.tensor(20)

    # Step 4: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = T.arrange.sample(num_subj=num_sub, burnin=30)
    Y_train = T.emission.sample(U)
    # Y_test = T.emission.sample(U)

    # Making incomplete data if needed
    D = []
    if missingdata is not None:
        for m in missingdata:
            radius = np.sqrt(m * grid.P/np.pi)
            centroid = np.random.choice(grid.P, (num_sub,))
            mask = pt.ones(num_sub, grid.P)
            mask[pt.where(grid.Dist[centroid] < radius)] = pt.nan
            Y_train = mask.unsqueeze(1) * Y_train

            # Step 6: Generate new models for fitting
            arrangeM = ar.ArrangeIndependent(K=K, P=grid.P, spatial_specific=True,
                                             remove_redundancy=False)
            emissionM = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
            M = FullModel(arrangeM, emissionM)
            M, ll, theta, Uhat_fit = M.fit_em(Y=Y_train, iter=max_iter, tol=0.00001,
                                              fit_arrangement=True)
            D.append({'U_nan': mask*U, 'Uhat_fit': Uhat_fit, 'M': M, 'theta': theta})
    else:
        mask = pt.ones(num_sub, grid.P)
        # Step 6: Generate new models for fitting
        arrangeM = ar.ArrangeIndependent(K=K, P=grid.P, spatial_specific=True,
                                         remove_redundancy=False)
        emissionM = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
        M = FullModel(arrangeM, emissionM)
        M, ll, theta, Uhat_fit = M.fit_em(Y=Y_train, iter=max_iter, tol=0.00001,
                                          fit_arrangement=True)
        D.append({'U_nan': mask*U, 'Uhat_fit': Uhat_fit, 'M': M, 'theta': theta})

    return grid, T, U, D


def do_simulation_missingData(K=5, width=30, height=30, num_sub=10,
                              missingRate=[0.01, 0.05, 0.1, 0.2], savePic=True):
    """Run the missing data simulation at given missing rate
    Args:
        K: the clusters number
        num_sub: the subject number
        missingRate: the missing data percentage
        savePic: if True, save the simulation figures
    Returns:
        theta_all: All parameters at each EM iteration
        Uerr_all: The absolute error between U and U_hat for each missing rate
        U: The ground truth Us
        U_nan_all: the ground truth Us with missing data
        U_hat_all: the predicted U_hat for each missing rate
    """
    print('Start simulation')
    grid, T, U, D = _simulate_missingData(K=K, N=20, width=width, height=height, num_sub=num_sub,
                                          max_iter=100, sigma2=0.2, missingdata=missingRate)

    Uerr_all, theta_all, U_nan_all, U_hat_all = [], [], [], []
    # Plot the individual parcellations sampled from prior
    for m in range(len(missingRate)):
        U_nan = D[m]['U_nan']
        theta = D[m]['theta']
        M = D[m]['M']
        Uhat_fit, U_err = ev.matching_U(U, D[m]['Uhat_fit'])
        plots = [U, U_nan, Uhat_fit]
        plt.figure(figsize=(20, 2))
        for i in range(len(plots)):
            if savePic:
                plt.figure(figsize=(10, 4))
                grid.plot_maps(plots[i], cmap='tab20', vmax=K, grid=[1, int(num_sub)])
                plt.savefig('missing%d_%d.png' % (missingRate[m] * 100, i), format='png')
                plt.clf()

        U_nan_all.append(U_nan)
        U_hat_all.append(Uhat_fit)
        Uerr_all.append(U_err)
        idx = matching_params(T.emission.V, theta[:, M.get_param_indices('emission.V')], once=False)
        diff_V = _compute_diff(T.emission.V, theta[:, M.get_param_indices('emission.V')],
                               index=idx, compute_single=True)
        diff_sigma = pt.abs(T.emission.sigma2 -
                            theta[:, M.get_param_indices('emission.sigma2')].reshape(-1))
        theta_all.append(diff_V + diff_sigma)

    return theta_all, Uerr_all, U, U_nan_all, U_hat_all


def _plot_maps(U, cmap='tab20', grid=None, offset=1, dim=(30, 30), vmax=19, row_labels=None):
    # Step 7: Plot fitting results
    N, P = U.shape
    if grid is None:
        grid = np.zeros((2,), np.int32)
        grid[0] = np.ceil(np.sqrt(N))
        grid[1] = np.ceil(N / grid[0])

    for n in range(N):
        ax = plt.subplot(grid[0], grid[1], n+offset)
        ax.imshow(U[n].reshape(dim), cmap='tab20', interpolation='nearest', vmax=vmax)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if (row_labels is not None) and (n % num_sub == 0):
            ax.axes.yaxis.set_visible(True)
            ax.set_yticks([])
            ax.set_ylabel(row_labels[int(n / num_sub)])


if __name__ == '__main__':
    # Set up experiment parameters
    K = 5
    num_sub = 10
    rate = [0.01, 0.05, 0.1, 0.2]
    labels = ['r = ' + str(x) for x in rate]
    w, h = 30, 30
    theta_all, Uerr_all, U, U_nan_all, U_hat_all = do_simulation_missingData(K=K, num_sub=num_sub,
                                                                             width=w, height=h,
                                                                             missingRate=rate,
                                                                             savePic=False)

    # Plot the true Us, true Us with missing data, and the estimated Us
    _plot_maps(U, cmap='tab20', grid=[1, num_sub], dim=(w, h), row_labels=['True_U'])
    plt.show()

    U_nan = pt.stack(U_nan_all).flatten(end_dim=1)
    _plot_maps(U_nan, cmap='tab20', grid=[len(rate), num_sub], dim=(w, h),
               row_labels=labels)
    plt.show()

    U_hat = pt.stack(U_hat_all).flatten(end_dim=1)
    _plot_maps(U_hat, cmap='tab20', grid=[len(rate), num_sub], dim=(w, h),
               row_labels=labels)
    plt.show()

    # Plot the difference of model recovery params, and absolute errors of U and U_hat
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(pt.stack(theta_all).T, label=labels)
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel('number of iterations')
    axs[0].set_ylabel(r'Difference between $\theta$ and $\hat{\theta}$')

    axs[1].bar(labels, Uerr_all)
    axs[1].set_ylabel(r'Absolute error between $\mathbf{U}$ and $\hat{\mathbf{U}}$')

    fig.suptitle('Simulation results on different missing data percentage.')
    plt.tight_layout()
    plt.show()
    print('Done simulation missing data.')

