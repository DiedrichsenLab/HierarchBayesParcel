#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script of simulating the generative model training when across dataset,
and test the model recovery ability.

Created on 5/3/2022 at 2:09 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# for testing and evaluating models
from full_model import FullModel, FullMultiModel
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


def _simulate_dataFusion(K=5, width=30, height=30, N=40, max_iter=50, sigma2=1.0,
                          uniform_kappa=True, missingdata=None, nsub_list=None):
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
    # emissionT = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
    emissionT1 = em.MixVMF(K=K, N=N, P=grid.P, uniform_kappa=uniform_kappa)
    emissionT1.kappa = pt.tensor(sigma2)
    # emissionT2 = em.MixGaussian(K=K, N=N, P=grid.P, std_V=False)
    # emissionT2.sigma2 = pt.tensor(sigma2)
    emissionT2 = em.MixVMF(K=K, N=N, P=grid.P, uniform_kappa=uniform_kappa)
    emissionT2.kappa = pt.tensor(sigma2)

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist, theta_mu=50)
    arrangeT.theta_w = pt.tensor(20)

    # Step 4: Generate data by sampling from the above model
    D = []
    T = FullMultiModel(arrangeT, [emissionT1, emissionT2])
    for ns in nsub_list:
        U, Y_train = T.sample(num_subj=ns)
        num_sub = sum(ns)
        if missingdata is not None:
            Y_train = pt.cat(Y_train, dim=0)
            radius = np.sqrt(missingdata * grid.P / np.pi)
            centroid = np.random.choice(grid.P, (num_sub,))
            mask = pt.ones(num_sub, grid.P)
            mask[pt.where(grid.Dist[centroid] < radius)] = pt.nan
            Y_train = mask.unsqueeze(1) * Y_train
            U_nan = mask * U
            Y_train = pt.split(Y_train, ns, dim=0)
        else:
            U_nan = U

        # Step 6: Generate new models for fitting
        arrangeM = ar.ArrangeIndependent(K=K, P=grid.P, spatial_specific=True,
                                         remove_redundancy=False)
        emissionM1 = em.MixVMF(K=K, N=N, P=grid.P, uniform_kappa=uniform_kappa)
        emissionM2 = em.MixVMF(K=K, N=N, P=grid.P, uniform_kappa=uniform_kappa)

        # set params of recovery model to truth
        arrangeM.logpi = T.arrange.logpi
        # emissionM1.V = emissionT1.V
        # emissionM1.kappa = emissionT1.kappa
        # emissionM2.V = emissionT2.V
        # emissionM2.kappa = emissionT2.kappa

        M = FullMultiModel(arrangeM, [emissionM1, emissionM2])
        M, ll, theta, Uhat_fit = M.fit_em(Y=Y_train, iter=max_iter,
                                          tol=0.00001, fit_arrangement=False)
        D.append({'U': U, 'U_nan': U_nan, 'Uhat_fit': Uhat_fit, 'M': M, 'theta': theta})

    return grid, T, D


def do_simulation_dataFusion(K=5, width=30, height=30, nsub_list=[[5, 5]],
                              missingRate=None, savePic=True):
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
    grid, T, D = _simulate_dataFusion(K=K, N=20, width=width, height=height, max_iter=50,
                                      sigma2=20, missingdata=missingRate, nsub_list=nsub_list)

    U_all, Uerr_all, theta_all, U_nan_all, U_hat_all = [], [], [], [], []
    # Plot the individual parcellations sampled from prior
    for m in range(len(nsub_list)):
        U = D[m]['U']
        U_nan = D[m]['U_nan']
        theta = D[m]['theta']
        M = D[m]['M']
        Uhat_fit, U_err = ev.matching_U(U, D[m]['Uhat_fit'])
        plots = [U, U_nan, Uhat_fit]
        plt.figure(figsize=(20, 2))
        for i in range(len(plots)):
            if savePic:
                plt.figure(figsize=(10, 4))
                grid.plot_maps(plots[i], cmap='tab20', vmax=19, grid=[1, int(sum(nsub_list[m]))])
                plt.savefig('missing%s_%d.eps' % (str(m), i), format='eps')
                plt.clf()

        U_all.append(U)
        U_nan_all.append(U_nan)
        U_hat_all.append(Uhat_fit)
        Uerr_all.append(U_err)
        diff_V, diff_sigma = [], []
        for n, emT in enumerate(T.emissions):
            idx = matching_params(emT.V, theta[:, M.get_param_indices('emissions.%d.V' % n)],
                                  once=False)
            this_diff_V = _compute_diff(emT.V, theta[:, M.get_param_indices('emissions.%d.V' % n)],
                                        index=idx, compute_single=True)
            this_diff_sigma = pt.abs(emT.kappa - theta[:, M.get_param_indices('emissions.%d.kappa'
                                                                              % n)].reshape(-1))
            diff_V.append(this_diff_V)
            diff_sigma.append(this_diff_sigma)

        diff_V = pt.stack(diff_V).sum(dim=0)
        diff_sigma = pt.stack(diff_sigma).sum(dim=0)
        theta_all.append(diff_V + diff_sigma)

    return theta_all, Uerr_all, U_all, U_nan_all, U_hat_all


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
    nsub_list = [[3, 7], [2, 8]]
    rate = [0.01, 0.1, 0.2]
    labels = [str(x) for x in nsub_list]
    w, h = 30, 30
    theta_all, Uerr_all, U, U_nan_all, U_hat_all = do_simulation_dataFusion(K=K, width=w, height=h,
                                                                            nsub_list=nsub_list,
                                                                            savePic=False)

    # Plot the true Us and the estimated Us
    U = pt.stack(U).flatten(end_dim=1)
    _plot_maps(U, cmap='tab20', grid=[len(nsub_list), num_sub], dim=(w, h), row_labels=labels)
    plt.show()

    U_hat = pt.stack(U_hat_all).flatten(end_dim=1)
    _plot_maps(U_hat, cmap='tab20', grid=[len(nsub_list), num_sub], dim=(w, h),
               row_labels=labels)
    plt.show()

    # Plot the difference of model recovery params, and absolute errors of U and U_hat
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(pt.stack(theta_all).T, label=labels)
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel('number of iterations')
    axs[0].set_ylabel(r'Difference between $\theta$ and $\hat{\theta}$')

    axs[1].bar(labels, pt.stack(Uerr_all).mean(dim=1), yerr=pt.stack(Uerr_all).std(dim=1),
               capsize=10)
    axs[1].set_ylim([0, 1])
    axs[1].set_ylabel(r'Absolute error between $\mathbf{U}$ and $\hat{\mathbf{U}}$')

    fig.suptitle('Simulation results on different missing data percentage.')
    plt.tight_layout()
    plt.savefig('curves.eps', format='eps')
    plt.show()
    print('Done simulation missing data.')

