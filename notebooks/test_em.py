# Test convergence of EM for a multi-model
# general import packages
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import time
import copy
import seaborn as sb


import evaluation as ev
from full_model import FullMultiModel
from arrangements import ArrangeIndependent, expand_mn
from emissions import MixGaussianExp, MixGaussian, MixGaussianGamma, MixVMF, wMixVMF
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from test_emissions import _plt_single_param_diff,_plot_loglike,_plot_diff,matching_params


def simulate_full_VMF(X, K=5, P=100, num_sub=10, max_iter=20,
                       uniform_kappa=True, missingdata=None, n_inits=10):
    """Simulation function to check a single VMF data set using fit_em
    Args:
        K: the number of clusters
        P: the number of data points
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        uniform_kappa: If true, the kappa is a scaler across different K's;
                       Otherwise, the kappa is different across K's
    Returns:
        Several evaluation plots.
    """
    N = X.shape[0]

    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
    emissionT = MixVMF(K=K, N=X.shape[0], P=P, X=X, uniform_kappa=uniform_kappa)

    # Step 2: Generate data by sampling from the above model
    T = FullMultiModel(arrangeT, [emissionT])
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)


    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=[Y])  # Run only Estep!
    theta_true = T.get_params()


    # Step 4: Generate new models for fitting
    # Multiple random initializations and then retain best
    LL = np.full((n_inits,max_iter),np.nan)
    best_ll = -np.inf
    best_M = None

    for i in range(n_inits):
        arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
        emissionM = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)
        M = FullMultiModel(arrangeM, [emissionM])

        # Step 5: Estimate the parameter thetas to fit the new model using EM
        M, ll, theta, _ = M.fit_em(Y=[Y], iter=max_iter, tol=0.0001, fit_arrangement=True)
        LL[i,:ll.shape[0]]=ll
        if ll[-1]>best_ll:
            best_ll = ll[-1]
            best_M = M
            best_theta = theta

    # Plot fitting results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], LL[:,1:].T, loglike_true, color='b')

    ind = best_M.get_param_indices('emissions.0.V')
    true_V = theta_true[ind].reshape(emissionM.M, K)
    predicted_V = best_theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = best_M.get_param_indices('emissions.0.kappa')
    if uniform_kappa or emissionT.uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[2],
                    theta_true[T.get_param_indices('emissions.0.kappa')],
                    best_theta[:, ind], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[ind].reshape(1, K),
                            best_theta[:, ind], once=False)
        _plot_diff(axs[2], theta_true[ind].reshape(1, K),
                            best_theta[:, ind], index=idx, name='kappa')

    fig.suptitle('VMF fitting results, run = %d' % int(emissionT.X.shape[0]/emissionT.X.shape[1]))
    plt.tight_layout()
    plt.show()

def simulate_full_VMF2(X, K=5, P=100, num_sub=10, max_iter=20,
                       uniform_kappa=True, missingdata=None,
                       part_vec = None, n_inits=10):
    """Simulation function used for testing full model with a VMF emission
    using em_ninits
    Args:
        K: the number of clusters
        P: the number of data points
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        uniform_kappa: If true, the kappa is a scaler across different K's;
                       Otherwise, the kappa is different across K's
        missingdata: Make some data miss at random?
        part_vec: Partition vector
        n_inits: how many random initis?
    Returns:
        Several evaluation plots.
    """
    N = X.shape[0]

    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
    arrangeT.random_params()
    emissionT = MixVMF(K=K, N=X.shape[0], P=P, X=X, uniform_kappa=uniform_kappa,part_vec=part_vec)

    # Step 2: Generate data by sampling from the above model
    T = FullMultiModel(arrangeT, [emissionT])
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)


    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=[Y])  # Run only Estep!
    theta_true = T.get_params()

    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
    emissionM = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa,part_vec=part_vec)
    M = FullMultiModel(arrangeM, [emissionM])

    M, ll, theta, U_hat, first_ll = M.fit_em_ninits(Y=[Y], iter=max_iter, first_iter=6,
                                  tol=0.001, fit_arrangement=True)

    # Plot fitting EM -results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], first_ll[:,1:].T, loglike_true, color='b')
    axs[0].plot(ll[1:], color='r')

    ind = M.get_param_indices('emissions.0.V')
    true_V = theta_true[ind].reshape(emissionM.M, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emissions.0.kappa')
    if uniform_kappa or emissionT.uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[2],
                    theta_true[T.get_param_indices('emissions.0.kappa')],
                    theta[:, ind], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[ind].reshape(1, K),
                            theta[:, ind], once=False)
        _plot_diff(axs[2], theta_true[ind].reshape(1, K),
                            theta[:, ind], index=idx, name='kappa')

    fig.suptitle('VMF fitting results, run = %d' % int(emissionT.X.shape[0]/emissionT.X.shape[1]))
    plt.tight_layout()
    plt.show()

def simulate_multi(X, K=5, P=100, num_sub=[10], max_iter=20,
                       uniform_kappa=True, missingdata=None,
                       part_vec = None, n_inits=10,align='arrange'):
    """Simulation function used for testing full model with a VMF emission
    using em_ninits
    Args:
        K: the number of clusters
        P: the number of data points
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        uniform_kappa: If true, the kappa is a scaler across different K's;
                       Otherwise, the kappa is different across K's
        missingdata: Make some data miss at random?
        part_vec: Partition vector
        n_inits: how many random initis?
        align: Alignment passed to ninits
    Returns:
        Several evaluation plots.
    """
    n_sets = len(X)
    if part_vec is None:
        part_vec = [None]*n_sets

    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
    arrangeT.random_params()
    emissionT = []
    for i in range(n_sets):
        emissionT.append(MixVMF(K=K, P=P, N=X[i].shape[0], X=X[i],
                                uniform_kappa=uniform_kappa,
                                part_vec=part_vec[i]))

    # Step 2: Generate data by sampling from the above model
    T = FullMultiModel(arrangeT, emissionT)
    U,Y = T.sample(num_subj=num_sub)

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y)  # Run only Estep!
    theta_true = T.get_params()

    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=True, remove_redundancy=False)
    emissionM = []
    for i in range(n_sets):
        emissionM.append(MixVMF(K=K, P=P, N=X[i].shape[0], X=X[i],
                        uniform_kappa=uniform_kappa,
                        part_vec=part_vec[i]))
    M = FullMultiModel(arrangeM, emissionM)

    M, ll, theta, U_hat, first_ll = M.fit_em_ninits(Y=Y,
                iter=max_iter, first_iter=6,
                tol=0.001, fit_arrangement=True,
                align=align)

    # Plot fitting EM -results
    fig, axs = plt.subplots(n_sets, 3, figsize=(12, 8))
    _plot_loglike(axs[0,0], first_ll[:,1:].T, loglike_true, color='b')
    axs[0,0].plot(ll[1:], color='r')
    axs[0,0].set_xlim([-0.5,len(ll)-1.5])
    for i in range(n_sets):
        ind = M.get_param_indices(f'emissions.{i}.V')
        true_V = theta_true[ind].reshape(emissionM[i].M, K)
        predicted_V = theta[1:, ind]
        idx = matching_params(true_V, predicted_V, once=False)
        _plot_diff(axs[i,1], true_V, predicted_V, index=idx, name='V')

        ind = M.get_param_indices(f'emissions.{i}.kappa')
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[i,2],
                    theta_true[T.get_param_indices(f'emissions.{i}.kappa')],
                    theta[1:, ind], name='kappa')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Test full model fitting:
    X = pt.eye(9).repeat(4, 1)  # simulate task design matrix X
    part_vec = np.kron(np.arange(4),np.ones((9,)))
    simulate_multi([X,X,X],num_sub=[6,8,5],
                           part_vec = [part_vec,part_vec,part_vec],
                           align='arrange')