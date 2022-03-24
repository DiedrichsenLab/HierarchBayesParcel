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

import evaluation
from full_model import FullModel
from arrangements import ArrangeIndependent
from emissions import MixGaussianExp, MixGaussian, MixGaussianGamma, MixVMF
import time
import copy
import seaborn as sb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import evaluation as ev
import pandas as pd


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


def _plot_diff(ax, true_param, predicted_params, index=None, name='V'):
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

    ax.plot(diff)
    ax.set_title('differences of true/estimated %ss' % name)
    return ax


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


def sample_spherical(npoints, ndim=3):
    """ Sampling a set of spherical data
    Args:
        npoints: the number of data points
        ndim: the dimensions of the data points

    Returns:
        The sphereical data
    """
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


def generate_data(emission, k=2, dim=3, p=1000, num_sub=10, dispersion=1.2,
                  beta=1.0, do_plot=False, same_signal=True):
    """Generate (and plots) the generated data from a given emission model
    Args:
        emission model: GMM, GME, GMG, VMF
        k: The number of clusters
        dim: The number of data dimensions
        p: the number of generated data points
        num_sub: the number of subjects
        dispersion: Sigma2 or kappa for distribution
        beta (float):
        same_signal (bool): Is signal strength the same across training and test?
    Returns:
        The generated data
    """
    # Step 1: Create the true model and initialize parameters

    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False, remove_redundancy=False)
    if emission == 'GMM':
        emissionT = MixGaussian(K=k, N=dim, P=p, std_V=False)
        emissionT.sigma2 = pt.tensor(dispersion)
    elif emission == 'GME':
        emissionT = MixGaussianExp(K=k, N=dim, P=p, num_signal_bins=100, std_V=True)
        emissionT.sigma2 = pt.tensor(dispersion)
        emissionT.beta = pt.tensor(beta)
    elif emission == 'GMG':  # GMM with gamma signal strength
        emissionT = MixGaussianGamma(K=k, N=dim, P=p)
        emissionT.beta = pt.tensor(beta)
    elif emission == 'VMF':
        emissionT = MixVMF(K=k, N=dim, P=p)
        emissionT.kappa = pt.tensor(dispersion)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), or 3(VMF).")
    MT = FullModel(arrangeT, emissionT)

    # Step 2: Generate data by sampling from the true model
    U = MT.arrange.sample(num_subj=num_sub)
    if emission == 'GME':
        Y_train, signal = MT.emission.sample(U, return_signal=True)
        if same_signal:
            Y_test = MT.emission.sample(U, signal=signal)
        else:
            Y_test = MT.emission.sample(U)
    elif emission == 'VMF':
        signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, 1, p))
        Y_train = MT.emission.sample(U)
        Y_test = MT.emission.sample(U)
        Y_train = Y_train * signal
        if not same_signal:
            signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, 1, p))
        Y_test = Y_test * signal
    else:
        Y_train = MT.emission.sample(U)
        Y_test = MT.emission.sample(U)
        signal = None

    # Step 3: Plot the generated data from the true model (select the first 3 dims if N>3)
    if do_plot:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sb.scatterplot(x=Y_train[0, 0, :], y=Y_train[0, 1, :], hue=U[0], palette="deep")
        plt.title('Training data (first two dimensions)')

        plt.subplot(1, 2, 2)
        sb.scatterplot(x=Y_test[0, 0, :], y=Y_test[0, 1, :], hue=U[0], palette="deep")
        plt.title('Test data (first two dimensions)')
        plt.show()

    return Y_train, Y_test, signal, U, MT


def _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=50,sigma2=1.0):
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
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussian(K=K, N=N, P=P, std_V=False)
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
    emissionM = MixGaussian(K=K, N=N, P=P, std_V=False)
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, Uhat_fit = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Plot fitting results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    true_V = theta_true[M.get_param_indices('emission.V')].reshape(N, K)
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(axs[2], np.log(theta_true[ind]),
                           np.log(theta[:, ind]), name='log sigma2')

    fig.suptitle('GMM fitting results')
    plt.tight_layout()
    plt.show()
    print('Done simulation GMM.')


def _simulate_full_GMM_from_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50, beta=0.5, sigma2=1.0):
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
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixVMF(K=K, N=N, P=P)
    # emissionT.sigma2 = pt.tensor(sigma2)

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
    signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, P))
    Ys = Y * signal.unsqueeze(1).repeat(1, N, 1)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=["Raw VMF", "Raw VMF with signal strength", "GMM fit"])

    fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                               mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=Ys[0, 0, :], y=Ys[0, 1, :], z=Ys[0, 2, :],
                               mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=2)

    M, ll, theta, Uhat_fit = M.fit_em(Y=Ys, iter=max_iter, tol=0.00001, fit_arrangement=False)
    fig.add_trace(go.Scatter3d(x=Ys[0, 0, :], y=Ys[0, 1, :], z=Ys[0, 2, :],
                               mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat_fit, dim=1)[0])), row=1, col=3)

    fig.update_layout(title_text='Comparison of data and fitting')
    fig.show()

    # # Plot fitting results
    # _plot_loglike(ll, loglike_true, color='b')
    # true_V = theta_true[M.get_param_indices('emission.V')].reshape(N, K)
    # predicted_V = theta[:, M.get_param_indices('emission.V')]
    # idx = matching_params(true_V, predicted_V, once=False)
    #
    # _plot_diff(true_V, predicted_V, index=idx, name='V')
    # _plt_single_param_diff(theta_true[M.get_param_indices('emission.sigma2')],
    #                        theta[:, M.get_param_indices('emission.sigma2')], name='sigma2')
    #
    # plt.show()
    print('Done simulation GMM from VMF.')


def _simulate_full_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100,
        sigma2=1.0, beta=1.0, num_bins=100, std_V=True,
        type_estep='linspace'):
    """Simulation function used for testing full model with a GMM_exp emission
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of dimensions
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        sigma2: the sigma2 for GMM_exp emission model
        beta: the beta for GMM_exp emission model
    Returns:
        Several evaluation plots.
    """
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins, std_V=std_V)
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
    emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins, std_V=std_V,type_estep=type_estep)
    emissionM.std_V = std_V
    # emissionM.V =emissionT.V
    # emissionM.sigma2 =emissionT.sigma2
    # emissionM.beta =emissionT.beta
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

    # Plotfitting results
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    ind = M.get_param_indices('emission.V')
    true_V = theta_true[ind].reshape(N, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(axs[2], np.log(theta_true[ind]), np.log(theta[:, ind]), name='log sigma2')

    ind = M.get_param_indices('emission.beta')
    _plt_single_param_diff(axs[3], theta_true[ind],theta[:, ind], name='beta')
    # SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=True, soft_assign=False)

    fig.suptitle('GME fitting results')
    plt.tight_layout()
    plt.show()
    print('Done simulation GME.')


def _simulate_full_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50, uniform_kappa=True):
    """Simulation function used for testing full model with a VMF emission
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of dimensions
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        uniform_kappa: If true, the kappa is a scaler across different K's;
                       Otherwise, the kappa is different across K's
    Returns:
        Several evaluation plots.
    """
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
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    ind = M.get_param_indices('emission.V')
    true_V = theta_true[ind].reshape(N, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.kappa')
    if uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[2], theta_true[ind], theta[:, ind], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[ind].reshape(1, K), theta[:, ind], once=False)
        _plot_diff(axs[2], theta_true[ind].reshape(1, K), theta[:, ind], index=idx, name='kappa')

    fig.suptitle('VMF fitting results')
    plt.tight_layout()
    plt.show()
    print('Done simulation VMF.')


def _test_GME_Estep(K=5, P=200, N=8, num_sub=10, max_iter=100,
                    sigma2=1.0, beta=1.0):
    """Testing different E_step of the GME model
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of data dimensions
        num_sub: the number of subject in the simulation
        max_iter: the maximum iteration number for the EM procedure
        sigma2: the sigma squared used for emission model
        beta: the beta used for the emission model
    Returns:
        Speed comparison
    """
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
    LL1 = em1.Estep_linspace(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    t = time.time()
    LL2 = em2.Estep_import(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    pass

def _param_recovery_GME(K=5, P=20, N=20, num_sub=10, max_iter=100,
        sigma2=[0.1,0.5,1.0,1.5,2.0], beta=[0.1,0.5,1.0,2.0,10.0], num_bins=100, std_V=True,
        type_estep='linspace',num_iter =10):
    """Simulation function used for testing full model with a GMM_exp emission
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of dimensions
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        sigma2: the sigma2 for GMM_exp emission model
        beta: the beta for GMM_exp emission model
    Returns:
        Several evaluation plots.
    """
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins, std_V=std_V)
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)

    D= pd.DataFrame()
    for s in sigma2:
        for b in beta:
            for n in range(num_iter):
                dd = {}
                print(f'{s} {b}')
                T.emission.sigma2 = pt.tensor(s)
                T.emission.beta = pt.tensor(b)
                # Step 2: Generate data by sampling from the above model
                Y, signal = emissionT.sample(U, return_signal=True)

                # Step 3: Compute the log likelihood from the true model
                Uhat_true, loglike_true = T.Estep(Y=Y)
                theta_true = T.get_params()

                # Step 4: Generate new models for fitting
                arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
                emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins, std_V=std_V,type_estep=type_estep)
                emissionM.std_V = std_V
                # emissionM.V =emissionT.V
                # emissionM.sigma2 =emissionT.sigma2
                # emissionM.beta =emissionT.beta
                M = FullModel(arrangeM, emissionM)

                # Step 5: Estimate the parameter thetas to fit the new model using EM
                M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

                ind = M.get_param_indices('emission.sigma2')
                dd['sigma2_true'] = [s]
                dd['beta_true'] = [b]
                dd['sigma2_hat'] = [M.emission.sigma2.item()]
                dd['beta_hat'] = [M.emission.beta.item()]
                D=pd.concat([D,pd.DataFrame(dd)],ignore_index=True)
    return D


def _full_comparison_emission(data_type='GMM', num_sub=10, P=1000, K=5, N=20, beta=1.0,
                              dispersion=2.0, max_iter=100, tol=0.00001, do_plotting=False,same_signal=True):
    """The evaluation and comparison routine between the emission models
    Args:
        data_type: which model used to generate data (GMM, GME, VMF)
        num_sub: the number of subjects
        P: the number of voxels
        K: the number of clusters
        N: the number of data dimensions
        beta: the parameter beta for the signal strength
        max_iter: the maximum number of iteration for EM procedure
        tol: the tolerance of EM iterations
        do_plotting: if True, plot the fitting results
    Returns:
        the evaluation results across the emission models given the criterion
        shape of (num_criterion, num_emissionModels)
    """
    # Step 1. generate the training dataset from VMF model given a signal length
    Y_train, Y_test, signal_true, U, MT = generate_data(data_type, k=K, dim=N, p=P, dispersion=dispersion,
                                                        beta=beta,  do_plot=False,same_signal=same_signal)
    model=['GMM','GME','VMF','true']

    # Step 2. Fit the competing emission model using the training data
    emissionM = []
    emissionM.append(MixGaussian(K=K, N=N, P=P, std_V=False))
    emissionM.append(MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True))
    emissionM.append(MixVMF(K=K, N=N, P=P, uniform_kappa=True))
    M = []
    Uhat_train = [] # Probability of assignments
    V_train = [] # Predicted mean directions
    T = pd.DataFrame()
    for i in range(len(model)):
        if model[i]=='true':
            M.append(MT)
            Uhat,ll = MT.Estep(Y_train)
        else:
            M.append(FullModel(MT.arrange, emissionM[i]))
            M[i], _, _, Uhat = M[i].fit_em(Y=Y_train, iter=max_iter, tol=tol, fit_arrangement=False)
        Uhat_train.append(Uhat)
        V_train.append(M[i].emission.V)

        # Step 4. evaluate the emission model (freezing arrangement model) by a given criterion.
        criterion = ['nmi', 'ari', 'coserr_E','coserrA_E']
        D={}
        D['data_type']=[data_type]
        D['K']=[K]
        D['model']=model[i]
        for c in criterion:
            if c in ['nmi','ari']:
                D[c] = [ev.evaluate_U(U, Uhat_train[i], crit=c)]
            elif c in ['coserr_E']: # expected cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=False,soft_assign=True)]
            elif c in ['coserr_H']: # hard assigned cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=False,soft_assign=False)]
            elif c in ['coserrA_E']: # expected adjusted cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=True,soft_assign=True)]
            elif c in ['coserrA_H']: # hard assigned adjusted cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=True,soft_assign=False)]
        T=pd.concat([T,pd.DataFrame(D)])
    # Step 3.5. Do plot of the clustering results if required
    if do_plotting:
        fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=["True", "GMM", "GME", "VMF"])
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat_train[0], dim=1)[0])), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat_train[1], dim=1)[0])), row=2, col=1)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat_train[2], dim=1)[0])), row=2, col=2)
        fig.update_layout(title_text='Comparison of fitting', height=800, width=800)
        fig.show()

    return T


def do_full_comparison_emission(clusters=5, iters=2, N=20, P=500, subjs=10, beta=0.4,
                                true_models=['GMM', 'GME', 'VMF'], disper=[0.1, 0.1, 18],
                                same_signal=True):
    D = pd.DataFrame()
    for m, e in enumerate(true_models):
        for i in range(iters):
            # beta is to control the signal strength for VMF, sigma2 is for GMM and GME
            T = _full_comparison_emission(data_type=e, num_sub=subjs, P=P, K=clusters,
                                          N=N, beta=beta, dispersion=disper[m],
                                          same_signal=same_signal)
            D = pd.concat([D, T])
    return D


def plot_comparision_emission(T, criterion=['nmi', 'ari', 'coserr_E', 'coserrA_E'],
                              true_models=['GMM', 'GME', 'VMF']):
    num_rows = len(criterion)
    num_cols = len(true_models)
    fig1, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharey='row')
    for i in range(num_rows):
        for j in range(num_cols):
            plt.sca(axs[i, j])
            ind = (T.data_type == true_models[j]) & (T.model != 'true')
            sb.barplot(data=T[ind], x='model', y=criterion[i])
            axs[i][0].set_ylabel(criterion[i])
            axs[0][j].set_title(true_models[j])
            ind = (T.data_type == true_models[j]) & (T.model == 'true')
            plt.axhline(y=T[ind][criterion[i]].mean(), linestyle=':', color='k')


# if __name__ == '__main__':
#     _simulate_full_VMF(K=5, P=100, N=20, num_sub=10, max_iter=100, uniform_kappa=False)
#     _simulate_full_GMM(K=5, P=500, N=20, num_sub=10, max_iter=100, sigma2=10)
#     _simulate_full_GME(K=5, P=200, N=20, num_sub=5, max_iter=100, sigma2=0.1, beta=2.0,
#                        num_bins=100, std_V=True)
#     T = _param_recovery_GME(K=5, P=200, N=20, num_sub=5, max_iter=100, num_bins=300,
#                             std_V=True, num_iter=5, type_estep='import')
#     pass
#
#     T = do_full_comparison_emission(clusters=5, iters=10, beta=0.4, true_models=['GMM', 'GME', 'VMF'],
#                                     disper=[0.1, 0.1, 18], same_signal=True)
#     T.to_csv('notebooks/emission_modelrecover_2.csv')
#     T=pd.read_csv('notebooks/emission_modelrecover_2.csv')
#     plot_comparision_emission(T)
#
#     plt.subplot(2,1,1)
#     sb.lineplot(data=T,x='sigma2_true',y='sigma2_hat',hue='beta_true')
#     plt.subplot(2,1,2)
#     sb.lineplot(data=T,x='beta_true',y='beta_hat',hue='sigma2_true')
#     plt.show()
#     pass