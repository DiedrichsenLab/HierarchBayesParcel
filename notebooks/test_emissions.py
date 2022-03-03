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
import seaborn as sb

def _plot_loglike(loglike, true_loglike, color='b'):
    """Plot the log-likelihood curve and the true log-likelihood
    Args:
        loglike: The log-likelihood curve of the EM iterations
        true_loglike: The true log-likelihood from the true model
        color: the color of the log-likelihood curve

    Returns:
        The plot
    """
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
    """Plot the single estimated parameter array and the true parameter
    Args:
        theta_true: the true parameter
        theta: the estimated parameter
        name: the name of the plotted parameter

    Returns:
        The plot
    """
    plt.figure()
    if name is not None:
        plt.title('True %s (red) vs estimated %s (blue)' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


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


def generate_data(emission, k=2, dim=3, p=1000, num_sub=10, sigma2=1.2,
                  beta=1.0, kappa=15, signal_strength=None, do_plot=False):
    """Generate (and plots) the generated data from a given emission model
    Args:
        emission: 0-GMM, 1-GMM_exp, 2-GMM_gamma, 3-VMF
                  is VMF, the input argument signal_strength must be given
        k: The number of clusters
        dim: The number of data dimensions
        p: the number of generated dat points
        num_sub: the number of subjects
        signal_strength: if None, no precomputed signal strength will be passed;
                         Otherwise, use precomputed signal strength to generate data from VMF
    Returns:
        The generated data
    """
    # Step 1: Create the true model and initialize parameters
    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False, remove_redundancy=False)
    if emission == 0:  # GMM
        emissionT = MixGaussian(K=k, N=dim, P=p)
        emissionT.sigma2 = pt.tensor(sigma2)
    elif emission == 1:  # GMM with exponential signal strength
        emissionT = MixGaussianExp(K=k, N=dim, P=p)
        emissionT.sigma2 = pt.tensor(sigma2)
        emissionT.beta = pt.tensor(beta)
    elif emission == 2:  # GMM with gamma signal strength
        emissionT = MixGaussianGamma(K=k, N=dim, P=p)
        emissionT.beta = pt.tensor(beta)
    elif emission == 3:
        if signal_strength is None:
            raise ValueError("A signal strength must be given for generating data from VMF.")
        emissionT = MixVMF(K=k, N=dim, P=p)
        emissionT.kappa = pt.tensor(kappa)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), or 3(VMF).")
    MT = FullModel(arrangeT, emissionT)

    # Step 2: Generate data by sampling from the true model
    U = MT.arrange.sample(num_subj=num_sub)
    if emission == 1 or emission == 2:
        Y_train, signal = MT.emission.sample(U, return_signal=True)
        Y_test = MT.emission.sample(U, signal=signal, return_signal=False)
    elif emission == 3 and signal_strength is not None:
        Y_train = MT.emission.sample(U)
        Y_test = MT.emission.sample(U)
        Y_train = Y_train * signal_strength.unsqueeze(1).repeat(1, dim, 1)
        Y_test = Y_test * signal_strength.unsqueeze(1).repeat(1, dim, 1)
        signal = signal_strength
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
    emissionT = MixGaussian(K=K, N=N, P=P)
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
    # signal = pt.distributions.exponential.Exponential(0.5).sample((num_sub, P))
    # Ys = Y * signal.unsqueeze(1).repeat(1, N, 1)
    #
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    #                     subplot_titles=["Raw VMF", "Raw VMF with signal strength", "fit"])
    #
    # fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
    #                            mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=1)
    # fig.add_trace(go.Scatter3d(x=Ys[0, 0, :], y=Ys[0, 1, :], z=Ys[0, 2, :],
    #                            mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=2)

    M, ll, theta, Uhat_fit = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)
    # fig.add_trace(go.Scatter3d(x=Ys[0, 0, :], y=Ys[0, 1, :], z=Ys[0, 2, :],
    #                            mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat_fit, dim=1)[0])), row=1, col=3)
    #
    # fig.update_layout(title_text='Comparison of data and fitting')
    # fig.show()

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


def _simulate_full_GMM_from_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50,sigma2=1.0):
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
    signal = pt.distributions.exponential.Exponential(0.5).sample((num_sub, P))
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
        sigma2=1.0, beta=1.0, num_bins=100):
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
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins)
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
    emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins)
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
    LL1 = em1.Estep(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    t = time.time()
    LL2 = em2.Estep_old(Y=Y)
    print(f"time 2:{time.time()-t:.5f}")
    pass


if __name__ == '__main__':
    # _simulate_full_VMF(K=5, P=1000, N=20, num_sub=10, max_iter=100, uniform_kappa=False)
    _simulate_full_GMM(K=5, P=500, N=20, num_sub=10, max_iter=100)
    # _simulate_full_GME(K=7, P=200, N=20, num_sub=10, max_iter=50,sigma2=2.0,beta=1.0,num_bins=100)
    pass
    # _test_GME_Estep(P=500)