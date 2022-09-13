#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/15/2022
The script the test different emission models standalone

Author: DZHI
"""
# general import packages
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# for testing and evaluating models
import evaluation
import os
from full_model import FullModel
from arrangements import ArrangeIndependent, expand_mn
from emissions import MixGaussianExp, MixGaussian, MixGaussianGamma, MixVMF, wMixVMF
import time
import copy
import seaborn as sb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import evaluation as ev
import pandas as pd
import h5py
import scipy.io as spio
import nibabel as nib
from SUITPy import flatmap, make_label_gifti
import scipy.stats as spst
from scipy.cluster.vq import kmeans, vq


def convert_to_vol(data, xyz, voldef):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        data (list or 1d numpy array)
            voxel data, shape (num_vox, )
        xyz (nd-array)
            3 x P array world coordinates of voxels
        voldef (nib obj)
            nib obj with affine
    Returns:
        list of Nib Obj

    """
    # get dat, mat, and dim from the mask
    dim = voldef.shape
    mat = voldef.affine

    # xyz to ijk
    ijk = flatmap.coords_to_voxelidxs(xyz, voldef)
    ijk = ijk.astype(int)

    vol_data = np.zeros(dim)
    vol_data[ijk[0],ijk[1],ijk[2]] = data

    # convert to nifti
    nib_obj = nib.Nifti1Image(vol_data, mat)
    return nib_obj


def convert_cerebellum_to_nifti(data):
    """
    Args:
        data (np-arrray): N x 6937 length data array
        or 1-d (6937,) array
    Returns:
        nifti (List of nifti1image): N output images
    """
    # Load the region file
    region = spio.loadmat('../data/group/regions_cerebellum_suit.mat')["R"][0][0][0][0][4]

    # NII File for volume definition
    nii_suit = nib.load('../data/group/cerebellarGreySUIT3mm.nii')

    # Map the data
    nii_mapped = []
    if data.ndim == 2:
        for i in range(data.shape[0]):
            nii_mapped.append(convert_to_vol(data[i], region.T, nii_suit))
    elif data.ndim == 1:
        nii_mapped.append(convert_to_vol(data, region.T, nii_suit))
    else:
        raise(NameError('data needs to be 1 or 2-dimensional'))
    return nii_mapped


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
                  beta=1.0, do_plot=False, same_signal=True, missingdata=None):
    """Generate (and plots) the generated data from a given emission model
    Args:
        emission: GMM, GME, GMG, VMF
        k: The number of clusters
        dim: The number of data dimensions
        p: the number of generated data points
        num_sub: the number of subjects
        dispersion: Sigma2 or kappa for distribution
        beta (float): the beta parameter for GME model
        do_plot: plot the generated data if necessary
        same_signal (bool): Is signal strength the same across training and test?
        missingdata: the portion of missing data if applied
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
        emissionT = MixVMF(K=k, N=dim, P=p, X=None, uniform_kappa=True)
        emissionT.kappa = pt.tensor(dispersion)
    elif emission == 'wVMF':
        emissionT = wMixVMF(K=k, N=dim, P=p, X=None, uniform_kappa=True)
        emissionT.kappa = pt.tensor(dispersion)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), "
                         "or 3(VMF).")
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
        # 1. The SNR is generated from exponential distribution
        # signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, 1, p))
        # signal[signal > 1] = 1  # Trim SNR to 1 if > 1

        # 2. The (SNR) ~ bernoulli(80% - 0.1, 20% - 1)
        # signal = pt.distributions.bernoulli.Bernoulli(0.2).sample((num_sub, p))
        # signal[signal == 0] = 0.05
        # signal = signal.unsqueeze(1)

        # 3. The (SNR) ~ (80% no signal - 0.01, 20% full signal - 1)
        signal = pt.ones((p,)) * 0.5
        signal[pt.randint(p, (int(p * 0.2),))] = 1
        signal = signal.unsqueeze(0).unsqueeze(1)

        Y_train = MT.emission.sample(U)
        Y_test = MT.emission.sample(U)
        Y_train = Y_train * signal
        if not same_signal:
            signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, 1, p))
        Y_test = Y_test * signal
    elif emission == 'wVMF':
        Y_train = MT.emission.sample(U)
        Y_test = MT.emission.sample(U)
        signal = MT.emission.W
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

    # Making incomplete dataset if applied
    if missingdata is not None:
        idx = pt.randint(0, p-1, (num_sub, int(missingdata*p)))
        Y_train = pt.transpose(Y_train, 1, 2)
        Y_train[pt.arange(Y_train.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y_train = pt.transpose(Y_train, 1, 2)

    return Y_train, Y_test, signal, U, MT


def _simulate_full_GMM(X, K=5, P=100, N=40, num_sub=10, max_iter=50,sigma2=1.0,missingdata=None):
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
    emissionT = MixGaussian(K=K, N=N, P=P, X=X, std_V=False)
    emissionT.sigma2 = pt.tensor(sigma2)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)
    prior = expand_mn(U, K) * 7.0
    prior = prior.softmax(dim=1)

    # Step 3: Compute the true log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y)
    theta_true = T.get_params()

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussian(K=K, N=N, P=P, X=X, std_V=False)
    emissionM.initialize(Y, X=X)
    emissionM.Mstep(prior)

    M = FullModel(arrangeM, emissionM)
    M, ll, theta, Uhat_fit = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Plot fitting results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    true_V = theta_true[M.get_param_indices('emission.V')].reshape(emissionM.M, K)
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(axs[2], np.log(theta_true[ind]),
                           np.log(theta[:, ind]), name='log sigma2')

    fig.suptitle('GMM fitting results, run = %d' % int(emissionT.X.shape[0]/emissionT.X.shape[1]))
    plt.tight_layout()
    plt.show()
    print('Done simulation GMM.')


def _simulate_full_GME(X, K=5, P=1000, N=20, num_sub=10, max_iter=100,
        sigma2=1.0, beta=1.0, num_bins=100, std_V=True,
        type_estep='linspace', missingdata=None):
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
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, X=X, num_signal_bins=num_bins,
                               std_V=std_V)
    emissionT.sigma2 = pt.tensor(sigma2)
    emissionT.beta = pt.tensor(beta)
    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U, return_signal=True)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)
    theta_true = T.get_params()

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionM = MixGaussianExp(K=K, N=N, P=P, X=X, num_signal_bins=num_bins,
                               std_V=std_V, type_estep=type_estep)
    emissionM.std_V = std_V
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, Uhat_fit = M.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

    # Plotfitting results
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    true_V = theta_true[M.get_param_indices('emission.V')].reshape(emissionM.M, K)
    predicted_V = theta[:, M.get_param_indices('emission.V')]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.sigma2')
    _plt_single_param_diff(axs[2], np.log(theta_true[ind]),
                           np.log(theta[:, ind]), name='log sigma2')

    ind = M.get_param_indices('emission.beta')
    _plt_single_param_diff(axs[3], theta_true[ind],theta[:, ind], name='beta')
    # SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=True, soft_assign=False)

    fig.suptitle('GME fitting results')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sb.scatterplot(x=Y.squeeze(dim=0)[0], y=Y.squeeze(dim=0)[1], hue=U[0], palette='tab10')
    plt.subplot(1, 2, 2)
    sb.scatterplot(x=Y.squeeze(dim=0)[0], y=Y.squeeze(dim=0)[1], hue=Uhat_fit.squeeze(0).argmax(0),
                   palette='tab10')
    plt.show()

    print('Done simulation GME.')


def _simulate_full_VMF(X, K=5, P=100, N=40, num_sub=10, max_iter=50,
                       uniform_kappa=True, missingdata=None, n_inits=None):
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
    emissionT = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)  # Run only Estep!
    theta_true = T.get_params()

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Multiple random initializations to check local maxima if required
    if n_inits is not None:
        max_ll = -pt.inf
        for i in range(n_inits):
            this_emissionM = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)
            this_M = FullModel(arrangeM, this_emissionM)
            # Step 5: Estimate the parameter thetas to fit the new model using EM
            this_M, this_ll, this_theta, _ = this_M.fit_em(Y=Y, iter=max_iter, tol=0.00001,
                                                           fit_arrangement=False)
            if this_ll[-1] > max_ll:
                M = this_M
                max_ll = this_ll[-1]
                ll = this_ll
                theta = this_theta
                emissionM = this_emissionM

    # Plot fitting results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    ind = M.get_param_indices('emission.V')
    true_V = theta_true[ind].reshape(emissionM.M, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.kappa')
    if uniform_kappa or emissionT.uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[2], theta_true[T.get_param_indices('emission.kappa')],
                               theta[:, ind], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[ind].reshape(1, K), theta[:, ind], once=False)
        _plot_diff(axs[2], theta_true[ind].reshape(1, K), theta[:, ind], index=idx, name='kappa')

    fig.suptitle('VMF fitting results, run = %d' % int(emissionT.X.shape[0]/emissionT.X.shape[1]))
    plt.tight_layout()
    plt.show()
    print('Done simulation VMF.')


def _simulate_full_wVMF(X, K=5, P=100, N=40, num_sub=10, max_iter=50,
                       uniform_kappa=True, missingdata=None, n_inits=None):
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
    emissionT = wMixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)  # Run only Estep!
    theta_true = T.get_params()

    if missingdata is not None:
        idx = pt.randint(0, P-1, (num_sub, int(missingdata*P)))
        Y = pt.transpose(Y, 1, 2)
        Y[pt.arange(Y.shape[0]).unsqueeze(-1), idx] = pt.nan
        Y = pt.transpose(Y, 1, 2)

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = wMixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)
    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001, fit_arrangement=False)

    # Multiple random initializations to check local maxima if required
    if n_inits is not None:
        max_ll = -pt.inf
        for i in range(n_inits):
            this_emissionM = MixVMF(K=K, N=N, P=P, X=X, uniform_kappa=uniform_kappa)
            this_M = FullModel(arrangeM, this_emissionM)
            # Step 5: Estimate the parameter thetas to fit the new model using EM
            this_M, this_ll, this_theta, _ = this_M.fit_em(Y=Y, iter=max_iter, tol=0.00001,
                                                           fit_arrangement=False)
            if this_ll[-1] > max_ll:
                M = this_M
                max_ll = this_ll[-1]
                ll = this_ll
                theta = this_theta
                emissionM = this_emissionM

    # Plot fitting results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_loglike(axs[0], ll, loglike_true, color='b')

    ind = M.get_param_indices('emission.V')
    true_V = theta_true[ind].reshape(emissionM.M, K)
    predicted_V = theta[:, ind]
    idx = matching_params(true_V, predicted_V, once=False)
    _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

    ind = M.get_param_indices('emission.kappa')
    if uniform_kappa or emissionT.uniform_kappa:
        # Plot if kappa is uniformed
        _plt_single_param_diff(axs[2], theta_true[T.get_param_indices('emission.kappa')],
                               theta[:, ind], name='kappa')
    else:
        # Plot if kappa is not uniformed
        idx = matching_params(theta_true[ind].reshape(1, K), theta[:, ind], once=False)
        _plot_diff(axs[2], theta_true[ind].reshape(1, K), theta[:, ind], index=idx, name='kappa')

    fig.suptitle('VMF fitting results, run = %d' % int(emissionT.X.shape[0]/emissionT.X.shape[1]))
    plt.tight_layout()
    plt.show()
    print('Done simulation wVMF.')


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


def _simulate_full_GME_from_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50,
                                beta=0.5, sigma2=1.0, plot=False):
    """Simulation function used for testing GME model recovery from VMF data
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
    emissionT.kappa = pt.tensor(sigma2)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)
    signal = pt.distributions.exponential.Exponential(beta).sample((num_sub, P))
    Ys = Y * signal.unsqueeze(1).repeat(1, N, 1)

    # Step 3: Compute the true log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y)
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM1 = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True)
    emissionM2 = MixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True)
    M1 = FullModel(arrangeM, emissionM1)
    M2 = FullModel(arrangeM, emissionM2)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M1, ll_1, theta_1, Uhat_fit_1 = M1.fit_em(Y=Ys, iter=max_iter, tol=0.00001,
                                              fit_arrangement=False)
    M2, ll_2, theta_2, Uhat_fit_2 = M2.fit_em(Y=Ys, iter=max_iter, tol=0.00001,
                                              fit_arrangement=False)

    # Step 4. evaluate the emission model (freezing arrangement model) by a given criterion.
    D = pd.DataFrame()
    _, this_uerr_1 = ev.matching_U(U, Uhat_fit_1)
    _, this_uerr_2 = ev.matching_U(U, Uhat_fit_2)
    D['uerr_GME'] = [this_uerr_1.mean().item()]
    D['uerr_VMF'] = [this_uerr_2.mean().item()]

    if plot:
        # Plot fitting results
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        _plot_loglike(axs[0], ll_1, loglike_true, color='b')
        _plot_loglike(axs[1], ll_2, loglike_true, color='b')
        fig.suptitle('GME fitting results from VMF.')
        plt.tight_layout()
        plt.show()

    print('Done simulation GME from VMF.')
    return D


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
                        sigma2=[0.1,0.5,1.0,1.5,2.0], beta=[0.1,0.5,1.0,2.0,10.0],
                        num_bins=100, std_V=True, type_estep=['linspace'], num_iter=10):
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
            for t in type_estep:
                for n in range(num_iter):
                    dd = {}
                    # print(f'sigma={s}, beta={b}, iter={n}, type={t}')
                    T.emission.sigma2 = pt.tensor(s)
                    T.emission.beta = pt.tensor(b)
                    # Step 2: Generate data by sampling from the above model
                    Y, signal = emissionT.sample(U, return_signal=True)

                    # Step 3: Compute the log likelihood from the true model
                    Uhat_true, loglike_true = T.Estep(Y=Y)
                    theta_true = T.get_params()

                    # Step 4: Generate new models for fitting
                    emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins,
                                               std_V=std_V, type_estep=t)
                    M = FullModel(arrangeT, emissionM)

                    # Step 5: Estimate the parameter thetas to fit the new model using EM
                    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001,
                                               fit_arrangement=False)

                    ind = M.get_param_indices('emission.sigma2')
                    dd['type'] = [t]
                    dd['sigma2_true'] = [s]
                    dd['beta_true'] = [b]
                    dd['sigma2_hat'] = [M.emission.sigma2.item()]
                    dd['beta_hat'] = [M.emission.beta.item()]
                    D = pd.concat([D, pd.DataFrame(dd)], ignore_index=True)
    return D


def _test_sampling_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100, sigma2=1.0,
                       beta=1.0, num_bins=100, std_V=True, type_estep=['linspace']):
    """ Comparing the GME model recovery using different sampling methods
    Args:
        K: the number of clusters
        P: the number of data points
        N: the number of dimensions
        num_sub: the number of subject to simulate
        max_iter: the maximum iteration for EM procedure
        sigma2: the sigma2 for GMM_exp emission model
        beta: the beta for GMM_exp emission model
        num_bins: the sampling size
        std_V: standardize V's if set to True. Otherwise, no standardization
        type_estep: specify the sampling methods
    Returns:
        Several evaluation plots to compare the results.
    """
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins,
                               std_V=std_V)
    emissionT.sigma2 = pt.tensor(sigma2)
    emissionT.beta = pt.tensor(beta)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U, return_signal=True)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)
    theta_true = T.get_params()

    # Step 4: Generate new models using different sampling for fitting
    for type in type_estep:
        emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=num_bins,
                                   std_V=std_V, type_estep=type)
        emissionM.std_V = std_V
        M = FullModel(arrangeT, emissionM)

        # Step 5: Estimate the parameter thetas to fit the new model using EM
        M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

        # Step 6: Plot fitting results
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        # fig 1: true log-likelihood vs. predicted log-likelihood curve
        _plot_loglike(axs[0], ll, loglike_true, color='b')

        # fig 2: the plot of differences between true Vs and predicted Vs at each iter
        ind = M.get_param_indices('emission.V')
        true_V = theta_true[ind].reshape(N, K)
        predicted_V = theta[:, ind]
        idx = matching_params(true_V, predicted_V, once=False)
        _plot_diff(axs[1], true_V, predicted_V, index=idx, name='V')

        # fig 3: true sigma2 vs. predicted sigma2
        ind = M.get_param_indices('emission.sigma2')
        _plt_single_param_diff(axs[2], np.log(theta_true[ind]), np.log(theta[:, ind]), name='log sigma2')

        # fig 4: true beta vs. the predicted beta
        ind = M.get_param_indices('emission.beta')
        _plt_single_param_diff(axs[3], theta_true[ind], theta[:, ind], name='beta')

        fig.suptitle('GME fitting results using %s' % type)
        plt.tight_layout()
        plt.show()

    print('Done comparing GME E_step.')


def _full_comparison_emission(data_type='GMM', num_sub=10, P=1000, K=5, N=20, beta=1.0,
                              dispersion=2.0, max_iter=100, tol=0.001, do_plotting=False,
                              same_signal=True, missingdata=0.1):
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
    Y_train, Y_test, signal_true, U, MT = generate_data(data_type, k=K, dim=N, p=P,
                                                        dispersion=dispersion, beta=beta,
                                                        do_plot=False, same_signal=same_signal,
                                                        missingdata=missingdata)
    model=['GMM', 'GME', 'VMF', 'wVMF', 'true']

    # Step 2. Fit the competing emission model using the training data
    emissionM = []
    emissionM.append(MixGaussian(K=K, N=N, P=P, X=None, std_V=False))
    emissionM.append(MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True))
    emissionM.append(MixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True))
    emissionM.append(wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True))
    M = []
    Uhat_train = []  # Probability of assignments
    V_train = []  # Predicted mean directions
    T = pd.DataFrame()
    for i in range(len(model)):
        if model[i] == 'true':
            M.append(MT)
            Uhat, ll = MT.Estep(Y_train)
        else:
            M.append(FullModel(MT.arrange, emissionM[i]))
            M[i], this_ll, _, Uhat = M[i].fit_em(Y=Y_train, iter=max_iter, tol=tol, fit_arrangement=False)
        Uhat_train.append(Uhat)
        V_train.append(M[i].emission.V)

        # Step 4. evaluate the emission model (freezing arrangement model) by a given criterion.
        criterion = ['coserr_E', 'coserrA_E', 'Uerr']
        D = {}
        D['data_type'] = [data_type]
        D['K'] = [K]
        D['model'] = model[i]
        for c in criterion:
            if c in ['nmi', 'ari']:
                D[c] = [ev.evaluate_U(U, Uhat_train[i], crit=c)]
            elif c in ['coserr_E']:  # expected cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=False,soft_assign=True).mean().item()]
            elif c in ['coserr_H']: # hard assigned cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=False,soft_assign=False).mean().item()]
            elif c in ['coserrA_E']: # expected adjusted cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=True,soft_assign=True).mean().item()]
            elif c in ['coserrA_H']: # hard assigned adjusted cosine error
                D[c]=[ev.coserr(Y_test,V_train[i],Uhat_train[i],adjusted=True,soft_assign=False).mean().item()]
            elif c in ['Uerr']:  # absolute prediction error
                _, this_uerr = ev.matching_U(U, Uhat_train[i])
                D[c]=[this_uerr.mean().item()]
            elif c in ['homogeneity']:
                U_group = pt.sum(Uhat_train[i], dim=0)  # sum across subjects
                U_group = U_group / pt.sum(U_group, dim=0, keepdim=True)  # normalization to 1
                D[c]=[ev.inhomogeneity_task(Y_test,U_group,z_transfer=False, single_return=True)]
        T=pd.concat([T,pd.DataFrame(D)])
    # Step 3.5. Do plot of the clustering results if required
    if do_plotting:
        fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}],
                                                   [{'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=["True", "GMM", "GME", "VMF", "wVMF"])
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                      row=1, col=1)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7,
                                                               color=pt.argmax(Uhat_train[0], dim=1)[0])),
                      row=1, col=2)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7,
                                                               color=pt.argmax(Uhat_train[1], dim=1)[0])),
                      row=2, col=1)
        fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7,
                                                               color=pt.argmax(Uhat_train[2], dim=1)[0])),
                      row=2, col=2)
        fig.update_layout(title_text='Comparison of fitting', height=800, width=800)
        fig.show()

    return T


def do_full_comparison_emission(clusters=5, iters=2, N=20, P=1000, subjs=10, beta=0.4,
                                true_models=['GMM', 'GME', 'VMF'], disper=[0.1, 0.1, 18],
                                same_signal=True, missingdata=0.1):
    D = pd.DataFrame()
    for m, e in enumerate(true_models):
        for i in range(iters):
            # beta is to control the signal strength for VMF, sigma2 is for GMM and GME
            T = _full_comparison_emission(data_type=e, num_sub=subjs, P=P, K=clusters,
                                          N=N, beta=beta, dispersion=disper[m],
                                          same_signal=same_signal, missingdata=missingdata)
            D = pd.concat([D, T])
            # print('Done iter=', i)
    return D


def plot_comparision_emission(T, K=5, criterion=['coserr_E', 'coserrA_E', 'Uerr'],
                              true_models=['GMM', 'GME', 'VMF', 'wVMF'],
                              dispersion=[0.1, 0.1, 18, 18]):
    num_rows = len(criterion)
    num_cols = len(true_models)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharey='row')
    for i in range(num_rows):
        for j in range(num_cols):
            plt.sca(axs[i, j])
            ind = (T.data_type == true_models[j]) & (T.model != 'true')
            sb.barplot(data=T[ind], x='model', y=criterion[i])
            axs[i][0].set_ylabel(criterion[i])
            axs[0][j].set_title(true_models[j]+f' {dispersion[j]}')
            ind = (T.data_type == true_models[j]) & (T.model == 'true')
            plt.axhline(y=T[ind][criterion[i]].mean(), linestyle=':', color='k')
    fig.suptitle('The emission models comparison, k = %d' %K, fontsize=16)


def plot_comparison_samplingGME(T, params_name=['sigma2', 'beta'],
                                type_estep=['linspace', 'import', 'reject']):
    num_rows = len(params_name)
    num_cols = len(type_estep)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4), sharey='row')
    for i in range(num_rows):
        for j in range(num_cols):
            data = T[T['type'] == type_estep[j]]
            plt.sca(axs[i, j])
            sb.lineplot(ax=axs[i, j], data=data, x=params_name[i]+'_true',
                        y=np.abs(data[params_name[i]+'_true'] - data[params_name[i]+'_hat']),
                        hue=params_name[1-i]+'_true', palette='tab10')
            axs[i][j].set_ylabel(params_name[i]+' recovery')
            axs[i][j].set_xlabel(params_name[i])
            axs[i][j].set_title(type_estep[j])

    plt.suptitle('parameter recovery using different E_step (GME)')
    plt.tight_layout()


def getData_from_T(T, crit='Uerr', true_model='GME', fit_model=['GME'], trim_locmin=True):
    D = []
    for fm in fit_model:
        ind = (T.data_type == true_model) & (T.model != 'true') & (T.model == fm)
        data = T[ind][crit].values
        D.append(data)

    if trim_locmin:
        ind_matrix = []
        for d in D:
            # Find a threshold or standard to remove those local minimas
            codebook, _ = kmeans(d, 2)  # Cluster array into two group
            cluster_ind, _ = vq(d, codebook)

            if d[np.where(cluster_ind == 0)].mean() > d[np.where(cluster_ind == 1)].mean():
                cluster_ind = 1 - cluster_ind

            ind_matrix.append(cluster_ind)
        final_idx = np.where(np.asarray(ind_matrix).sum(axis=0) == 0)
        return [this_d[final_idx] for this_d in D]
    else:
        return D


def train_mdtb_dirty(root_dir='Y:/data/Cerebellum/super_cerebellum/sc1/beta_roi/glm7',
                     train_participants=None, test_participants=None, sess=None,
                     K = 10, emission='VMF', max_iter=200):
    subj_name = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
                 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23',
                 's24', 's25', 's26', 's27', 's28', 's29', 's30', 's31']
    goodsubj = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29,
                30, 31]
    if train_participants is None:
        train_participants = [2, 3]
    if test_participants is None:
        test_participants = [4]
    if sess is None:
        sess = 1
    Y_train, Y_test = [], []

    # Preparing training data from MDTB dataset
    for sub in train_participants:
        file = h5py.File(os.path.join(root_dir, f'{subj_name[sub - 1]}/Y_glm7_cerebellum_suit.mat'))
        c = pt.chunk(pt.tensor(np.array(file['data'])), 2, dim=1)
        this_data = pt.stack(pt.chunk(c[sess], 8, dim=1))
        this_data[this_data == 0] = pt.nan
        Y_train.append(this_data.mean(dim=0).T)

    # Preparing test data from MDTB dataset
    for sub in test_participants:
        file = h5py.File(os.path.join(root_dir, f'{subj_name[sub - 1]}/Y_glm7_cerebellum_suit.mat'))
        c = pt.chunk(pt.tensor(np.array(file['data'])), 2, dim=1)
        this_data = pt.stack(pt.chunk(c[sess], 8, dim=1))
        this_data[this_data == 0] = pt.nan
        Y_test.append(this_data.mean(dim=0).T)

    Y_train, Y_test = pt.stack(Y_train), pt.stack(Y_test)
    num_sub_train, N, P = Y_train.shape

    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    if emission == 'GMM':
        emissionM = MixGaussian(K=K, N=N, P=P, std_V=False)
    elif emission == 'GME':
        emissionM = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=False,
                                   type_estep='linspace')
    elif emission == 'VMF':
        emissionM = MixVMF(K=K, N=N, P=P, uniform_kappa=False)
    else:
        raise ValueError('An emission type must be given to train the model.')

    M = FullModel(arrangeM, emissionM)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M, ll, theta, U_hat = M.fit_em(Y=Y_train, iter=max_iter, tol=0.0001, fit_arrangement=True)

    colors = plt.cm.jet(np.linspace(0, 1, K))
    # colors = np.random.uniform(size=(K, 3))
    # colors = np.c_[colors, np.ones(K)]
    labels = U_hat.mean(dim=0).T.detach().numpy()
    G = convert_cerebellum_to_nifti(np.argmax(labels, axis=1))
    G = flatmap.vol_to_surf(G, stats='mode')
    G = make_label_gifti(G, anatomical_struct='Cerebellum', label_names=[],
                         column_names=[], label_RGBA=colors)

    Uhat_test, _ = M.Estep(Y=Y_test)

    D = []
    D.append(ev.coserr(Y_test, M.emission.V, Uhat_test, adjusted=False, soft_assign=True))
    D.append(ev.coserr(Y_test, M.emission.V, Uhat_test, adjusted=False, soft_assign=False))
    D.append(ev.coserr(Y_test, M.emission.V, Uhat_test, adjusted=True, soft_assign=True))
    D.append(ev.coserr(Y_test, M.emission.V, Uhat_test, adjusted=True, soft_assign=False))
    print(D)
    return G, D


if __name__ == '__main__':
    X = pt.eye(46).repeat(10, 1)  # simulate task design matrix X
    # _simulate_full_VMF(X=None, K=10, P=500, N=20, num_sub=10, max_iter=100, uniform_kappa=True,
    #                    missingdata=None, n_inits=None)
    # _simulate_full_wVMF(X=None, K=10, P=500, N=20, num_sub=10, max_iter=100, uniform_kappa=True,
    #                     missingdata=None, n_inits=None)
    # _simulate_full_GMM(X=None, K=5, P=500, N=20, num_sub=10, max_iter=200, sigma2=2.0,
    #                    missingdata=None)
    # _simulate_full_GME(X=None, K=5, P=1000, N=2, num_sub=1, max_iter=100, sigma2=0.1, beta=2.0,
    #                    num_bins=100, std_V=True, type_estep='linspace', missingdata=None)

    # T = pd.DataFrame()
    # for i in range(10):
    #     D = _simulate_full_GME_from_VMF(K=5, P=2000, N=20, num_sub=10, max_iter=100,
    #                                     beta=2.0, sigma2=40, plot=True)
    #     T = pd.concat([T, D])
    # print(T)

    # _test_sampling_GME(K=5, P=200, N=20, num_sub=10, max_iter=100, sigma2=3.0,
    #                    beta=0.5, num_bins=200, std_V=True,
    #                    type_estep=['linspace', 'import', 'import', 'reject', 'mcmc'])
    # T = _param_recovery_GME(K=5, P=200, N=20, num_sub=5, max_iter=100, num_bins=300,
    #                         std_V=True, num_iter=5, sigma2=[0.1, 0.5, 5.0], beta=[0.1, 0.5, 2.0],
    #                         type_estep=['linspace', 'import', 'reject'])
    # plot_comparison_samplingGME(T, type_estep=['linspace', 'import', 'reject'])

    ######### emission completion test #########
    K = 5
    T = do_full_comparison_emission(clusters=K, iters=10, beta=2.0,
                                    true_models=['GMM', 'GME', 'VMF', 'wVMF'],
                                    disper=[0.1, 0.1, 50, 50], same_signal=True, missingdata=None)
    # T.to_csv('emission_modelrecover_k%d_iter50_new.csv' %K)
    # T = pd.read_csv('emission_modelrecover_k%d_iter50_new.csv' %K)
    plot_comparision_emission(T, K=K, dispersion=[0.1, 0.1, 50, 50])
    plt.show()

    # T = pd.read_csv('emission_modelrecover_k5_iter100.csv')
    # D = getData_from_T(T, crit='Uerr', true_model='GME', fit_model=['GME', 'VMF'])
    # spst.ttest_rel(D[0], D[1])
    # pass

    # G, D = train_mdtb_dirty(K=5, train_participants=[2, 3, 4, 6, 8, 9, 10, 12, 14, 15],
    #                         test_participants=[29, 30, 31], emission='VMF')
    # nib.save(G, 'test_5.label.gii')

    print('Done simulation.')
