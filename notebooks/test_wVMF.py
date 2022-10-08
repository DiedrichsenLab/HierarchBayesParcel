#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test weigted VMF model

Created on 10/6/2022 at 5:20 PM
Author: dzhi
"""
# general import packages
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

# for testing and evaluating models
import os
from full_model import FullModel
from arrangements import ArrangeIndependent, expand_mn
from emissions import MixGaussianExp, MixVMF, wMixVMF
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import evaluation as ev
import scipy.stats as spst

def _simulate_VMF_and_wVMF_from_GME(K=5, P=100, N=40, num_sub=10, max_iter=50,
                                    beta=0.5, sigma2=1.0, uniform_kappa=True, plot=False):
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
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True)
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
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionM = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa)
    emVMF = MixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa)
    M1 = FullModel(arrangeM, emissionM)
    M2 = FullModel(arrangeM, emissionM)
    M_vmf = FullModel(arrangeM, emVMF)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M1, ll1, theta1, Uhat_fit_1 = M1.fit_em(Y=Y, signal=None, iter=max_iter, tol=0.0001,
                                           fit_arrangement=False)

    M2, ll2, theta2, Uhat_fit_2 = M2.fit_em(Y=Y, signal=signal, iter=max_iter, tol=0.0001,
                                           fit_arrangement=False)

    M3, ll3, theta3, Uhat_fit_3 = M_vmf.fit_em(Y=Y, iter=max_iter, tol=0.0001, fit_arrangement=False)

    U_recon_1, this_uerr_1 = ev.matching_U(U, Uhat_fit_1)
    # idx = ev.matching_greedy(expand_mn(pt.clone(U), K)[0], pt.clone(Uhat_fit_1[0]))
    # U_recon_1 = ev.matching_greedy(U, Uhat_fit_1)

    U_recon_2, this_uerr_2 = ev.matching_U(U, Uhat_fit_2)
    U_recon_3, this_uerr_3 = ev.matching_U(U, Uhat_fit_3)
    # print(this_uerr_1, this_uerr_2, this_uerr_3)

    if plot:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        mask = pt.where(signal == 0, signal, pt.tensor(1.0, dtype=pt.float32))
        Y_masked = Y * mask.unsqueeze(1)
        fig = make_subplots(rows=2, cols=4,
                            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
                                   [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=["GME data", "wVMF reconstruction", "wVMF recon - true signal", "standard VMF"])

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                                   row=1, col=1)
        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
                                   row=1, col=2)
        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_2[0])),
                                   row=1, col=3)
        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_3[0])),
                                   row=1, col=4)

        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                      row=2, col=1)
        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers',
                                   marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
                      row=2, col=2)
        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers',
                                   marker=dict(size=3, opacity=0.7, color=U_recon_2[0])),
                      row=2, col=3)
        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers',
                                   marker=dict(size=3, opacity=0.7, color=U_recon_3[0])),
                      row=2, col=4)

        fig.update_layout(title_text='Visualization of data generation')
        fig.show()

    return this_uerr_1, this_uerr_2, this_uerr_3


def _simulate_VMF_and_wVMF_from_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50,
                                    kappa=30, plot=False):
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
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionT = MixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True)
    emissionT.kappa = pt.tensor(kappa)

    # Step 2: Generate data by sampling from the above model
    T = FullModel(arrangeT, emissionT)
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    Uhat_true, loglike_true = T.Estep(Y=Y)
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False,
                                  remove_redundancy=False)
    emissionM = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True)
    M1 = FullModel(arrangeM, emissionM)

    W = pt.tensor(np.random.choice(3, P, p=[0.3, 0.4, 0.3]), dtype=pt.float32)
    signal = W.expand(num_sub, -1)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M1, ll1, theta1, Uhat_fit_1 = M1.fit_em(Y=Y, signal=signal, iter=max_iter, tol=0.0001,
                                            fit_arrangement=False)

    id_0 = (signal == 0).nonzero(as_tuple=True)[1]
    id_1 = (signal == 1).nonzero(as_tuple=True)[1]
    id_2 = (signal == 2).nonzero(as_tuple=True)[1]
    Y_vmf = pt.cat((Y[:, :, id_1], Y[:, :, id_2], Y[:, :, id_2]), 2)
    U_vmf = pt.cat((U[:,id_1], U[:,id_2], U[:,id_2]), 1)

    emVMF = MixVMF(K=K, N=N, P=Y_vmf.shape[2], X=None, uniform_kappa=True)
    M_vmf = FullModel(arrangeM, emVMF)
    M_vmf.emission.V = M1.emission.V
    M_vmf.emission.kappa = M1.emission.kappa
    M2, ll2, theta2, Uhat_fit_2 = M_vmf.fit_em(Y=Y_vmf, iter=max_iter, tol=0.0001,
                                               fit_arrangement=False)

    U_recon_1, this_uerr_1 = ev.matching_U(U, Uhat_fit_1)
    U_recon_2, this_uerr_2 = ev.matching_U(U_vmf, Uhat_fit_2)

    # print(this_uerr_1, this_uerr_2)

    if plot:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        mask = pt.where(signal == 0, signal, pt.tensor(1.0, dtype=pt.float32))
        Y_masked = Y * signal.unsqueeze(1) * mask.unsqueeze(1)

        fig = make_subplots(rows=2, cols=3,
                            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
                                   [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=["raw VMF data + signals", "wVMF recon", "standard VMF"])

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                                   row=1, col=1)
        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
                                   row=1, col=2)
        fig.add_trace(go.Scatter3d(x=Y_vmf[0, 0, :], y=Y_vmf[0, 1, :], z=Y_vmf[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_2[0])),
                                   row=1, col=3)

        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                      row=2, col=1)
        fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                                   mode='markers',
                                   marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
                      row=2, col=2)
        fig.add_trace(go.Scatter3d(x=Y_vmf[0, 0, :], y=Y_vmf[0, 1, :], z=Y_vmf[0, 2, :],
                                   mode='markers',
                                   marker=dict(size=3, opacity=0.7, color=U_recon_2[0])),
                      row=2, col=3)

        fig.update_layout(title_text='Visualization of data generation')
        fig.show()

    return this_uerr_1, this_uerr_2


if __name__ == '__main__':
    A, B ,C, D = [], [], [], []
    for i in range(50):
        a, b, c = _simulate_VMF_and_wVMF_from_GME(K=5, P=5000, N=20, num_sub=1, max_iter=100,
                                                  beta=0.5, sigma2=0.001, plot=False)
        # a, b, = _simulate_VMF_and_wVMF_from_VMF(K=5, P=5000, N=20, num_sub=1, max_iter=100,
        #                                              kappa=30, plot=True)
        A.append(a)
        B.append(b)
        C.append(c)

    A = pt.stack(A).reshape(-1)
    B = pt.stack(B).reshape(-1)
    C = pt.stack(C).reshape(-1)
    # D = pt.stack(D).reshape(-1)

    plt.bar(['wVMF', 'wVMF+true signal', 'VMF'], [A.mean(), B.mean(), C.mean()], yerr=[A.std(), B.std(), C.std()])
    plt.show()