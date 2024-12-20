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
from full_model import FullModel, FullMultiModel
from arrangements import ArrangeIndependent, expand_mn
from emissions import MixGaussianExp, MixVMF, wMixVMF
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import evaluation as ev
import scipy.stats as spst
from scipy.cluster.vq import kmeans, vq

def _simulate_VMF_and_wVMF_from_GME(K=5, P=100, N=40, num_sub=10, max_iter=50, beta=0.5, sigma2=1.0,
                                    uniform_kappa=True, plot_ll=True, plot_weight=False, plot=False):
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
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True)
    emissionT.sigma2 = pt.tensor(sigma2)
    emissionT.beta = pt.tensor(beta)
    emissionT.num_subj = num_sub

    # Step 2: Generate data by sampling from the above model
    T = FullMultiModel(arrangeT, [emissionT])
    T.initialize()
    U,_ = T.sample()

    W = pt.tensor(np.random.choice(2, P, p=[0.8, 0.2]), dtype=pt.float32)
    Y, signal = emissionT.sample(U, signal=W.expand(num_sub, -1)+0.1, return_signal=True)
    signal_normal = (signal - signal.min()) / (signal.max() - signal.min())

    # Step 3: Compute the log likelihood from the true model
    T.emissions[0].initialize(Y)
    Uhat_true, loglike_true = T.Estep()
    theta_true = T.get_params()

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
    # model 1 - use data magnitude as weights (default)
    emissionM1 = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa, weighting='length')
    emissionM1.num_subj = num_sub
    # model 2 - use normalized data magnitude + density as weights
    emissionM2 = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa, weighting='lsquare')
    emissionM2.num_subj = num_sub
    # model 3 - use true data signal
    emissionM3 = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa)
    emissionM3.num_subj = num_sub
    # model 4 - wVMF as VMF
    emissionM4 = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa, weighting='ones')
    emissionM4.num_subj = num_sub
    # model 5 - true VMF comparison
    emVMF = MixVMF(K=K, N=N, P=P, X=None, uniform_kappa=uniform_kappa)
    emVMF.num_subj = num_sub

    list_M1, list_M2, list_M3, list_M4, list_M0 = [],[],[],[],[]
    lls_1, lls_2, lls_3, lls_4, lls_0 = [],[],[],[],[]
    Ufit_1, Ufit_2, Ufit_3, Ufit_4, Ufit_0 = [], [], [], [], []
    max_1, max_2, max_3, max_4, max_0 = 1,1,1,1,1

    M1 = FullMultiModel(arrangeM, [emissionM1])
    M2 = FullMultiModel(arrangeM, [emissionM2])
    M3 = FullMultiModel(arrangeM, [emissionM3])
    M4 = FullMultiModel(arrangeM, [emissionM4])
    M_vmf = FullMultiModel(arrangeM, [emVMF])

    M1.initialize([Y])
    M2.initialize([Y])
    M3.initialize([Y])
    M3.emissions[0].initialize(Y, weight=signal)
    M4.initialize([Y])
    M_vmf.initialize([Y])

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M1, ll1, _, Uhat_fit_1,_ = M1.fit_em_ninits(n_inits=40, first_iter=20, iter=200, tol=0.01,
                                                fit_emission=True, fit_arrangement=False,
                                                init_emission=True, init_arrangement=True,
                                                align = 'arrange', verbose=False)
    list_M1.append(M1)
    lls_1.append(ll1)
    Ufit_1.append(Uhat_fit_1)

    M2, ll2, _, Uhat_fit_2,_ = M2.fit_em_ninits(n_inits=40, first_iter=20, iter=200, tol=0.01,
                                                fit_emission=True, fit_arrangement=False,
                                                init_emission=True, init_arrangement=True,
                                                align = 'arrange', verbose=False)
    list_M2.append(M2)
    lls_2.append(ll2)
    Ufit_2.append(Uhat_fit_2)

    M3, ll3, _, Uhat_fit_3,_ = M3.fit_em_ninits(n_inits=40, first_iter=20, iter=200, tol=0.01,
                                                fit_emission=True, fit_arrangement=False,
                                                init_emission=True, init_arrangement=True,
                                                align = 'arrange', verbose=False)
    list_M3.append(M3)
    lls_3.append(ll3)
    Ufit_3.append(Uhat_fit_3)

    M4, ll4, _, Uhat_fit_4,_ = M4.fit_em_ninits(n_inits=40, first_iter=20, iter=200, tol=0.01,
                                                fit_emission=True, fit_arrangement=False,
                                                init_emission=True, init_arrangement=True,
                                                align = 'arrange', verbose=False)
    list_M4.append(M4)
    lls_4.append(ll4)
    Ufit_4.append(Uhat_fit_4)

    M0, ll0, _, Uhat_fit_0,_ = M_vmf.fit_em_ninits(n_inits=40, first_iter=20, iter=200, tol=0.01,
                                                   fit_emission=True, fit_arrangement=False,
                                                   init_emission=True, init_arrangement=True,
                                                   align = 'arrange', verbose=False)
    list_M0.append(M0)
    lls_0.append(ll0)
    Ufit_0.append(Uhat_fit_0)

    U_recon_1, this_uerr_1 = ev.matching_U(U, Ufit_1[0])
    U_recon_2, this_uerr_2 = ev.matching_U(U, Ufit_2[0])
    U_recon_3, this_uerr_3 = ev.matching_U(U, Ufit_3[0])
    U_recon_4, this_uerr_4 = ev.matching_U(U, Ufit_4[0])
    U_recon_0, this_uerr_0 = ev.matching_U(U, Ufit_0[0])
    U_recon_true, this_uerr_true = ev.matching_U(U, Uhat_true)
    M1 = list_M1[0]
    M2 = list_M2[0]
    M3 = list_M3[0]
    M4 = list_M4[0]
    M0 = list_M0[0]

    if plot_weight:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.hist(pt.norm(Y, dim=1).reshape(-1).numpy(), bins=100)
        plt.title('data magnitude')
        plt.subplot(132)
        plt.hist(M2.emissions[0].W.reshape(-1).numpy(), bins=100)
        plt.title('wVMF magnitude + density')
        plt.subplot(133)
        plt.hist(signal.reshape(-1).numpy(), bins=100)
        plt.title('True magnitude')

        plt.suptitle('The distribution of the data magnitude')
        plt.show()

    if plot_ll:
        plt.figure(figsize=(20, 4))
        plt.subplot(151)
        plt.plot(ll1)
        plt.title('data magnitude')
        plt.subplot(152)
        plt.plot(ll2)
        plt.title('wVMF magnitude + density')
        plt.subplot(153)
        plt.plot(ll3)
        plt.title('True magnitude')
        plt.subplot(154)
        plt.plot(ll4)
        plt.title('wVMF as VMF')
        plt.subplot(155)
        plt.plot(ll0)
        plt.title('True VMF')

        plt.suptitle('The ll of different models')
        plt.show()

    if plot:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        max_length = Y.norm(dim=1).max()
        Ms = [T, M1, M2, M3, M4]
        x_lines = list()
        y_lines = list()
        z_lines = list()
        # create the coordinate list for the lines
        for i, m in enumerate(Ms):
            V = m.emissions[0].V * max_length
            this_x, this_y, this_z = [], [], []
            for p in range(K):
                this_x.append(0)
                this_y.append(0)
                this_z.append(0)
                this_x.append(V[0, p])
                this_y.append(V[1, p])
                this_z.append(V[2, p])
                this_x.append(None)
                this_y.append(None)
                this_z.append(None)

            x_lines.append(this_x)
            y_lines.append(this_y)
            z_lines.append(this_z)

        mask = pt.where(signal == 0, signal, pt.tensor(1.0, dtype=signal.dtype))
        Y_masked = Y * mask.unsqueeze(1)
        fig = make_subplots(rows=1, cols=4,
                            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=["GME data", "wVMF data mag", "wVMF recon - true signal", "standard VMF"])

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
                                   row=1, col=1)
        fig.add_trace(go.Scatter3d(x=x_lines[0], y=y_lines[0], z=z_lines[0],
                                   mode='lines', line=dict(width=10, color='red')),
                                   row=1, col=1)

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
                                   row=1, col=2)
        fig.add_trace(go.Scatter3d(x=x_lines[1], y=y_lines[1], z=z_lines[1],
                                   mode='lines', line=dict(width=10, color='red')),
                                   row=1, col=2)

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_3[0])),
                                   row=1, col=3)
        fig.add_trace(go.Scatter3d(x=x_lines[3], y=y_lines[3], z=z_lines[3],
                                   mode='lines', line=dict(width=10, color='red')),
                                   row=1, col=3)

        fig.add_trace(go.Scatter3d(x=Y_masked[0, 0, :], y=Y_masked[0, 1, :], z=Y_masked[0, 2, :],
                                   mode='markers', marker=dict(size=3, opacity=0.7, color=U_recon_4[0])),
                                   row=1, col=4)
        fig.add_trace(go.Scatter3d(x=x_lines[4], y=y_lines[4], z=z_lines[4],
                                   mode='lines', line=dict(width=10, color='red')),
                                   row=1, col=4)

        # fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
        #                            mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])),
        #               row=2, col=1)
        # fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
        #                            mode='markers',
        #                            marker=dict(size=3, opacity=0.7, color=U_recon_1[0])),
        #               row=2, col=2)
        # fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
        #                            mode='markers',
        #                            marker=dict(size=3, opacity=0.7, color=U_recon_2[0])),
        #               row=2, col=3)
        # fig.add_trace(go.Scatter3d(x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
        #                            mode='markers',
        #                            marker=dict(size=3, opacity=0.7, color=U_recon_3[0])),
        #               row=2, col=4)

        fig.update_layout(title_text='Visualization of data generation')
        fig.show()

    return this_uerr_1, this_uerr_2, this_uerr_3, this_uerr_4, this_uerr_0, this_uerr_true


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


def _check_VMF_and_wVMF_equivalent(K=5, P=100, N=40, num_sub=10, max_iter=50,
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

    # Step 3: randomly generate true signals of data points
    W = pt.tensor(np.random.choice(3, P, p=[0.3, 0.4, 0.3]), dtype=pt.float32)
    signal = W.expand(num_sub, -1)

    # Step 3.1: Making the same data and its true Us for original VMF
    id_0 = (signal == 0).nonzero(as_tuple=True)[1]
    id_1 = (signal == 1).nonzero(as_tuple=True)[1]
    id_2 = (signal == 2).nonzero(as_tuple=True)[1]
    Y_vmf = pt.cat((Y[:, :, id_1], Y[:, :, id_2], Y[:, :, id_2]), 2)
    U_vmf = pt.cat((U[:,id_1], U[:,id_2], U[:,id_2]), 1)

    # Step 4: Generate new emission models for fitting
    em_WVMF = wMixVMF(K=K, N=N, P=P, X=None, uniform_kappa=True)

    # Create VMF for fitting and align all params to the same to wVMF
    emVMF = MixVMF(K=K, N=N, P=Y_vmf.shape[2], X=None, uniform_kappa=True)
    emVMF.V = pt.clone(em_WVMF.V)
    emVMF.kappa = pt.clone(em_WVMF.kappa)

    # Run 1 time M step for both model
    em_WVMF.initialize(Y, signal=signal)
    em_WVMF.Mstep(expand_mn(U, K))
    wVMF_kappa = em_WVMF.kappa
    wVMF_V = em_WVMF.V

    emVMF.initialize(Y_vmf)
    emVMF.Mstep(expand_mn(U_vmf, K))
    VMF_kappa = emVMF.kappa
    VMF_V = emVMF.V

    return wVMF_kappa, VMF_kappa, wVMF_V, VMF_V


def trim_locmin(T1, T2):
    D = [T1, T2]
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


if __name__ == '__main__':
    A, B ,C, D, E, F = [], [], [], [], [], []

    # a,b = _check_VMF_and_wVMF_equivalent(K=5, P=5000, N=20, num_sub=1, max_iter=100,
    #                                      kappa=30, plot=True)
    iter = 10
    dof = np.sqrt(iter)
    for i in range(iter):
        print(f'iter - {i}')
        a, b, c, d, e, f = _simulate_VMF_and_wVMF_from_GME(K=5, P=5000, N=20, num_sub=1, max_iter=500,
                                                           beta=2.0, sigma2=0.01, plot_ll=False,
                                                           plot_weight=False, plot=False)
        # a, b, = _simulate_VMF_and_wVMF_from_VMF(K=5, P=5000, N=20, num_sub=1, max_iter=100,
        #                                              kappa=30, plot=True)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)
        E.append(e)
        F.append(f)

    A = pt.stack(A).reshape(-1)
    B = pt.stack(B).reshape(-1)
    C = pt.stack(C).reshape(-1)
    D = pt.stack(D).reshape(-1)
    E = pt.stack(E).reshape(-1)
    F = pt.stack(F).reshape(-1)

    plt.bar(['wVMF \n length', 'wVMF \n length+density','wVMF \n true signal', 'VMF'],
            [A.mean(), B.mean(), C.mean(), D.mean()],
            yerr=[A.std()/dof, B.std()/dof, C.std()/dof, D.std()/dof],
            color=['red', 'green', 'blue', 'orange'])
    plt.axhline(y=F.mean(), color='k', linestyle=':')
    # plt.ylim(0.6, 0.7)
    plt.title('simulation on GME data, max_length=EXP E[X]=0.5')
    plt.show()

    res = trim_locmin(C, D)
    pass