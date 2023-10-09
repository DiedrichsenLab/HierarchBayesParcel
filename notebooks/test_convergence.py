#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test convergence

Created on 6/14/2023 at 1:39 PM
Author: dzhi
"""
import os
import sys

sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torchvision.transforms as transforms
import arrangements as ar
import emissions as em
import full_model as fm
import spatial as sp
import evaluation as ev
import pandas as pd
import seaborn as sb
import copy

from FusionModel.evaluate import calc_test_dcbc
from HierarchBayesParcel.depreciated.full_model_symmetric import FullModel

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel of the given size and standard deviation"""
    size = int(size) // 2
    coords = pt.meshgrid(*[pt.arange(-size, size + 1)] * N)
    distances_sq = sum([(x ** 2) for x in coords])
    g = torch.exp(-distances_sq / (2 * sigma ** 2))
    return g / g.sum()

def gaussian_smoothing_Nd(image):
    """Apply a 3x3 Gaussian kernel smoothing on an Nd image"""
    kernel = gaussian_kernel(3, sigma=1).unsqueeze(0).unsqueeze(0).to(image.device)
    smoothed = F.conv2d(image.unsqueeze(0), kernel.repeat(image.shape[1], 1, 1, 1), padding=1)
    return smoothed.squeeze(0)

def make_cmpRBM_data(width=10, K=5, N=20, num_subj=20, theta_mu=20,
                     theta_w=1.0, emission_model=None, do_plot=1):
    """Generates (and plots Markov random field data)
    Args:
        width (int, optional): [description]. Defaults to 10.
        K (int, optional): [description]. Defaults to 5.
        N (int, optional): [description]. Defaults to 200.
        theta_mu (int, optional): [description]. Defaults to 20.
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.5.
        do_plot (int): 1: Plot of the first 10 samples 2: + sample path
    """
    P = width * width
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width, height=width)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model
    arrangeT = ar.cmpRBM(K, grid.P, Wc=W, theta=theta_w)
    arrangeT.name = 'cmpRBM_true'
    arrangeT.bu = grid.random_smooth_pi(K=K, theta_mu=theta_mu,
                                        centroids=[0, width - 1, int(P / 2 + width / 2), P - width,
                                                   P - 1])

    MT = FullModel(arrangeT, emission_model)

    # grid.plot_maps(pt.softmax(arrangeT.bu,0),cmap='hot',vmax=1,grid=[1,5])

    # Step 3: Plot the prior of the true mode
    # plt.figure(figsize=(7,4))
    # grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    # cluster = np.argmax(arrangeT.logpi,axis=0)
    # grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model
    p = pt.ones(K)
    U = ar.sample_multinomial(pt.softmax(arrangeT.bu, 0), shape=(num_subj, K, grid.P))
    if do_plot > 1:
        plt.figure(figsize=(20, 10))

    for i in range(10):
        ph, H = arrangeT.sample_h(U)
        pu, U = arrangeT.sample_U(H)
        if do_plot > 1:
            u = ar.compress_mn(U)
            grid.plot_maps(u[8], cmap='tab10', vmax=K, grid=[2, 5], offset=i + 1)

    Utrue = ar.compress_mn(U)
    MT.arrange.gibbs_U = U
    # This is the training data
    Ytrain = MT.emission.sample(Utrue)
    Ytest = MT.emission.sample(Utrue)  # Testing data

    # Plot first 10 samples
    if do_plot > 0:
        plt.figure(figsize=(13, 5))
        grid.plot_maps(Utrue[0:10], cmap='tab10', vmax=K, grid=[2, 5])

    return Ytrain, Ytest, Utrue, MT, grid

def make_train_model(model_name='cmpRBM', K=3, P=5, num_subj=20, eneg_iter=10,
                     epos_iter=10, Wc=None, theta=None, fit_W=True, fit_bu=False, lr=1):
    if model_name.startswith('idenp'):
        # 1 - Independent spatial arrangement model
        M = ar.ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
        M.random_params()
        M.name = model_name
    elif model_name == 'cRBM_W':
        # Boltzmann with a arbitrary fully connected model - P hiden nodes
        n_hidden = P
        M = ar.cmpRBM(K, P, nh=n_hidden, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = f'cRBM_{n_hidden}'
        M.W = pt.randn(n_hidden, P) * 0.1
        M.alpha = lr
        M.fit_W = fit_W
        M.fit_bu = fit_bu
    elif model_name == 'cRBM_Wc':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        M = ar.cmpRBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc'
        M.fit_W = fit_W
        M.fit_bu = fit_bu
        M.alpha = lr
    elif model_name == 'cRBM_Wc_true':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        M = ar.cmpRBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc_true'
        M.fit_W = False
        M.fit_bu = False
        M.alpha = lr
    elif model_name == 'cRBM_Wc2':
        if Wc is None:
            raise ValueError('Wc must be provided to create wcmDBM arrangement model')

        M = ar.wcmDBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc2'
        M.fit_W = fit_W
        M.fit_bu = fit_bu
        M.alpha = lr
    else:
        raise ValueError('Unknown model name')

    return M

def train_sml(arM, emM, Ytrain, Ytest, part, crit='Ecos_err',
              n_epoch=20, batch_size=20, verbose=False):
    """Trains only arrangement model, given a fixed emission
    likelhoood.

    Args:
        arM (ArrangementMode):
        emM (EmissionModel )
        Y_train (tensor): Y_testing log likelihood (KxP)
        Y_test (tensor): Y_training log likelihood test (KxP)
        part (tensor): 1xP partition number for completion test
        crit (str): _description_. Defaults to 'logpY'.
        n_epoch (int): _description_. Defaults to 20.
        batch_size (int): _description_. Defaults to 20.
        verbose (bool): _description_. Defaults to False.

    Returns:
        model: Fitted model
        T: Pandas data frame with epoch level performance metrics
        thetaH: History of fitted thetas
    """
    emlog_train = emM.Estep(Ytrain)
    emlog_test = emM.Estep(Ytest)
    num_subj = emlog_train.shape[0]
    Utrain = pt.softmax(emlog_train, dim=1)

    crit_types = ['train', 'marg', 'test']  # different evaluation types
    CR = np.zeros((len(crit_types), n_epoch))
    theta_list = pt.zeros((arM.nparams, n_epoch))
    marginals = pt.zeros((n_epoch, arM.K, arM.P))
    CE = pt.zeros((n_epoch,))
    # Intialize negative sampling
    for epoch in range(n_epoch):
        # Get test error
        EU, _ = arM.Estep(emlog_train, gather_ss=False)
        for i, ct in enumerate(crit_types):
            # Training emission logliklihood:
            if ct == 'train':
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytrain, EU, crit=crit)
            elif ct == 'marg':
                pi = arM.marginal_prob()
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytest, pi, crit=crit)
            elif ct == 'test':
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytest, EU, crit=crit)
            elif ct == 'compl':
                CR[i, epoch] = ev.evaluate_completion_arr(arM, emM, Ytest, part, crit=crit)
        if (verbose):
            print(f'epoch {epoch:2d} Test: {crit[2, epoch]:.4f}')

        theta_list[:, epoch] = arM.get_params()
        marginals[epoch, :, :] = arM.marginal_prob()
        # Update the model in batches
        for b in range(0, num_subj - batch_size + 1, batch_size):
            ind = range(b, b + batch_size)
            arM.Estep(emlog_train[ind, :, :])
            if hasattr(arM, 'Eneg'):
                arM.Eneg(use_chains=ind,
                         emission_model=emM)
            arM.Mstep()

        # Record the cross entropy parameters
        if arM.name.startswith('idenp'):
            CE[epoch] = 0
        else:
            CE[epoch] = ev.cross_entropy(pt.softmax(emlog_train, dim=1),
                                         arM.eneg_U)
            # CE[epoch] = pt.abs(pt.softmax(emlog_train, dim=1) - arM.eneg_U).sum()

    # Make a data frame for the results
    T = pd.DataFrame()
    for i, ct in enumerate(crit_types):
        T1 = pd.DataFrame({'model': [arM.name] * n_epoch,
                           'type': [ct] * n_epoch,
                           'iter': np.arange(n_epoch),
                           'crit': CR[i]})
        T = pd.concat([T, T1], ignore_index=True)

    return arM, T, theta_list, CE, marginals


def eval_dcbc(models, emM, Ytrain, Ytest, grid, Utrue_group, Utrue_indiv, SD,
              max_dist=10, bin_width=1):
    D = pd.DataFrame()
    group_par, indiv_par = [], []
    nsubj = Utrue_indiv.shape[0]

    for m in models:
        smooth = 0
        if isinstance(m, str) and m.startswith('data'):
            if m == 'data':
                ind = 0
            else:
                ind = int(m.split('_')[1])
            emloglik_train = emM.Estep(Ytrain[ind])
            this_Ugroup = pt.softmax(emloglik_train.sum(dim=0), dim=0).argmax(dim=0)
            this_Uindiv = pt.softmax(emloglik_train, 1).argmax(dim=1)
            name = m
            smooth = SD[ind]
            model_type = 'data'
        elif m == 'Utrue':
            this_Ugroup = Utrue_group
            this_Uindiv = Utrue_indiv
            name = m
            model_type = 'true'
        else:
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                this_Ugroup = m.marginal_prob().argmax(dim=0)
                this_Uindiv = m.estep_Uhat.argmax(dim=1)
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                this_Ugroup = pt.softmax(m.bu, dim=0).argmax(dim=0)
                this_Uindiv = m.epos_Uhat.argmax(dim=1)
                model_type = 'cRBM'
            elif m.name.startswith('cmpRBM'):
                emloglik_train = emM.Estep(Ytrain[ind])
                EU,_ = m.Estep(emloglik_train, gather_ss=False)
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name

        dcbc_group = calc_test_dcbc(this_Ugroup, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)
        dcbc_indiv = calc_test_dcbc(this_Uindiv, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)

        group_par.append(this_Ugroup)
        indiv_par.append(this_Uindiv)

        dict = {'model': [name] * nsubj,
                'type': ['test'] * nsubj,
                'smooth': [smooth] * nsubj,
                'arrangement': [model_type] * nsubj,
                'dcbc_group': dcbc_group.cpu(),
                'dcbc_indiv': dcbc_indiv.cpu()}
        D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)

    return D, group_par, indiv_par


def eval_arrange(models, emM, Ytrain, Ytest, SD, Utrue):
    D = pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue, emM.K)
    nsubj = Utrue.shape[0]

    for m in models:
        smooth = 0
        if isinstance(m, str) and m.startswith('data'):
            if m == 'data':
                ind = 0
            else:
                ind = int(m.split('_')[1])
            emloglik_train = emM.Estep(Ytrain[ind])
            EU = pt.softmax(emloglik_train, 1)
            smooth = SD[ind]
            name = m
            model_type = 'data'
        elif m == 'Utrue':
            EU = Utrue_mn
            name = m
            model_type = 'true'
        else:
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                EU = m.estep_Uhat
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                EU = m.epos_Uhat
                model_type = 'cRBM'
            elif m.name.startswith('cmpRBM'):
                emloglik_train = emM.Estep(Ytrain[ind])
                EU,_ = m.Estep(emloglik_train, gather_ss=False)
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name
        uerr_test1 = pt.mean(pt.abs(Utrue_mn - EU), dim=(1, 2)).cpu()
        cos_err = ev.coserr(Ytest, emM.V, EU, adjusted=False,
                            soft_assign=False).cpu()
        Ecos_err = ev.coserr(Ytest, emM.V, EU, adjusted=False,
                             soft_assign=True).cpu()

        dict = {'model': [name] * nsubj,
                'type': ['test'] * nsubj,
                'smooth': [smooth] * nsubj,
                'arrangement': [model_type] * nsubj,
                'uerr': uerr_test1,
                'cos_err': cos_err,
                'Ecos_err': Ecos_err}
        D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)
    return D


def eval_arrange_compl(models, emM, Y, part, Utrue):
    D = pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue, models[0].K)
    for m in models:
        cos_err_compl = ev.evaluate_completion_arr(m, emM, Y, part, crit='cos_err')
        Ecos_err_compl = ev.evaluate_completion_arr(m, emM, Y, part, crit='Ecos_err')
        uerr_compl = ev.evaluate_completion_arr(m, emM, Y, part,
                                                crit='u_abserr', Utrue=Utrue_mn)
        dict = {'model': [m.name],
                'type': ['compl'],
                'uerr': uerr_compl,
                'cos_err': cos_err_compl,
                'Ecos_err': Ecos_err_compl}
        D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)
    # get the baseline for Utrue
    cos_err = ev.coserr(Y, emM.V, Utrue_mn, adjusted=False,
                        soft_assign=False).mean(dim=0).item()
    Ecos_err = ev.coserr(Y, emM.V, Utrue_mn, adjusted=False,
                         soft_assign=True).mean(dim=0).item()
    dict = {'model': ['Utrue'],
            'type': ['compl'],
            'uerr': 0,
            'cos_err': cos_err,
            'Ecos_err': Ecos_err}
    D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)
    return D


def plot_Uhat_maps(models, emloglik, grid):
    plt.figure(figsize=(10, 7))
    n_models = len(models)
    K = emloglik.shape[1]
    for i, m in enumerate(models):
        if m is None:
            Uh = pt.softmax(emloglik, dim=1)
        else:
            Uh, _ = m.Estep(emloglik)
        grid.plot_maps(Uh[0], cmap='jet', vmax=1, grid=(n_models, K), offset=i * K + 1)


def plot_P_maps(pmaps, grid):
    n_models = len(pmaps)
    K = pmaps[0].shape[0]

    plt.figure(figsize=(K * 3, n_models * 3))
    for i, m in enumerate(pmaps):
        grid.plot_maps(m, cmap='jet', vmax=1, grid=(n_models, K), offset=i * K + 1)

    plt.show()


def plot_U_maps(pmaps, grid, title):
    n_models = len(pmaps)

    plt.figure(figsize=(n_models * 3, 4))
    for i, m in enumerate(pmaps):
        grid.plot_maps(m, cmap='tab20', vmax=19, grid=(1, n_models), offset=i + 1)
        plt.title(title[i])

    plt.show()


def plot_individual_Uhat(models, Utrue, emloglik, grid, style='prob'):
    # Get the expectation
    n_models = len(models) + 2
    K = emloglik.shape[1]
    P = emloglik.shape[2]

    Uh = []
    height = 2 if style == 'mixed' else 1
    plt.figure(figsize=(n_models * 3, height * 4))

    # Uh order: data -> models -> Utrue
    Uh.append(pt.softmax(emloglik[0:1], dim=1))
    for i, m in enumerate(models):
        A, _ = m.Estep(emloglik[0:1])
        Uh.append(A)
    Uh.append(ar.expand_mn(Utrue[0:1], K))

    if style == 'prob':
        for i, uh in enumerate(Uh):
            grid.plot_maps(uh[0], cmap='jet', vmax=1,
                           grid=(n_models, K),
                           offset=K * i + 1)
    elif style == 'argmax':
        ArgM = pt.zeros(n_models, P)
        for i, uh in enumerate(Uh):
            ArgM[i, :] = pt.argmax(uh[0], dim=0)
        grid.plot_maps(ArgM, cmap='tab10', vmax=K,
                       grid=(1, n_models))
    elif style == 'mixed':
        ArgM = pt.zeros(n_models, P)
        Prob = pt.zeros(n_models, P)

        for i, uh in enumerate(Uh):
            ArgM[i, :] = pt.argmax(uh[0], dim=0)
            Prob[i, :] = uh[0][2, :]
        grid.plot_maps(ArgM, cmap='tab10', vmax=K,
                       grid=(2, n_models))
        grid.plot_maps(Prob, cmap='jet', vmax=1,
                       grid=(2, n_models),
                       offset=n_models + 1)

    plt.savefig('Uhat_indiv.pdf', format='pdf')
    plt.show()

def plot_rbm(D, criteria=['uerr', 'Ecos_err', 'dcbc_indiv'],
             types=['test', 'compl'], save=True):
    # Get the final error and the true pott models
    ncrit = len(criteria)
    ntypes = len(types)
    plt.figure(figsize=(5 * ncrit, 5 * ntypes))
    for j in range(ntypes):
        for i in range(ncrit):
            plt.subplot(ntypes, ncrit, i + j * ncrit + 1)
            # sb.barplot(data=D[D.type==types[j]], x='model', y=criteria[i])

            df = DD.loc[(DD.model == 'cRBM_Wc')]
            sb.barplot(data=df, x='sim', y=criteria[i],
                       palette=sb.color_palette("tab10"), errorbar="se")

            rbm_true = DD.loc[(DD.model == 'cmpRBM_true') & (DD.sim == 0)]
            plt.axhline(rbm_true[criteria[i]].mean().item(), color='k', ls=':')

            plt.title(f'{criteria[i]}{types[j]}')
            plt.legend()
            # plt.xticks(rotation=45)

            # Ylim
            if criteria[i] == 'uerr':
                plt.ylim([0, 0.007])
            elif criteria[i] == 'cos_err':
                plt.ylim([0.42, 0.46])
            elif criteria[i] == 'Ecos_err':
                plt.ylim([0.425, 0.435])
            elif criteria[i] == 'dcbc_indiv':
                plt.ylim([0.34, 0.38])

    plt.suptitle(f'final errors')
    plt.tight_layout()

    if save:
        plt.savefig('test_errs_to_true.pdf', format='pdf')
    plt.show()

def plot_evaluation(D, criteria=['uerr', 'cos_err', 'Ecos_err', 'dcbc_group', 'dcbc_indiv'],
                    types=['test', 'compl']):
    # Get the final error and the true pott models
    ncrit = len(criteria)
    ntypes = len(types)
    plt.figure(figsize=(5 * ncrit, 5 * ntypes))
    for j in range(ntypes):
        for i in range(ncrit):
            plt.subplot(ntypes, ncrit, i + j * ncrit + 1)
            # sb.barplot(data=D[D.type==types[j]], x='model', y=criteria[i])

            df = D[(D.type == types[j]) & (D.arrangement == 'idenp')]
            sb.lineplot(data=df, x='smooth', y=criteria[i],
                        err_style="bars", markers=False)

            emlog = D[(D.type == types[j]) & (D.arrangement == 'data')]
            # plt.axhline(emlog[criteria[i]].mean().item(), color='k', ls=':',
            #             label='data')
            sb.lineplot(data=emlog, x='smooth', y=criteria[i],
                        err_style="bars", markers=False)

            rbm_wc = D[(D.type == types[j]) & (D.model == 'cRBM_Wc')]
            plt.axhline(rbm_wc[criteria[i]].mean().item(), color='r', ls=':',
                        label='cRBM_Wc')

            rbm_wc = D[(D.type == types[j]) & (D.model == 'Utrue')]
            plt.axhline(rbm_wc[criteria[i]].mean().item(), color='b', ls=':',
                        label='Utrue')

            plt.title(f'{criteria[i]}{types[j]}')
            plt.legend()
            # plt.xticks(rotation=45)

            # Ylim
            if criteria[i] == 'uerr':
                plt.ylim([-0.005, 0.03])
            elif criteria[i] == 'cos_err':
                plt.ylim([0.42, 0.46])
            elif criteria[i] == 'Ecos_err':
                plt.ylim([0.42, 0.46])
            elif criteria[i] == 'dcbc_indiv':
                plt.ylim([0.32, 0.42])

    plt.suptitle(f'final errors')
    plt.tight_layout()

    plt.savefig('test_errs.pdf', format='pdf')
    plt.show()


def plot_evaluation2():
    # Get the final error and the true pott models
    plt.figure(figsize=(3, 4))
    D = pd.read_csv('deepMRF.csv')
    T = D[(D.type == 'test') & (D.model != 'true') & (D.model != 'Utrue')]
    noisefloor = D[(D.type == 'test') & (D.model == 'Utrue')].Ecos_err.mean()
    sb.barplot(data=T, x='model', y='Ecos_err')
    plt.ylim([0.5, 0.8])
    plt.axhline(noisefloor)
    pass

def simulation_2(K=5, width=50, num_subj=30, batch_size=30, n_epoch=200, theta=1.2,
                 theta_mu=180, emission='gmm', epos_iter=20, eneg_iter=20, num_sim=10):
    P = width * width
    if emission == 'gmm':  # MixGaussian
        sigma2 = 0.2
        N = 10
        emissionM = em.MixGaussian(K, N, P)
        emissionM.sigma2 = pt.tensor(sigma2)
    elif emission == 'mn':  # Multinomial
        w = 2.0
        emissionM = em.MultiNomial(K=K, P=P)
        emissionM.w = pt.tensor(w)

    # Record the results
    TT = pd.DataFrame()
    DD = pd.DataFrame()
    HH = pt.zeros((num_sim, n_epoch))
    BU_all = pt.zeros((num_sim, n_epoch))
    BU_all_1 = pt.zeros((num_sim, n_epoch))
    BU_all_2 = pt.zeros((num_sim, n_epoch))
    BU_all_3 = pt.zeros((num_sim, n_epoch))
    CE_rbm1 = pt.zeros((num_sim, n_epoch))
    CE_rbm2 = pt.zeros((num_sim, n_epoch))
    GM, IM, BUs = [], [], []

    # REcorded bias parameter
    # SD = np.concatenate((np.linspace(0.1,1,10), np.linspace(1.5,3,4)))
    # SD = np.round(SD, decimals=2)
    SD = [0.5]
    Rec = pt.zeros((len(SD) + 3, num_sim, K, P))  # unsmooth + 2 rbms + 1 emloglik

    # Generate partitions for region-completion testing
    num_part = 4
    p = pt.ones(num_part) / num_part
    part = pt.multinomial(p, P, replacement=True)

    Ytrain, Ytest, Utrue, Mtrue, grid = make_cmpRBM_data(width, K, N=N,
                                                         num_subj=num_subj, theta_mu=theta_mu,
                                                         theta_w=theta,
                                                         emission_model=emissionM,
                                                         do_plot=0)

    # Get the smoothed training data
    Ytrain_smooth = []
    for smooth in SD:
        blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=smooth)
        Ys = blur_transform(Ytrain.view(Ytrain.shape[0], -1, width, width))
        Ys = Ys.view(Ytrain.shape[0], Ytrain.shape[1], -1)
        Ytrain_smooth.append(Ys)

    # Get the true arrangement model and its loglik
    emloglik_train = Mtrue.emission.Estep(Ytrain)
    emloglik_test = Mtrue.emission.Estep(Ytest)
    P = Mtrue.emission.P

    for s in range(num_sim):
        rbm = Mtrue.arrange

        # Make list of fitting models
        Models, fitted_M = [], []
        fitting_names = ['idenp_0'] + [f'idenp_{s}' for s in SD] + ['cRBM_Wc']
        Y_fit = [Ytrain] + Ytrain_smooth + [Ytrain]
        for nam in fitting_names:
            Models.append(make_train_model(model_name=nam, K=K, P=P,
                                           num_subj=num_subj, eneg_iter=eneg_iter,
                                           epos_iter=epos_iter, Wc=rbm.Wc.squeeze(2),
                                           theta=None,
                                           fit_W=True, fit_bu=True, lr=0.5))

        # Train different arrangement model
        TH, CE, MG = [], [], []
        T = pd.DataFrame()
        for i, m in enumerate(Models):
            # Give the model the true bias/W for rbms
            if m.name.startswith('cRBM') or m.name.startswith('wcmDBM'):
                # m.W = rbm.W.detach().clone()
                # m.bu = rbm.bu.detach().clone()
                pass

            m, T1, theta_hist, ce, marginals = train_sml(m, Mtrue.emission, Y_fit[i],
                                                         Ytest, part, batch_size=batch_size,
                                                         n_epoch=n_epoch)
            fitted_M.append(m)
            TH.append(theta_hist)
            CE.append(ce)
            MG.append(marginals)
            T = pd.concat([T, T1], ignore_index=True)

        # Evaluate overall
        # 1. u_absolute error, cos_err, and expected cos_err
        D = eval_arrange(['data'] + fitted_M + [rbm] + ['Utrue'],
                         Mtrue.emission, [Ytrain], Ytest, np.insert(SD, 0, 0),
                         Utrue=Utrue)

        # 2. DCBC
        binWidth = 5
        max_dist = binWidth * pt.ceil(grid.Dist.max() / binWidth)
        D1, group_map, indiv_map = eval_dcbc(['data'] + fitted_M + [rbm] + ['Utrue'],
                                             Mtrue.emission,
                                             [Ytrain], Ytest, grid,
                                             pt.softmax(rbm.bu, dim=0).argmax(dim=0),
                                             Utrue, np.insert(SD, 0, 0),
                                             max_dist=max_dist, bin_width=binWidth)

        GM.append(group_map)
        IM.append(indiv_map)
        # 3. Region completion test
        # D1 = eval_arrange_compl(fitted_M, Mtrue.emission, Ytest,
        #                         part=part, Utrue=Utrue)
        res = pd.concat([D, D1.iloc[:, 4:]], axis=1)
        res['sim'] = s
        res['fitted_theta'] = TH[-1][fitted_M[-1].get_param_indices('theta'), -1].item()
        DD = pd.concat([DD, res], ignore_index=True)
        TT = pd.concat([TT, T], ignore_index=True)

        # Record the theta for rbm_Wc model only
        HH[s, :] = TH[-1][fitted_M[-1].get_param_indices('theta'), :]
        # Record the distance measure of |bias - true bu|
        fit_bu = TH[-1][fitted_M[-1].get_param_indices('bu'), :]
        fit_bu = fit_bu.T.view(-1, rbm.bu.shape[0], rbm.bu.shape[1])
        for counter in range(fit_bu.shape[0]):
            # 2. marginals L2-norm
            BU_all_1[s, counter] = pt.norm(rbm.marginal_prob() - MG[-1][counter], p=2)
            BU_all_2[s, counter] = pt.norm(rbm.marginal_prob() - MG[0][counter], p=2)
            BU_all_3[s, counter] = pt.norm(MG[0][counter] - MG[-1][counter], p=2)

            # 3. BU L2-norm
            this_fb = fit_bu - fit_bu.mean(dim=1, keepdim=True)
            this_bu = rbm.bu - rbm.bu.mean(dim=0, keepdim=True)
            BU_all[s, counter] = pt.norm(this_fb[counter, :, :] - this_bu, p=2)

        # Record cross entropy for rbms
        CE_rbm1[s, :] = CE[-2]
        CE_rbm2[s, :] = CE[-1]
        BUs.append(fit_bu)

        # record the different fitting runs into structure
        Rec[0, s, :, :] = pt.softmax(emloglik_train, 1).mean(dim=0)  # first is data
        for j, fm in enumerate(fitted_M):
            if fm.name.startswith('idenp'):
                Rec[j + 1, s, :, :] = pt.softmax(fm.logpi, 0)
            elif fm.name.startswith('cRBM'):
                Rec[j + 1, s, :, :] = pt.softmax(fm.bu, 0)
            else:
                raise ValueError('Unknown model name')
        # Rec[-1,s,:,:] = ar.expand_mn(Utrue, K).mean(dim=0)

    # Plot learning curves by epoch
    fig = plt.figure(figsize=(12, 4))
    # plt.subplot(4, 2, 1)
    # plt.plot(CE_rbm1.T.cpu().numpy(), linestyle='-', label='rbm_Wc')
    # plt.plot(CE_rbm2.T.cpu().numpy(), linestyle=':', label='rbm_W')
    # plt.ylabel('Cross Entropy')
    # plt.legend(['rbm_Wc (solid)','rbm_W (dotted)'])
    # plt.subplot(4, 2, 2)
    # sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='test')], y='crit',
    #             x='iter', hue='model')
    # plt.ylabel('Test coserr')
    # plt.subplot(4, 2, 3)
    # sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='compl')]
    #         ,y='crit',x='iter',hue='model')
    # plt.ylabel('Compl coserr')
    plt.subplot(1, 3, 1)
    plt.plot(HH.T.cpu().numpy())
    plt.axhline(y=HH[:, -1].cpu().numpy().mean(), color='r', linestyle='-')
    plt.axhline(y=theta, color='k', linestyle='-')
    plt.ylabel('Theta')
    plt.subplot(1, 3, 2)
    plt.plot(BU_all.T.cpu().numpy())
    plt.ylabel('L2-norm - |bu - true bu|')
    plt.subplot(1, 3, 3)
    plt.plot(BU_all_1.T.cpu().numpy())
    plt.axhline(y=BU_all_2[:, -1].cpu().numpy().mean(), color='r', linestyle='-')
    plt.axhline(y=0, color='k', linestyle=':')
    plt.ylabel('L2-norm - marginals')

    plt.tight_layout()
    plt.savefig("learning_curves_2_new.pdf", format='pdf')
    plt.show()

    pt.save(HH.cpu(), 'thetas.pt')
    pt.save(BU_all.cpu(), 'L2_bu.pt')
    pt.save(BU_all_1.cpu(), 'L2_marginals.pt')
    pt.save(BU_all_2.cpu(), 'L2_marginals_indp.pt')
    pt.save(MG[0][-1].cpu(), 'learned_indp_marginals.pt')
    # records = [RecEmLog, RecLp1, RecLp2, RecLp3, RecLp4, RecLp5, RecBu1, RecBu2]
    return grid, DD, Rec, rbm, fitted_M, Utrue, emloglik_train, GM, IM


if __name__ == '__main__':
    # emissionM = em.MixGaussian(5, 10, 2500)
    # emissionM.sigma2 = pt.tensor(0.2)
    # TM = [60, 240, 680]
    # TH = [0.2, 0.5, 1, 1.5, 5]
    # samples = []
    # for i, theta_mu in enumerate(TM):
    #     for j, theta in enumerate(TH):
    #         Ytrain, Ytest, Utrue, Mtrue, grid = make_cmpRBM_data(50, 5, N=10,
    #                                                              num_subj=10, theta_mu=theta_mu,
    #                                                              theta_w=theta,
    #                                                              emission_model=emissionM,
    #                                                              do_plot=0)
    #         samples.append(Utrue[0])
    #
    # plt.figure(figsize=(15, 5))
    # grid.plot_maps(pt.stack(samples), cmap='tab10', vmax=5, grid=[3, 5])
    # plt.savefig('true_maps.pdf', format='pdf')
    # plt.show()

    # grid, DD, records, rbm, Models, Utrue, emloglik_train, GM, IM = simulation_2(theta_mu=240,
    #                                                                              num_sim=10)

    # Get the final error and the true pott models
    # DD.to_csv(f'eval_cpmRBM_fit.tsv', index=False, sep='\t')
    DD = pd.read_csv('eval_cpmRBM_fit.tsv', delimiter='\t')
    plot_rbm(DD,types=['test'], save=True)
    plot_evaluation(DD, types=['test'])

    # OPtional: Plot the last maps of prior estimates
    # plot_Uhat_maps([None,indepAr,rbm3,Mtrue.arrange],emloglik_test[0:1],grid)

    # Optimal: plot the prob maps by K
    plot_P_maps(pt.cat((records.mean(dim=1), pt.softmax(rbm.bu, 0).unsqueeze(0)), dim=0),
                grid)

    # plot the group reconstructed U maps
    # plot_U_maps(pt.stack(GM[0]), grid, title=['data'] + [m.name for m in Models] + ['true'])

    plot_individual_Uhat(Models, Utrue[0:1], emloglik_train[0:1],
                         grid, style='mixed')
    # plot_individual_Uhat(Models,Utrue[0:1],emloglik_train[0:1],
    #                grid,style='argmax')
    pass

    # plot_evaluation2()
    # test_cmpRBM_Estep()
    # test_sample_multinomial()
    # train_RBM()
