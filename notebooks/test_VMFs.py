#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on 2/22/2023 at 12:59 PM
Author: dzhi
"""
# global package import
from copy import copy, deepcopy
import pandas as pd
import seaborn as sb
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# for testing and evaluating models
from generativeMRF.full_model import FullModel, FullMultiModel
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.spatial as sp
import generativeMRF.evaluation as ev
from sklearn.metrics.pairwise import cosine_similarity
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.evaluate import calc_test_dcbc

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

def u_err(U, Uhat):
    """Absolute error on U
    Args:
        U (tensor): Real U's
        uhat (tensor): Estimated U's from arrangement model
    """
    err = pt.abs(U - Uhat).to(pt.bool)
    return err.sum(dim=1) / err.size(dim=1)

def _compute_adjacency(map, k):
    """Compute the adjacency matrix and return k clusters label that are neighbours
    Args:
        map: the original cluster assignment map
        k: the number of neighbouring clusters
    Returns:
        G: the adjacency matrix
        base_label: the label of the seed parcel
        neighbours: the labels of k-1 clusters that are neighbouring
    """
    num_parcel = map.unique().shape[0]

    # Handling corner case: the parcel labels are not continous
    # i.e [0,1,2,4,5] which causes adjacency mapping array index overflow
    A, B = pt.unique(map.unique(), return_inverse=True)
    new_map = (map.view(-1, 1) == A).int().argmax(dim=1).reshape(map.shape)

    # Making adjacency matrix
    G = pt.zeros([num_parcel] * 2)
    # left-right pairs
    G[new_map[:, :-1], new_map[:, 1:]] = 1
    # right-left pairs
    G[new_map[:, 1:], new_map[:, :-1]] = 1
    # top-bottom pairs
    G[new_map[:-1, :], new_map[1:, :]] = 1
    # bottom-top pairs
    G[new_map[1:, :], new_map[:-1, :]] = 1
    # G.fill_diagonal_(0)

    visited = [False] * num_parcel
    start = pt.randint(num_parcel, ())
    q = [start]
    visited[start] = True
    neighbours = []
    # Standard BFS search
    while q:
        vis = q[0]
        neighbours.append(vis)
        q.pop(0)
        for i in range(num_parcel):
            if (G[vis][i] == 1) and (not visited[i]):
                q.append(pt.tensor(i))
                visited[i] = True

    neighbours = pt.stack(neighbours)[:k]
    # Return A[neighbours] to map back the original labels
    return G, start, A[neighbours]

def make_true_model_GME(grid, K=5, P=100, nsubj_list=[10,10],
                        M=[5,5], # Number of conditions per data set
                        theta_mu=150, theta_w=20, inits=None,
                        sigma2=[1.0,1.0], high_norm=0.9, low_norm=0.1,
                        same_subj=False, relevant=True):
    """Making a full model contains an arrangement model and one or more
       emission models with desired settings
    Args:
        K: the number of clusers
        P: the number of voxels
        nsubj_list: the number of subjects in emission models
        M: the number of conditions per emission model
        num_part: the number of partitions per emission model
        common_kappa: if True, using common kappa. Otherwise, separate kappas
        same_subj: if True, the same set of subjects across emission models
    Returns:
        M: the full model object
    """
    A = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    A.random_smooth_pi(grid.Dist, theta_mu=theta_mu, centroids=inits)
    A.theta_w = pt.tensor(theta_w)
    label_map = pt.argmax(A.logpi, dim=0).reshape(grid.dim)

    emissions, idx_all = [], []
    K_group = pt.arange(int(K/2)+1, K).chunk(len(M))
    for i,m in enumerate(M):
        # Step 2: up the emission model and sample from it with a specific signal
        emissionT = em.MixGaussianExp(K=K, N=m, P=P, num_signal_bins=100, std_V=True)
        emissionT.sigma2 = pt.tensor(sigma2[i])
        # Initialize all V to be the high norm
        emissionT.V = emissionT.V * high_norm

        # Making ambiguous boundaries between random number parcelsby set low signal
        # for k-neighbouring parcels. The parcels have same V magnitude in this emission
        num_badpar = K_group[i][int(pt.randint(K_group[i].numel(), ()))]
        if relevant:
            _, _, idx = _compute_adjacency(label_map, 0)
        else:
            _, _, idx = _compute_adjacency(label_map, num_badpar)
        ######## Uncomment this for irrelevant two sessions ########
            if i==0:
                idx_1 = idx
            elif i==1:
                idx = pt.tensor([j for j in label_map.unique() if j not in idx_1])

        print(f'Dataset {i+1} bad parcels are: {idx}')
        # Making the bad parcels V to have the low norm (signal)
        emissionT.V[:,idx] = emissionT.V[:,idx] * (low_norm/high_norm)
        emissionT.num_subj=nsubj_list[i]
        emissions.append(emissionT)
        idx_all.append(idx)

    T = FullMultiModel(A,emissions)
    if same_subj: # All emission models have the same set of subjects
        # This is used to simulate fusion across sessions
        assert np.all(nsubj_list == nsubj_list[0]), \
            "The number of subjects should be same across all emission models"
        sub_list = [np.arange(x) for x in nsubj_list]
        T.initialize(subj_ind=sub_list)
    else: # Emission models to have separate set subjects
        T.initialize()

    # Sampling individual Us and data, data_test
    Y, Y_test, signal = [], [], []
    U, _ = T.sample()
    U_test, _ = T.sample()
    for m, Us in enumerate(T.distribute_evidence(U)):
        Y.append(T.emissions[m].sample(Us, signal=pt.ones(Us.shape)))

    # Build a separate emission model contains all parcel infomation
    # for generating test data
    em_test = em.MixGaussianExp(K=K, N=sum(M)*2, P=P, num_signal_bins=100, std_V=True)
    em_test.sigma2 = pt.tensor(0.1)
    em_test.V = em_test.V * high_norm
    if same_subj:
        Uind = [U]
    else:
        Uind = T.distribute_evidence(U_test)

    # Get the true signals and Y_test
    for i, Us in enumerate(Uind):
        Y_test.append(em_test.sample(Us, signal=pt.ones(Us.shape)))
        signal.append(pt.where(pt.isin(Us, idx_all[i]), low_norm, high_norm))

    return T, Y, Y_test, U, U_test, signal, idx_all

def make_full_model(K=5, P=100, nsubj_list=[10,10],
                    M=[5,5], # Number of conditions per data set
                    num_part=3, common_kappa=False,
                    same_subj=False, model_type='VMF'):
    """Making a full fitting model contains an arrangement model and
       one or more emission models with desired settings
    Args:
        K: the number of clusers
        P: the number of voxels
        nsubj_list: the number of subjects in emission models
        M: the number of conditions per emission model
        num_part: the number of partitions per emission model
        common_kappa: if True, using common kappa. Otherwise, separate kappas
        same_subj: if True, the same set of subjects across emission models
    Returns:
        M: the full model object
    """
    A = ar.ArrangeIndependent(K,P,spatial_specific=True,remove_redundancy=False)
    A.random_params()
    emissions =[]
    for i,m in enumerate(M):
        X = np.kron(np.ones((num_part,1)),np.eye(m))
        part_vec = np.kron(np.arange(num_part),np.ones((m,)))

        if model_type == 'VMF':
            emission = em.MixVMF(K=K, X=X, P=P, part_vec=part_vec,
                                 uniform_kappa=common_kappa)
        elif model_type == 'wVMF':
            emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                                  uniform_kappa=common_kappa)
        elif model_type == 'wVMF_ones':
            emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                                  uniform_kappa=common_kappa, weighting='ones')
        elif (model_type == 'wVMF_t2') or (model_type == 'wVMF_t2P'):
            emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                                  uniform_kappa=common_kappa)
        elif model_type == 'wVMF_l2':
            emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                                  uniform_kappa=common_kappa, weighting='lsquare_sum2PJ')
        else:
            raise (NameError('Unknown model type'))

        emission.num_subj=nsubj_list[i]
        emissions.append(emission)

    M = FullMultiModel(A,emissions)
    if same_subj: # All emission models have the same set of subjects
        # This is used to simulate fusion across sessions
        assert np.all(nsubj_list == nsubj_list[0]), \
            "The number of subjects should be same across all emission models"
        sub_list = [np.arange(x) for x in nsubj_list]
        M.initialize(subj_ind=sub_list)
    else: # Emission models to have separate set subjects
        M.initialize()

    return M

def plot_uerr(D, names=['dataset 1', 'dataset 2', 'fusion'],
              sigma2=0.1, plot=True, save=False):
    """The helper function to plot the U prediction error
    Args:
        D (list): data structure of U_pred. e.g [[uerr1, uerr2, uerr3],...
                                                [uerr1, uerr2, uerr3]]
        names (list): list of names on x-axis
        plot: if True, plot the figure
        save: if True, save the plot as .eps file
    Returns:
        Figure plot
    """
    fig, axs = plt.subplots(1, len(D), figsize=(6*len(D), 6), sharey=True)
    sub_titles = ['VMF, commonKappa=True',
                  'VMF, commonKappa=False',
                  'wVMF, commonKappa=True',
                  'wVMF, commonKappa=False']
    for i, data in enumerate(D):
        A, B, C = data[0].cpu().numpy(), data[1].cpu().numpy(), data[2].cpu().numpy()
        axs[i].bar(names,
                   [A.mean(), B.mean(), C.mean()],
                   yerr=[A.std() / np.sqrt(len(A)),
                         B.std() / np.sqrt(len(B)),
                         C.std() / np.sqrt(len(C))],
                   color=['red', 'green', 'blue'],
                   capsize=10)
        min_err = min([A.mean(), B.mean(), C.mean()])
        axs[i].axhline(y=min_err, color='k', linestyle=':')
        axs[i].set_ylabel(f'U reconstruction error')
        axs[i].set_title(f'{sub_titles[i]}')
        # plt.ylim(0.6, 0.7)

    fig.suptitle(f'Simulation common/separate kappas and VMF/wVMF combination, sigma2={sigma2}')
    plt.tight_layout()

    if save:
        plt.savefig('Uerr.pdf', format='pdf')
    if plot:
        plt.show()
        plt.clf()
    else:
        pass

def plot_result(grid, MM, ylabels=["VMF, common",
                                   "VMF, separate",
                                   "wVMF, common",
                                   "wVMF, separate"],
                names = ["True", "Dataset 1", "Dataset 2", "Dataset 1+2"],
                save=False):
    """The helper function to plot the fitted group prior
    Args:
        grid (object): the markov random field grid obj
        MM (list): the list of propulation map
        ylabels: the labels of plot
        names: the labels on the columns
        save: if True, save the plot
    Returns:
        Figure plot
    """
    # Plotting results
    row = len(MM)
    col = MM[0].shape[0]

    for i, m in enumerate(MM):
        for j in range(len(m)):
            plt.subplot(row, col, i*col+j+1)
            parcel = pt.argmax(m[j, :, :], dim=0)
            grid.plot_maps(parcel)
            if i == 0:
                plt.title(names[j])
            if j == 0:
                plt.ylabel(ylabels[i])

    if save:
        plt.savefig('group_results.pdf', format='pdf')
    plt.show()
    plt.clf()

def _plot_diffK(D, hue="dataset", style=None, title=None, save=False):
    # D = D.loc[D.common_kappa == True]
    plt.figure(figsize=(8, 5))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        if style is not None:
            sb.lineplot(data=D, x="K_fit", y=c, hue=hue, hue_order=['D1', 'D2', 'D_fusion'],
                        style=style,
                        style_order=D[style].unique(), markers=True)
        else:
            sb.lineplot(data=D, x="K_fit", y=c, hue=hue, hue_order=['D1', 'D2', 'D_fusion'],
                        markers=True)

    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle(f'Two datasets fusion, diff K = {D.K_fit.unique()}')
    plt.tight_layout()
    if save:
        plt.savefig('sim_twoSess_fusion_diffK.pdf', format='pdf')
    plt.show()

def _plot_maps(U, cmap='tab20', grid=None, offset=1, dim=(30, 30),
               vmax=19, row_labels=None, save=True):
    """The helper function to plot the individual maps
    Args:
        U (pt.Tensor or np.ndarray): the individual maps,
                                     shape (num_subj, P)
        cmap: the color map of parcels
        grid: the grid layout of plot. e.g. [3, 4] means
              the maps will be displayed in 3 rows 4 columns
        offset: offset of figures
        dim: the dimensionality of the map. (width of MRF)
        vmax: the data range that the colormap covers for plotting
        row_labels: the labels of each row
        save: if True, save the figure
    Returns:
        Figure plot
    """
    N, P = U.shape
    if grid is None:
        grid = np.zeros((2,), np.int32)
        grid[0] = np.ceil(np.sqrt(N))
        grid[1] = np.ceil(N / grid[0])

    for n in range(N):
        ax = plt.subplot(grid[0], grid[1], n+offset)
        ax.imshow(U[n].reshape(dim).cpu().numpy(), cmap='tab20', interpolation='nearest', vmax=vmax)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if (row_labels is not None) and (n % N == 0):
            ax.axes.yaxis.set_visible(True)
            ax.set_yticks([])
            # ax.set_ylabel(row_labels[int(n / num_sub)])

    if save:
        plt.savefig(f'{row_labels}.pdf', format='pdf')
    plt.show()

def do_sessFusion_diffK(K_true=10, K=5, M=np.array([5],dtype=int),
                        nsubj_list=None, num_part=3, width=10,
                        low=3, high=30, arbitrary=None, sigma2=0.1,
                        plot_trueU=False, relevant=True):
    """Run the simulation of common kappa vs separate kappas across
       different sessions in the same set of subjects
    Args:
        K (int): the number of parcels
        M (list): the list of number of conditions per emission
        nsubj_list (list): the list of number of subjects per emission
        num_part (int): the number of partitions
        width (int): the width of MRF grid
        low_kappa (int or float): the lower kappa value
        high_kappa (int or float): the higher kappa value
        arbitrary (list): the list to specify the bad parcels in each
                          session. e.g [[0,1,2], [2,3]] indicates the parcel
                          0,1,2 are the parcels with same Vs in the first
                          session, parcel 2,3 are the parcels with same Vs
                          in the second session. If None, the default is to
                          have 3 bad parcels in first session and K-3 bad
                          parcels in the second session.
        plot_trueU (bool): Plot the true U and prior maps if True.
    Returns:
        grid (object): the MRF grid object initialized
        U (pt.Tensor): the true Us
        U_indv (list): the estimated individual U_hat fitted by different
                       models. They are all aligned with the true Us.
        Uerrors (list): the list of individual reconstruction errors by
                        different models
        Props (pt.Tensor): the True + fitted group prior
    """
    #Generate grid for easier visualization
    grid = sp.SpatialGrid(width=width, height=width)
    U_prior = []
    # generating data from GME
    # inits = np.array([820, 443, 188, 305, 717])
    T, Y, Y_test, U, _, signal, idx_all = make_true_model_GME(grid, K=K_true, P=grid.P,
                                                      nsubj_list=nsubj_list,
                                                  M=M, theta_mu=120, theta_w=1.5, sigma2=sigma2,
                                                  high_norm=high, low_norm=low, same_subj=True,
                                                  inits=None, relevant=relevant)
    # record number of bad parcels
    num_bp = [i.numel() for i in idx_all]
    U_prior.append(T.marginal_prob())
    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K_true), offset=1)
        plt.show()

    # Basic setting for fitting/evauation
    kappas, U_indv, Props = [], [], []
    all_ses = [x for x in np.arange(len(M))]
    em_indx = [all_ses] + [[x] for x in np.arange(len(M))] + [all_ses]*2
    fitting_model = ['D' + str(i + 1) for i in all_ses] + ['D_fusion','D_fusion_joint']
    results = pd.DataFrame()

    for model_typename in ['VMF', 'wVMF']:
        for common_kappa in [True, False]:
            models = []
            # Initialize multiple fitting models: dataset1, dataset2,..., dataset1 to N
            for j, fm_name in enumerate(fitting_model):
                if fm_name == 'D_fusion_joint': # model 1: joint session, common kappa
                    A = ar.ArrangeIndependent(K, grid.P, spatial_specific=True,
                                              remove_redundancy=False)
                    A.random_params()
                    # Making condition X and partition vector
                    part_vec = []
                    for i, m in enumerate(M):
                        this_X = np.kron(np.ones((num_part, 1)), np.eye(m))
                        if i == 0 :
                            X = block_diag(this_X)
                        else:
                            X = block_diag(X, this_X)

                        part_vec.append(np.kron(i, np.ones((m,))))

                    if model_typename == 'VMF':
                        emission = em.MixVMF(K=K, X=X, P=grid.P, part_vec=np.hstack(part_vec),
                                             uniform_kappa=common_kappa)
                    elif model_typename == 'wVMF':
                        emission = em.wMixVMF(K=K, X=X, P=grid.P, part_vec=np.hstack(part_vec),
                                              uniform_kappa=common_kappa)

                    emission.num_subj = nsubj_list[0]
                    this_M = FullMultiModel(A, [emission])
                    this_M.initialize()

                    models.append(this_M)
                    this_sub_list = [np.arange(x) for x in nsubj_list[em_indx[1]]]
                    models[j].initialize([pt.hstack(Y)], subj_ind=this_sub_list)
                else:
                    models.append(make_full_model(K=K, P=grid.P, M=M[em_indx[j+1]],
                                                  nsubj_list=nsubj_list[em_indx[j+1]],
                                                  num_part=num_part, common_kappa=common_kappa,
                                                  model_type=model_typename, same_subj=True))
                    data = [Y[i] for i in em_indx[j+1]]
                    this_sub_list = [np.arange(x) for x in nsubj_list[em_indx[j+1]]]
                    models[j].initialize(data, subj_ind=this_sub_list)

                # Fitting the full models
                models[j],_,_,Uhat,first_ll = \
                    models[j].fit_em_ninits(n_inits=40, first_iter=10, iter=100, tol=0.01,
                                            fit_emission=True, fit_arrangement=True,
                                            init_emission=True, init_arrangement=True,
                                            align = 'arrange')

                # ------------------------------------------
                # Now build the model for the test data and crossvalidate
                # across subjects
                if model_typename == 'VMF':
                    em_model = em.MixVMF(K=K, N=Y_test[0].shape[1], P=grid.P,
                                         X=None, uniform_kappa=common_kappa)
                elif model_typename == 'wVMF':
                    em_model = em.wMixVMF(K=K, N=Y_test[0].shape[1], P=grid.P,
                                          X=None, uniform_kappa=common_kappa)

                em_model.initialize(Y_test[0])
                models[j].emissions = [em_model]
                models[j].initialize()

                # Gets us the individual parcellation
                models[j], ll, theta, U_indiv = models[j].fit_em(iter=200, tol=0.1,
                                                                 fit_emission=True,
                                                                 fit_arrangement=False,
                                                                 first_evidence=False)
                # U_indiv = models[j].remap_evidence(U_indiv)

            # Split the grid by good-bad parcels
            group_true = pt.argmax(T.marginal_prob(), dim=0)
            g_idx, i_idx = [], []
            for idx in reversed(idx_all):
                g_idx.append((group_true == idx.unsqueeze(1)).nonzero()[:,1])
                i_idx.append([(U[i] == idx.unsqueeze(1)).nonzero()[:,1] for i in range(U.shape[0])])

            MM = [T] + models
            if K == K_true:
                # Align full models to the true
                Prop = ev.align_models(MM, in_place=True)
                Props.append(Prop[1:])
            else:
                # Align full models themselves
                Prop = ev.align_models(models, in_place=True)
                Props.append(Prop)

            models = MM[1:]
            UV_soft = []
            for mo in models:
                _, _, _, U_indiv = mo.fit_em(iter=200, tol=0.1, fit_emission=True,
                                             fit_arrangement=False, first_evidence=False)
                UV_soft.append(U_indiv)

            UV_hard = [pt.argmax(e, dim=1) for e in UV_soft]
            U_indv.append(UV_hard)

            # evaluation starts after model alignment
            for j, fm_name in enumerate(fitting_model):
                if K == K_true:
                    # 1. U reconstruction error
                    uerr_hard = u_err(U, UV_hard[j])
                    uerr_soft = ev.u_abserr(ar.expand_mn(U, K), UV_soft[j])
                    pd_Urecon = pd.DataFrame({'uerr_hard': [uerr_hard.mean().item()],
                                              'uerr_soft': [uerr_soft]})
                else:
                    pd_Urecon = pd.DataFrame(columns=['uerr_hard', 'uerr_soft'])
                # 1. ARI
                ari_group = ev.ARI(group_true, pt.argmax(models[j].arrange.logpi, dim=0))
                ari_indiv = [ev.ARI(U[i], UV_hard[j][i]) for i in range(U.shape[0])]
                # 2. dcbc
                Pgroup = pt.argmax(models[j].marginal_prob(), dim=0) + 1
                Pindiv = UV_hard[j]+1
                binWidth = 5
                max_dist = binWidth * pt.ceil(grid.Dist.max()/binWidth)
                dcbc_group = calc_test_dcbc(Pgroup, Y_test[0], grid.Dist,
                                            max_dist=int(max_dist),
                                            bin_width=binWidth)
                dcbc_indiv = calc_test_dcbc(Pindiv, Y_test[0], grid.Dist,
                                            max_dist=int(max_dist), bin_width=binWidth)
                # 3. non-adjusted/adjusted expected cosine error
                coserr, wcoserr = [], []
                for i, emi in enumerate(models[j].emissions):
                    coserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                            adjusted=False, soft_assign=True))
                    wcoserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                             adjusted=True, soft_assign=True))
                if not relevant:
                    if K == K_true:
                        # 1. U reconstruction error
                        uerr_hard_1 = [pt.abs(U[i][id] - UV_hard[j][i][id]).to(pt.bool).sum()
                                       /pt.abs(U[i][id] - UV_hard[j][i][id]).to(pt.bool).numel()
                                       for i, id in enumerate(i_idx[0])]
                        uerr_hard_2 = [pt.abs(U[i][id] - UV_hard[j][i][id]).to(pt.bool).sum()
                                       /pt.abs(U[i][id] - UV_hard[j][i][id]).to(pt.bool).numel()
                                       for i, id in enumerate(i_idx[1])]
                        uerr_soft_1 = [ev.u_abserr(ar.expand_mn_1d(U[i][id], K), UV_soft[j][i, :, id])
                                       for i, id in enumerate(i_idx[0])]
                        uerr_soft_2 = [ev.u_abserr(ar.expand_mn_1d(U[i][id], K), UV_soft[j][i, :, id])
                                       for i, id in enumerate(i_idx[1])]
                        pd_Urec_s = pd.DataFrame({'uerr_hard_1': [pt.stack(uerr_hard_1).mean().item()],
                                                  'uerr_hard_2': [pt.stack(uerr_hard_2).mean().item()],
                                                  'uerr_soft_1': [np.stack(uerr_soft_1).mean().item()],
                                                  'uerr_soft_2': [np.stack(uerr_soft_2).mean().item()]})
                    else:
                        pd_Urec_s = pd.DataFrame(columns=['uerr_hard_1', 'uerr_hard_2',
                                                          'uerr_soft_1', 'uerr_soft_2'])
                    # 1. ARI - split
                    ari_group_1 = ev.ARI(group_true[g_idx[0]],
                                         pt.argmax(models[j].arrange.logpi, dim=0)[g_idx[0]])
                    ari_group_2 = ev.ARI(group_true[g_idx[1]],
                                         pt.argmax(models[j].arrange.logpi, dim=0)[g_idx[1]])
                    ari_indiv_1 = [ev.ARI(U[i][id], UV_hard[j][i][id]) for i, id in enumerate(i_idx[0])]
                    ari_indiv_2 = [ev.ARI(U[i][id], UV_hard[j][i][id]) for i, id in enumerate(i_idx[1])]
                    # 2. DCBC - split
                    dcbc_group_1 = calc_test_dcbc(Pgroup[g_idx[0]], Y_test[0][:, :, g_idx[0]],
                                                  grid.Dist[g_idx[0], :][:, g_idx[0]],
                                                  max_dist=int(max_dist), bin_width=binWidth)
                    dcbc_group_2 = calc_test_dcbc(Pgroup[g_idx[1]], Y_test[0][:, :, g_idx[1]],
                                                  grid.Dist[g_idx[1], :][:, g_idx[1]],
                                                  max_dist=int(max_dist), bin_width=binWidth)
                    dcbc_indiv_1 = [compute_DCBC(maxDist=int(max_dist), binWidth=binWidth,
                                                 parcellation=Pindiv[i], dist=grid.Dist[id, :][:, id],
                                                 func=Y_test[0][i, :, id].T)['DCBC']
                                    for i, id in enumerate(i_idx[0])]
                    dcbc_indiv_2 = [compute_DCBC(maxDist=int(max_dist), binWidth=binWidth,
                                                 parcellation=Pindiv[i], dist=grid.Dist[id, :][:, id],
                                                 func=Y_test[0][i, :, id].T)['DCBC']
                                    for i, id in enumerate(i_idx[1])]
                    # 3. non-adjusted/adjusted expected cosine error - split
                    coserr1, coserr2, wcoserr1, wcoserr2 = [], [], [], []
                    for i, emi in enumerate(models[j].emissions):
                        coserr1.append(
                            pt.stack([ev.coserr(Y_test[0][i, :, id], emi.V, UV_soft[j][i, :, id],
                                                adjusted=False, soft_assign=True)
                                      for i, id in enumerate(i_idx[0])]).reshape(-1))
                        coserr2.append(
                            pt.stack([ev.coserr(Y_test[0][i, :, id], emi.V, UV_soft[j][i, :, id],
                                                adjusted=False, soft_assign=True)
                                      for i, id in enumerate(i_idx[1])]).reshape(-1))

                        wcoserr1.append(pt.stack([ev.coserr(Y_test[0][i, :, id], emi.V,
                                                            UV_soft[j][i, :, id],
                                                            adjusted=True, soft_assign=True)
                                                  for i, id in enumerate(i_idx[0])]).reshape(-1))
                        wcoserr2.append(pt.stack([ev.coserr(Y_test[0][i, :, id], emi.V,
                                                            UV_soft[j][i, :, id],
                                                            adjusted=True, soft_assign=True)
                                                  for i, id in enumerate(i_idx[1])]).reshape(-1))
                    add_pd = pd.DataFrame({'ari_group_1': [ari_group_1.item()],
                                           'ari_group_2': [ari_group_2.item()],
                                           'ari_indiv_1': [pt.stack(ari_indiv_1).mean().item()],
                                           'ari_indiv_2': [pt.stack(ari_indiv_2).mean().item()],
                                           'dcbc_group_1': [dcbc_group_1.mean().item()],
                                           'dcbc_group_2': [dcbc_group_2.mean().item()],
                                           'dcbc_indiv_1': [pt.stack(dcbc_indiv_1).mean().item()],
                                           'dcbc_indiv_2': [pt.stack(dcbc_indiv_2).mean().item()],
                                           'coserr_1': [pt.cat(coserr1).mean().item()],
                                           'coserr_2': [pt.cat(coserr2).mean().item()],
                                           'wcoserr_1': [pt.cat(wcoserr1).mean().item()],
                                           'wcoserr_2': [pt.cat(wcoserr2).mean().item()]})
                    add_pd = pd.concat([add_pd, pd_Urec_s], axis=1)
                else:
                    add_pd = pd.DataFrame(columns=['ari_group_1', 'ari_group_2', 'ari_indiv_1',
                                                   'ari_indiv_2', 'dcbc_group_1', 'dcbc_group_2',
                                                   'dcbc_indiv_1', 'dcbc_indiv_2', 'coserr_1',
                                                   'coserr_2', 'wcoserr_1', 'wcoserr_2', 'uerr_hard_1',
                                                   'uerr_hard_2', 'uerr_soft_1', 'uerr_soft_2'])

                res = pd.DataFrame({'model_type': [f'{model_typename}_{common_kappa}'],
                                    'K_true': [K_true],
                                    'K_fit': [K],
                                    'common_kappa': [common_kappa],
                                    'dataset': [fm_name],
                                    'sigma2': [sigma2],
                                    'low_signal': [low],
                                    'high_signal': [high],
                                    'num_bp_1': [num_bp[0]],
                                    'num_bp_2': [num_bp[1]],
                                    'ari_group': [ari_group.item()],
                                    'ari_indiv': [pt.stack(ari_indiv).mean().item()],
                                    'dcbc_group': [dcbc_group.mean().item()],
                                    'dcbc_indiv': [dcbc_indiv.mean().item()],
                                    'coserr': [pt.cat(coserr).mean().item()],
                                    'wcoserr': [pt.cat(wcoserr).mean().item()]})
                res = pd.concat([res, pd_Urecon, add_pd], axis=1)

                results = pd.concat([results, res], ignore_index=True)

            # Printing kappa fitting
            Kappa = pt.zeros((2, 4, K))
            MM = MM[1:]
            for j, ei in enumerate(em_indx[1:]):
                for k, i in enumerate(ei):
                    Kappa[i, j, :] = MM[j].emissions[0].kappa
            # print(Kappa)
            kappas.append(Kappa)

    return grid, U, U_prior, U_indv, Props, pt.stack(kappas), results

def simulation_3(K_true=10, K=6, width=30, nsub_list=np.array([10,10]),
                 M=np.array([10,10],dtype=int), num_part=2, sigma2=0.1,
                 low=0.1, high=1.1, iter=100, relevant=True):
    """Simulation of common kappa vs. separate kappas across subjects
    Args:
        width: The width and height of MRF grid
        nsub_list: the list of number of subject per dataset (emission model)
        M: The number of conditions per dataset (emission model)
        num_part: the number of partitions (sessions)
    Returns:
        Simulation result plots
    """
    results = pd.DataFrame()
    for i in range(iter):
        print(f'simulation {i}...')
        grid, U, U_prior, U_indv, Props, kappas, res = do_sessFusion_diffK(K_true=K_true,
                                                                           K=K, M=M,
                                                                           nsubj_list=nsub_list,
                                                                           num_part=num_part,
                                                                           width=width,
                                                                           low=low, high=high,
                                                                           sigma2=sigma2,
                                                                           plot_trueU=False,
                                                                           relevant=relevant)
        res['iter'] = i
        res['relevant'] = relevant
        results = pd.concat([results, res], ignore_index=True)

    # 1. Plot evaluation results
    plt.figure(figsize=(18, 10))
    crits = ['ari_group','dcbc_group','coserr','ari_indiv','dcbc_indiv','wcoserr']
    for i, c in enumerate(crits):
        plt.subplot(2, 3, i + 1)
        sb.barplot(x='dataset', y=c, hue='model_type', data=results, errorbar="se")
        plt.legend(loc='lower right')
        # if 'coserr' in c:
        #     plt.ylim(0.8, 1)

    plt.suptitle(f'Simulation 3, K_true={K_true}, K_fit={K}, iter={iter}')
    plt.show()

    # plot_result(grid, Props, names = ["Sess 1", "Sess 2", "Fusion", "Fusion_joint"], save=False)
    #
    # # Plot all true individual maps
    # _plot_maps(U, cmap='tab20', dim=(width, width), row_labels='True', save=True)
    #
    # # Plot fitted and aligned individual maps in dataset1, 2 and fusion
    # labels = ["commonKappa_", "separateKappa_"]
    # names = ['Sess 1', 'Sess 2', 'Fusion']
    # for i, us in enumerate(U_indv):
    #     for j in range(len(us)):
    #         _plot_maps(us[j], cmap='tab20', dim=(width, width),
    #                    row_labels=labels[i]+names[j], save=True)
    #
    return results

def cal_gradient(Y):
    """Calculate the functional gradients from data
       for simulation
    Args:
        Y: raw data. shape (width, height, N)
    Returns:
        functional gradient map
    """
    from scipy import ndimage
    w,h,d = Y.shape

    sobel = []
    for i in range(d):
        img = Y[:, :, i]
        sx = ndimage.sobel(img, axis=0, mode='constant')
        sy = ndimage.sobel(img, axis=1, mode='constant')
        sobel.append(np.hypot(sx, sy))

    sobel = np.stack(sobel).sum(axis=0)
    #sobel = (sobel - sobel.min())/ (sobel.max()-sobel.min())
    return sobel

def example_fusion_group(K_true=20, K=20, width=50, nsub_list=np.array([10,10]),
                         M=np.array([10,10],dtype=int), num_part=2, sigma2=0.1,
                         low=0.1, high=1.1, relevant=True):
    results = pd.DataFrame()
    grid, U, U_prior, U_indv, Props, kappas, res = do_sessFusion_diffK(K_true=K_true,
                                                                       K=K, M=M,
                                                                       nsubj_list=nsub_list,
                                                                       num_part=num_part,
                                                                       width=width,
                                                                       low=low, high=high,
                                                                       sigma2=sigma2,
                                                                       plot_trueU=False,
                                                                       relevant=relevant)

    # names = ["True", "Dataset 1", "Dataset 2", "Model_1", "Model_3"]
    #
    # true_group = pt.argmax(U_prior[0], dim=0)
    #
    # plt.figure(figsize=(12, 3))
    # plt.subplot(1, 5, 1)
    # grid.plot_maps(true_group)
    # plt.title(names[0])
    # plt.subplot(1, 5, 2)
    # grid.plot_maps(pt.argmax(Props[0][0, :, :], dim=0))
    # plt.title(names[1])
    # plt.subplot(1, 5, 3)
    # grid.plot_maps(pt.argmax(Props[0][1, :, :], dim=0))
    # plt.title(names[2])
    # plt.subplot(1, 5, 4)
    # grid.plot_maps(pt.argmax(Props[0][3, :, :], dim=0))
    # plt.title(names[3])
    # plt.subplot(1, 5, 5)
    # grid.plot_maps(pt.argmax(Props[0][2, :, :], dim=0))
    # plt.title(names[4])

    names = ["True", "Dataset 1", "Dataset 2", "Fusion", "Fusion_joint"]

    plt.figure(figsize=(15, 6))
    plt.subplot(2, 5, 1)
    grid.plot_maps(pt.argmax(U_prior[0], dim=0))
    plt.title(names[0])
    plt.subplot(2, 5, 2)
    grid.plot_maps(pt.argmax(Props[0][0, :, :], dim=0))
    plt.title(names[1])
    plt.subplot(2, 5, 3)
    grid.plot_maps(pt.argmax(Props[0][1, :, :], dim=0))
    plt.title(names[2])
    plt.subplot(2, 5, 4)
    grid.plot_maps(pt.argmax(Props[0][2, :, :], dim=0))
    plt.title(names[3])
    plt.subplot(2, 5, 5)
    grid.plot_maps(pt.argmax(Props[0][3, :, :], dim=0))
    plt.title(names[4])

    plt.subplot(2, 5, 6)
    grid.plot_maps(pt.argmax(U_prior[0], dim=0))
    plt.subplot(2, 5, 7)
    grid.plot_maps(pt.argmax(Props[1][0, :, :], dim=0))
    plt.subplot(2, 5, 8)
    grid.plot_maps(pt.argmax(Props[1][1, :, :], dim=0))
    plt.subplot(2, 5, 9)
    grid.plot_maps(pt.argmax(Props[1][2, :, :], dim=0))
    plt.subplot(2, 5, 10)
    grid.plot_maps(pt.argmax(Props[1][3, :, :], dim=0))

    plt.savefig('example_model34.pdf', format='pdf')
    plt.show()

def compare_model_1_3(D, title=None):
    D = D.replace(['D_fusion_joint'], ['Model_1'])
    D = D.replace(['D_fusion'], ['Model_3'])

    D = D.loc[(D['sigma2'] == '[0.5, 0.8]')]
    plt.figure(figsize=(8, 5))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        sb.barplot(data=D, x="dataset", order=['D1','D2','Model_1','Model_3'], y=c,
                   errorbar="se")
        if c =='dcbc_group':
            plt.ylim(0, 0.09)
        elif c =='dcbc_indiv':
            plt.ylim(0, 0.09)

    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle(f'Two datasets fusion, averaged across K and relevancy')
    plt.tight_layout()
    plt.savefig('fig_sim_model13.pdf', format='pdf')
    plt.show()

def compare_model_3_4(D, title=None):
    # D = D.replace(['D_fusion_joint'], ['Model_1'])
    # D = D.replace(['D_fusion'], ['Model_3'])
    # Fusion
    # D.loc[(D['dataset'] == 'D_fusion_joint') & (D['common_kappa'] == True), 'dataset'] = 'Model_1'
    # D.loc[(D['dataset'] == 'D_fusion') & (D['common_kappa'] == True), 'dataset'] = 'Model_3'
    # D.loc[(D['dataset'] == 'D_fusion_joint') & (D['common_kappa'] == False), 'dataset'] = 'Model_2'
    # D.loc[(D['dataset'] == 'D_fusion') & (D['common_kappa'] == False), 'dataset'] = 'Model_4'

    plt.figure(figsize=(8, 5))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        bar = sb.barplot(data=D, x="dataset", order=['D1', 'D2', 'D_fusion_joint', 'D_fusion'],
                         y=c, hue='common_kappa', hue_order=[True, False], errorbar="se")
        # sb.barplot(data=D, x="dataset", order=['D1','D2','Model_1','Model_3'], y=c,
        #            errorbar="se")

        if c =='dcbc_group':
            plt.ylim(0, 0.09)
        elif c =='dcbc_indiv':
            plt.ylim(0, 0.09)

    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle(f'Two datasets fusion, averaged across K and relevancy')
    plt.tight_layout()
    plt.savefig('fig_sim_model134_supp.pdf', format='pdf')
    plt.show()

def compare_diffK(generate_ck=False, save=False):
    if generate_ck:
        fname = model_dir + f'/Results/2.simulation/k_diff_simulation_heatmap_ck_aliMd.tsv'
    else:
        fname = model_dir + f'/Results/2.simulation/k_diff_simulation_heatmap_sk_aliMd.tsv'

    res_plot = pd.read_csv(fname, delimiter='\t')
    # 1. Plot evaluation results
    plt.figure(figsize=(10, 5))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        result = res_plot.pivot(index='K_true', columns='K_fit', values=c + '_dif')
        rdgn = sb.color_palette("vlag", as_cmap=True)
        # rdgn = sb.color_palette("Spectral", as_cmap=True)
        sb.heatmap(result, annot=False, cmap=rdgn, vmin=-0.03, vmax=0.03, center=0.00, fmt='.2g')
        plt.title(c)

    plt.suptitle(f'Simulation 4, different region signal strength, iter=100')
    plt.tight_layout()

    if save:
        plt.savefig('diff_Ktrue20_Kfit5to40.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    ########## session fusion (same subjects, different Ks) ##########
    # conditions = [([0.5,0.8], 1.1, 1.1, True),
    #               ([0.2,0.5], 1.1, 1.1, True),
    #               ([0.2,0.2], 0.1, 1.1, False),
    #               ([0.8,0.8], 0.1, 1.1, False)]
    conditions = [([0.2,0.2], 0.1, 1.1, False)]
    for i, (noise_level,low,high,re) in enumerate(conditions):
        D = pd.DataFrame()
        for k in [20]:
            res = simulation_3(K_true=20, K=k, width=50,
                               nsub_list=np.array([10, 10]),
                               M=np.array([40, 20], dtype=int),
                               num_part=1, sigma2=noise_level,
                               low=low, high=high,
                               iter=10, relevant=re)
            D = pd.concat([D, res], ignore_index=True)
        D.to_csv(f'eval_Ktrue_20_Kfit_20_Fusion_merged_joint.tsv', index=False, sep='\t')

    # fname = model_dir + f'/Results/2.simulation/eval_Ktrue_20_Kfit_5to40_all_models.tsv'
    # D = pd.read_csv(fname, delimiter='\t')
    #_plot_diffK(D, style="common_kappa", save=False)

    ########## Comparing model 1 and 3 ##########
    # fname = model_dir + f'/Results/2.simulation/eval_Ktrue_20_Kfit_20_Fusion_merged_joint.tsv'
    # D = pd.read_csv(fname, delimiter='\t')
    # D = D.loc[(D['K_fit']==20) & (D['relevant']==True)
    #           & (D.common_kappa==True) & (D['sigma2'] == '[0.5, 0.8]')]
    # compare_model_1_3(D, title='K_true=20, K_fit=20, relevant=True, model1 and model3')

    ########## Comparing model 3 and 4 ##########
    # fname = model_dir + f'/Results/2.simulation/eval_Ktrue_20_Kfit_20_Fusion_merged_joint.tsv'
    # # fname = f'eval_Ktrue_20_Kfit_20_Fusion_merged_joint.tsv'
    # D = pd.read_csv(fname, delimiter='\t')
    # # D = D.loc[(D['K_fit']==20) & (D['relevant']==True)]
    # compare_model_3_4(D, title='K_true=20, K_fit=20, relevant=False, model3 and model4')

    # example_fusion_group(K_true=20, K=20, width=50, nsub_list=np.array([10,10]),
    #                      M=np.array([40,20],dtype=int), num_part=1, sigma2=[0.3,0.6],
    #                      low=0.7, high=0.7, relevant=True)

    ########## Comparing model 3vs4 when different fitting/true k ##########
    compare_diffK(generate_ck=False, save=True)