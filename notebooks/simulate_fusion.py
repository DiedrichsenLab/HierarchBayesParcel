#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation 1: datasets fusion
   The figure 2 in the manuscript

Created on 11/24/2022 at 11:49 AM
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
from sklearn.metrics.pairwise import cosine_similarity
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.evaluate import calc_test_dcbc
from copy import copy, deepcopy
import pandas as pd
import seaborn as sb

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
                        sigma2=1.0, high_norm=0.9, low_norm=0.1,
                        same_subj=False):
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
    K_group = pt.arange(1, K).chunk(len(M))
    for i,m in enumerate(M):
        # Step 2: up the emission model and sample from it with a specific signal
        emissionT = em.MixGaussianExp(K=K, N=m, P=P, num_signal_bins=100, std_V=True)
        emissionT.sigma2 = pt.tensor(sigma2)
        # Initialize all V to be the high norm
        emissionT.V = emissionT.V * high_norm

        # Making ambiguous boundaries between random number parcelsby set low signal
        # for k-neighbouring parcels. The parcels have same V magnitude in this emission
        num_badpar = K_group[i][int(pt.randint(K_group[i].numel(), ()))]
        _, _, idx = _compute_adjacency(label_map, int(num_badpar))
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

    # # Making ambiguous boundaries by set the same V_k for k-neighbouring parcels
    # label_map = pt.argmax(T.arrange.logpi, dim=0).reshape(grid.dim)
    # # the parcels have same V magnitude in dataset1
    # _, base, idx_1 = _compute_adjacency(label_map, int(pt.randint(1, K, ())))
    #
    # # the parcels have same V magnitude in dataset2
    # _, base, idx_2 = _compute_adjacency(label_map, int(pt.randint(1, K, ())))
    # # idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
    # idx_all = [idx_1, idx_2]
    # print(idx_all)
    # for i, par in enumerate(idx_all):
    #     # Making the V magnitude of bad parcels as low norm
    #     for j in range(0, len(par)):
    #         unit_vec = T.emissions[i].V[:, par[j]] / pt.sqrt(pt.sum(T.emissions[i].V[:, par[j]] **
    #                                                                2, dim=0))
    #         T.emissions[i].V[:, par[j]] = unit_vec * low_norm

    # Sampling individual Us and data, data_test
    Y, Y_test, signal = [], [], []
    U, _ = T.sample()
    U_test, _ = T.sample()
    for m, Us in enumerate(T.distribute_evidence(U)):
        Y.append(T.emissions[m].sample(Us, signal=pt.ones(Us.shape)))

    # Build a separate emission model contains all parcel infomation
    # for generating test data
    em_test = em.MixGaussianExp(K=K, N=sum(M)*2, P=P, num_signal_bins=100, std_V=True)
    em_test.sigma2 = pt.tensor(sigma2)
    em_test.V = em_test.V * high_norm
    if same_subj:
        Uind = [U]
    else:
        Uind = T.distribute_evidence(U_test)

    # Get the true signals and Y_test
    for i, Us in enumerate(Uind):
        Y_test.append(em_test.sample(Us, signal=pt.ones(Us.shape)))
        signal.append(pt.where(pt.isin(Us, idx_all[i]), low_norm, high_norm))

    return T, Y, Y_test, U, U_test, signal

def make_true_model_VMF(grid, K=5, P=100, nsubj_list=[10,10],
                        M=[5,5], # Number of conditions per data set
                        num_part=1, theta_mu=150, theta_w=20,
                        inits=None, common_kappa=False,
                        high_kappa=30, low_kappa=3, same_subj=False):
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

    emissions =[]
    for i,m in enumerate(M):
        X = np.kron(np.ones((num_part,1)),np.eye(m))
        part_vec = np.kron(np.arange(num_part),np.ones((m,)))
        emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                              uniform_kappa=common_kappa,
                              weighting='ones')
        emission.num_subj=nsubj_list[i]
        # Initialize all kappas to be the high value
        emission.kappa = pt.full(emission.kappa.shape, high_kappa)
        emissions.append(emission)

    T = FullMultiModel(A,emissions)
    if same_subj: # All emission models have the same set of subjects
        # This is used to simulate fusion across sessions
        assert np.all(nsubj_list == nsubj_list[0]), \
            "The number of subjects should be same across all emission models"
        sub_list = [np.arange(x) for x in nsubj_list]
        T.initialize(subj_ind=sub_list)
    else: # Emission models to have separate set subjects
        T.initialize()

    # Making ambiguous boundaries by set the same V_k for k-neighbouring parcels
    label_map = pt.argmax(T.arrange.logpi, dim=0).reshape(grid.dim)
    _, base, idx = _compute_adjacency(label_map, 2)

    # the parcels have same V magnitude in dataset1
    idx_1 = pt.cat((idx, base.view(1)))
    # the parcels have same V magnitude in dataset2
    idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
    idx_all = [idx_1, idx_2]
    print(base, idx_all)
    for i, par in enumerate([idx_1, idx_2]):
        # Making the V align to the first parcel
        for j in range(1, len(par)):
            T.emissions[i].V[:, par[j]] = T.emissions[i].V[:, par[0]]

        # Set kappas of k-neighbouring parcels to low_kappa
        for k in range(len(par)):
            if not T.emissions[i].uniform_kappa:
                T.emissions[i].kappa[par[k]] = low_kappa
            else:
                raise ValueError(
                    "The kappas of emission models need to be separate in simulation")

    # Sampling individual Us and data, data_test
    Y, Y_test = [], []
    U, Y = T.sample()
    _, Y_test = T.sample(U=U)

    # Get the true signals
    signal = [None] * len(T.subj_ind)
    for i, us in enumerate(T.distribute_evidence(U)):
        signal[i] = pt.where(pt.isin(us, idx_all[i]), low_kappa, high_kappa)

    return T, Y, Y_test, U, signal

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

def data_visualize(Y, U):
    # Visualizing sampled data
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{'type': 'surface'}]],
                        subplot_titles=["GME data"])

    fig.add_trace(go.Scatter3d(x=Y[0][0, 0, :].cpu().numpy(),
                               y=Y[0][0, 1, :].cpu().numpy(),
                               z=Y[0][0, 2, :].cpu().numpy(),
                               mode='markers', marker=dict(size=3, opacity=0.7,
                                                           color=U[0].cpu().numpy())),
                  row=1, col=1)
    fig.show()

def do_simulation_sessFusion(K=5, nsubj_list=None,width = 10):
    """Run the missing data simulation at given missing rate
    Args:
        K: the clusters number
        P: the voxel number
        nsubj_list:
    Returns:
        theta_all: All parameters at each EM iteration
        Uerr_all: The absolute error between U and U_hat for each missing rate
        U: The ground truth Us
        U_nan_all: the ground truth Us with missing data
        U_hat_all: the predicted U_hat for each missing rate
    """
    M=np.array([5,5],dtype=int)
    #Generate grid for easier visualization
    grid = sp.SpatialGrid(width=width, height=width)
    pm = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    pm.random_smooth_pi(grid.Dist, theta_mu=20)

    T = make_full_model(K=K, P=grid.P, nsubj_list=nsubj_list)
    T.arrange.logpi = pm.logpi
    T.emissions[0].kappa = pt.tensor([30,30,30,3,3])
    T.emissions[1].kappa = pt.tensor([3,3,3,30,30])
    U,Y = T.sample()
    models = []

    em_indx = [[0,1],[0],[1],[0,1]]

    for j in range(3):
        models.append(make_full_model(K=K,P=grid.P,M=M[em_indx[j+1]],nsubj_list=nsubj_list[em_indx[j+1]]))
        data = [Y[i] for i in em_indx[j+1]]
        models[j].initialize(data)

    for i,m in enumerate(models):
        models[i],ll,theta,Uhat,first_ll = \
            models[i].fit_em_ninits(n_inits=40, first_iter=7, iter=100, tol=0.01,
            fit_emission=True, fit_arrangement=True,
            init_emission=True, init_arrangement=True,
            align = 'arrange')
    MM = [T]+models
    Prop = ev.align_models(MM)

    for i in range(len(MM)):
        plt.subplot(2,2,i+1)
        parcel = np.argmax(Prop[i,:,:],axis=0)
        grid.plot_maps(parcel,vmax=5)

    Kappa = np.zeros((2,4,K))
    for j,ei in enumerate(em_indx):
        for k,i in enumerate(ei):
            Kappa[i,j,:]=MM[j].emissions[k].kappa

    print(Kappa.round(1))
    pass

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
                        plot_trueU=False):
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
    # Option 2: generating data from GME
    # inits = np.array([820, 443, 188, 305, 717])
    T, Y, Y_test, U, _, signal = make_true_model_GME(grid, K=K_true, P=grid.P, nsubj_list=nsubj_list,
                                                  M=M, theta_mu=120, theta_w=1.5, sigma2=sigma2,
                                                  high_norm=high, low_norm=low, same_subj=True,
                                                  inits=None)
    U_prior.append(T.marginal_prob())
    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K_true), offset=1)
        plt.show()

    # Basic setting for fitting/evauation
    kappas, U_indv, Props = [], [], []
    all_ses = [x for x in np.arange(len(M))]
    em_indx = [all_ses] + [[x] for x in np.arange(len(M))] + [all_ses]
    fitting_model = ['D' + str(i + 1) for i in all_ses] + ['D_fusion']
    results = pd.DataFrame()

    for common_kappa in [True, False]:
        models = []
        # Initialize multiple fitting models: dataset1, dataset2,..., dataset1 to N
        for j, fm_name in enumerate(fitting_model):
            models.append(make_full_model(K=K, P=grid.P, M=M[em_indx[j+1]],
                                          nsubj_list=nsubj_list[em_indx[j+1]],
                                          num_part=num_part, common_kappa=common_kappa,
                                          model_type='VMF', same_subj=True))
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
            em_model = em.MixVMF(K=K, N=Y_test[0].shape[1], P=grid.P,
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

        # Align full models to the true
        MM = [T] + models
        Prop = ev.align_models(models, in_place=True)
        Props.append(Prop)
        UV_soft = [e.Estep()[0] for e in MM[1:]]
        UV_hard = [pt.argmax(e, dim=1) for e in UV_soft]
        U_indv.append(UV_hard)

        # evaluation starts after model alignment
        models = MM[1:]
        for j, fm_name in enumerate(fitting_model):
            # 1. ARI
            ari_group = ev.ARI(pt.argmax(T.marginal_prob(), dim=0),
                               pt.argmax(models[j].arrange.logpi, dim=0))
            ari_indiv = [ev.ARI(U[i], UV_hard[j][i]) for i in range(U.shape[0])]
            # 2. dcbc
            Pgroup = pt.argmax(models[j].marginal_prob(), dim=0) + 1
            dcbc_group = calc_test_dcbc(Pgroup, Y_test[0], grid.Dist)
            dcbc_indiv = calc_test_dcbc(UV_hard[j], Y_test[0], grid.Dist)
            # 3. non-adjusted/adjusted expected cosine error
            coserr, wcoserr = [], []
            for i, emi in enumerate(models[j].emissions):
                coserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                        adjusted=False, soft_assign=True))
                wcoserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                         adjusted=True, soft_assign=True))

            res = pd.DataFrame({'model_type': [f'VMF_{common_kappa}'],
                                'dataset': [fm_name],
                                'ari_group': [ari_group.item()],
                                'ari_indiv': [pt.stack(ari_indiv).mean().item()],
                                'dcbc_group': [dcbc_group.mean().item()],
                                'dcbc_indiv': [dcbc_indiv.mean().item()],
                                'coserr': [pt.cat(coserr).mean().item()],
                                'wcoserr': [pt.cat(wcoserr).mean().item()]})
            results = pd.concat([results, res], ignore_index=True)

        # Printing kappa fitting
        Kappa = pt.zeros((2, 3, K))
        MM = MM[1:]
        for j, ei in enumerate(em_indx[1:]):
            for k, i in enumerate(ei):
                Kappa[i, j, :] = MM[j].emissions[0].kappa
        # print(Kappa)
        kappas.append(Kappa)

    return grid, U, U_prior, U_indv, Props, pt.stack(kappas), results

def do_simulation_sessFusion_subj(K=5, M=np.array([5,5],dtype=int), nsubj_list=None,
                                  num_part=3, width=10, low=3, high=30,
                                  sigma2=0.1, plot_trueU=False):
    """Run the missing data simulation at given missing rate
    Args:
        K: the clusters number
        P: the voxel number
        nsubj_list:
    Returns:
        theta_all: All parameters at each EM iteration
        Uerr_all: The absolute error between U and U_hat for each missing rate
        U: The ground truth Us
        U_nan_all: the ground truth Us with missing data
        U_hat_all: the predicted U_hat for each missing rate
    """
    #Generate grid for easier visualization
    grid = sp.SpatialGrid(width=width, height=width)

    # Option 1: generating data from VMF
    # T, Y, Y_test, U, signal = make_true_model_VMF(grid, K=K, P=grid.P, nsubj_list=nsubj_list,
    #                                               M=M, theta_mu=20, theta_w=1.3, inits=None,
    #                                               common_kappa=False, high_kappa=high,
    #                                               low_kappa=low)

    # Option 2: generating data from GME
    # inits = np.array([820, 443, 188, 305, 717])
    T, Y, Y_test, U, U_test, signal = make_true_model_GME(grid, K=K, P=grid.P,
                                                          nsubj_list=nsubj_list,
                                                  M=M, theta_mu=120, theta_w=1.5, sigma2=sigma2,
                                                  high_norm=high, low_norm=low,
                                                  inits=None)

    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.show()

    # Visualizing sampled data
    # data_visualize(Y, U)

    # Basic setting for fitting/evauation
    U_indv, Props, kappas = [], [], []
    all_ses = [x for x in np.arange(len(M))]
    em_indx = [all_ses] + [[x] for x in np.arange(len(M))] + [all_ses]
    fitting_model = ['D' + str(i + 1) for i in all_ses] + ['D_fusion']
    results = pd.DataFrame()

    Ys = [[Y[0]], [Y[1]], Y]
    # Two options: the nsubj of Y_test can be either combined the two
    # datasets, or pick the first datasets.
    # Ys_test = pt.vstack(Y_test)
    U_test = T.distribute_evidence(U_test)[0]
    Ys_test = Y_test[0]
    signals = [[signal[0]], [signal[1]], signal]

    # Main loop
    for w in ['VMF']:
        for common_kappa in [True, False]:
            models = []
            # Initialize three full models: dataset1, dataset2, dataset1 and 2
            for j, fm_name in enumerate(fitting_model):
                models.append(make_full_model(K=K,P=grid.P,M=M[em_indx[j+1]],num_part=num_part,
                                              nsubj_list=nsubj_list[em_indx[j+1]],
                                              common_kappa=common_kappa, model_type=w))
                if w == 'wVMF_t2':
                    for i, emi in enumerate(models[j].emissions):
                        emi.initialize(Ys[j][i], weight=signals[j][i]**2)
                elif w == 'wVMF_t2P':
                    for i, emi in enumerate(models[j].emissions):
                        this_W = signals[j][i]**2
                        ratio = this_W.size(dim=1)/this_W.sum(dim=1, keepdim=True)
                        emi.initialize(Ys[j][i], weight=this_W * ratio)
                else:
                    models[j].initialize(Ys[j])

                # Fitting the full models
                models[j],ll,theta,Uhat,first_ll = \
                    models[j].fit_em_ninits(n_inits=40, first_iter=10, iter=100, tol=0.01,
                    fit_emission=True, fit_arrangement=True,
                    init_emission=True, init_arrangement=True,
                    align = 'arrange')

                # ------------------------------------------
                # Now build the model for the test data and crossvalidate
                # across subjects
                em_model = em.MixVMF(K=K, N=Ys_test.shape[1], P=grid.P,
                                     X=None, uniform_kappa=common_kappa)
                em_model.initialize(Ys_test)
                models[j].emissions = [em_model]
                models[j].initialize()

                # Gets us the individual parcellation
                models[j], ll, theta, U_indiv = models[j].fit_em(iter=200, tol=0.1,
                                                                 fit_emission=True,
                                                                 fit_arrangement=False,
                                                                 first_evidence=False)
                # U_indiv = models[j].remap_evidence(U_indiv)

            # Align full models to the true, and store U and U_indv
            MM = [T]+models
            Prop = ev.align_models(MM, in_place=True)
            Props.append(Prop)
            UV_soft = [e.Estep()[0] for e in MM[1:]]
            UV_hard = [pt.argmax(e, dim=1) for e in UV_soft]
            U_indv.append(UV_hard)

            # evaluation starts after model alignment
            models = MM[1:]
            for j, fm_name in enumerate(fitting_model):
                # 1. Hard U reconstruction error
                uerr_hard = u_err(U_test, UV_hard[j])
                # 2. Soft U reconstruction error
                uerr_soft = ev.u_abserr(ar.expand_mn(U_test, K), UV_soft[j])
                # 3. non-adjusted/adjusted expected cosine error
                coserr, wcoserr = [], []
                for i, emi in enumerate(models[j].emissions):
                    coserr.append(ev.coserr(Ys_test, emi.V, UV_soft[j],
                                            adjusted=False, soft_assign=True))
                    wcoserr.append(ev.coserr(Ys_test, emi.V, UV_soft[j],
                                             adjusted=True, soft_assign=True))

                res = pd.DataFrame({'model_type': [f'{w}_{common_kappa}'],
                                    'dataset': [fm_name],
                                    'uerr_hard': [uerr_hard.mean().item()],
                                    'uerr_soft': [uerr_soft],
                                    'coserr': [pt.cat(coserr).mean().item()],
                                    'wcoserr': [pt.cat(wcoserr).mean().item()]})
                results = pd.concat([results, res], ignore_index=True)

            # Visualizing fitted results (checking results)
            # data_visualize([Y, [Y[0]], [Y[1]], Y], [U]+this_UV)

            # Printing kappa fitting
            Kappa = pt.zeros((2,3,K))
            MM = MM[1:]
            for j,ei in enumerate(em_indx[1:]):
                for k,i in enumerate(ei):
                    Kappa[i,j,:]=MM[j].emissions[0].kappa
            # print(Kappa)
            kappas.append(Kappa)

    return grid, U, U_test, U_indv, Props, pt.stack(kappas), results

def do_simulation_sessFusion_sess(K=5, M=np.array([5],dtype=int),
                                  nsubj_list=None,
                                  num_part=3,
                                  width=10,
                                  low=3,
                                  high=30,
                                  arbitrary=None,
                                  sigma2=0.1,
                                  plot_trueU=False):
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

    # Option 2: generating data from GME
    # inits = np.array([820, 443, 188, 305, 717])
    T, Y, Y_test, U, _, signal = make_true_model_GME(grid, K=K, P=grid.P, nsubj_list=nsubj_list,
                                                  M=M, theta_mu=120, theta_w=1.5, sigma2=sigma2,
                                                  high_norm=high, low_norm=low, same_subj=True,
                                                  inits=None)

    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.show()

    # Basic setting for fitting/evauation
    kappas, U_indv, Props = [], [], []
    all_ses = [x for x in np.arange(len(M))]
    em_indx = [all_ses] + [[x] for x in np.arange(len(M))] + [all_ses]
    fitting_model = ['D' + str(i + 1) for i in all_ses] + ['D_fusion']
    results = pd.DataFrame()

    for common_kappa in [True, False]:
        models = []
        # Initialize multiple fitting models: dataset1, dataset2,..., dataset1 to N
        for j, fm_name in enumerate(fitting_model):
            models.append(make_full_model(K=K, P=grid.P, M=M[em_indx[j+1]],
                                          nsubj_list=nsubj_list[em_indx[j+1]],
                                          num_part=num_part, common_kappa=common_kappa,
                                          model_type='VMF', same_subj=True))
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
            em_model = em.MixVMF(K=K, N=Y_test[0].shape[1], P=grid.P,
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

        # Align full models to the true
        MM = [T] + models
        Prop = ev.align_models(MM, in_place=True)
        Props.append(Prop)
        UV_soft = [e.Estep()[0] for e in MM[1:]]
        UV_hard = [pt.argmax(e, dim=1) for e in UV_soft]
        U_indv.append(UV_hard)

        # evaluation starts after model alignment
        models = MM[1:]
        for j, fm_name in enumerate(fitting_model):
            # 1. U reconstruction error
            uerr_hard = u_err(U, UV_hard[j])
            uerr_soft = ev.u_abserr(ar.expand_mn(U, K), UV_soft[j])
            # 2. dcbc
            Pgroup = pt.argmax(models[j].marginal_prob(), dim=0) + 1
            dcbc_group = calc_test_dcbc(Pgroup, Y_test[0], grid.Dist)
            dcbc_indiv = calc_test_dcbc(UV_hard[j], Y_test[0], grid.Dist)
            # 3. non-adjusted/adjusted expected cosine error
            coserr, wcoserr = [], []
            for i, emi in enumerate(models[j].emissions):
                coserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                        adjusted=False, soft_assign=True))
                wcoserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[j],
                                         adjusted=True, soft_assign=True))

            res = pd.DataFrame({'model_type': [f'VMF_{common_kappa}'],
                                'dataset': [fm_name],
                                'uerr_hard': [uerr_hard.mean().item()],
                                'uerr_soft': [uerr_soft],
                                'dcbc_group': [dcbc_group.mean().item()],
                                'dcbc_indiv': [dcbc_indiv.mean().item()],
                                'coserr': [pt.cat(coserr).mean().item()],
                                'wcoserr': [pt.cat(wcoserr).mean().item()]})
            results = pd.concat([results, res], ignore_index=True)

        # Printing kappa fitting
        Kappa = pt.zeros((2, 3, K))
        MM = MM[1:]
        for j, ei in enumerate(em_indx[1:]):
            for k, i in enumerate(ei):
                Kappa[i, j, :] = MM[j].emissions[0].kappa
        # print(Kappa)
        kappas.append(Kappa)

    return grid, U, U_indv, Props, pt.stack(kappas), results

def do_sim_diffK_fit(K_true=10, K=5, M=np.array([5],dtype=int),
                     nsubj_list=None, num_part=3, width=10,
                     low=3, high=30, arbitrary=None, sigma2=0.1,
                     plot_trueU=False):
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
    # Option 2: generating data from GME
    # inits = np.array([820, 443, 188, 305, 717])
    T, Y, Y_test, U, _, signal = make_true_model_GME(grid, K=K_true, P=grid.P, nsubj_list=nsubj_list,
                                                  M=M, theta_mu=120, theta_w=1.5, sigma2=sigma2,
                                                  high_norm=high, low_norm=low, same_subj=True,
                                                  inits=None)
    U_prior.append(T.marginal_prob())
    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K_true), offset=1)
        plt.show()

    # Basic setting for fitting/evauation
    kappas, U_indv, Props = [], [], []
    results = pd.DataFrame()

    for common_kappa in [True, False]:
        # Initialize multiple fitting models: dataset1, dataset2,..., dataset1 to N

        model = make_full_model(K=K, P=grid.P, M=M,
                                nsubj_list=nsubj_list,
                                num_part=num_part, common_kappa=common_kappa,
                                model_type='VMF', same_subj=True)
        model.initialize(Y, subj_ind=T.subj_ind)

        # Fitting the full models
        model,_,_,Uhat,_ = model.fit_em_ninits(n_inits=40, first_iter=10, iter=100,
                                               tol=0.01,fit_emission=True,
                                               fit_arrangement=True,init_emission=True,
                                               init_arrangement=True, align = 'arrange')

        # ------------------------------------------
        # Now build the model for the test data and crossvalidate
        # across subjects
        em_model = em.MixVMF(K=K, N=Y_test[0].shape[1], P=grid.P,
                             X=None, uniform_kappa=common_kappa)
        em_model.initialize(Y_test[0])
        model.emissions = [em_model]
        model.initialize()

        # Gets us the individual parcellation
        model, ll, theta, U_indiv = model.fit_em(iter=200, tol=0.1,
                                                 fit_emission=True,
                                                 fit_arrangement=False,
                                                 first_evidence=False)
        # U_indiv = models[j].remap_evidence(U_indiv)
        MM = [T] + [model]
        Props.append(model.marginal_prob())
        UV_soft = [e.Estep()[0] for e in MM[1:]]
        UV_hard = [pt.argmax(e, dim=1) for e in UV_soft]
        U_indv.append(UV_hard)

        # evaluation starts after model alignment
        # 1. ARI
        ari_group = ev.ARI(pt.argmax(T.marginal_prob(), dim=0),
                           pt.argmax(model.arrange.logpi, dim=0))
        ari_indiv = [ev.ARI(U[i], UV_hard[0][i]) for i in range(U.shape[0])]
        # 2. dcbc
        Pgroup = pt.argmax(model.marginal_prob(), dim=0) + 1
        dcbc_group = calc_test_dcbc(Pgroup, Y_test[0], grid.Dist)
        dcbc_indiv = calc_test_dcbc(UV_hard[0], Y_test[0], grid.Dist)
        # 3. non-adjusted/adjusted expected cosine error
        coserr, wcoserr = [], []
        for i, emi in enumerate(model.emissions):
            coserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[0],
                                    adjusted=False, soft_assign=True))
            wcoserr.append(ev.coserr(Y_test[0], emi.V, UV_soft[0],
                                     adjusted=True, soft_assign=True))

        res = pd.DataFrame({'model_type': [f'VMF_{common_kappa}'],
                            'common_kappa': common_kappa,
                            'ari_group': [ari_group.item()],
                            'ari_indiv': [pt.stack(ari_indiv).mean().item()],
                            'dcbc_group': [dcbc_group.mean().item()],
                            'dcbc_indiv': [dcbc_indiv.mean().item()],
                            'coserr': [pt.cat(coserr).mean().item()],
                            'wcoserr': [pt.cat(wcoserr).mean().item()]})
        results = pd.concat([results, res], ignore_index=True)

    return grid, U, U_prior, U_indv, Props, results

def plot_results(results):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    sb.barplot(data=results,x='model_type',y='uerr')
    plt.subplot(1,3,2)
    sb.barplot(data=results,x='model_type',y='coserr')
    plt.subplot(1,3,3)
    sb.barplot(data=results,x='model_type',y='wcoserr')

    plt.show()

def simulation_1(K=5, width=30,
                 nsub_list=np.array([12,8]),
                 M=np.array([10,10],dtype=int),
                 num_part=2, sigma2=0.1):
    """Simulation of common kappa vs. separate kappas across subjects
    Args:
        width: The width and height of MRF grid
        nsub_list: the list of number of subject per dataset (emission model)
        M: The number of conditions per dataset (emission model)
    Returns:
        Simulation result plots
    """
    results = pd.DataFrame()
    for i in range(100):
        print(f'simulation {i}...')
        grid, U, Ut, U_indv, Props, kappas, res = do_simulation_sessFusion_subj(K=K, M=M,
                                                                        nsubj_list=nsub_list,
                                                                        num_part=num_part,
                                                                        width=width,
                                                                        low=0.1, high=1.1,
                                                                        sigma2=sigma2,
                                                                        plot_trueU=False)

        res['iter'] = i
        results = pd.concat([results, res], ignore_index=True)

    # 1. Plot evaluation results
    plt.figure(figsize=(15,20))
    crits = ['uerr_hard','uerr_soft','coserr','wcoserr']
    for i, c in enumerate(crits):
        plt.subplot(4, 1, i+1)
        sb.barplot(x='dataset', y=c, hue='model_type', data=results, errorbar="se")
        plt.legend(loc='upper right')
    plt.show()

    # 2. Plot the group reconstruction result from the last simulation
    plot_result(grid, Props, ylabels=results['model_type'].unique(), save=True)

    # Plot all true individual maps
    _plot_maps(Ut, cmap='tab20', dim=(width, width), row_labels='True', save=True)

    # Plot fitted and aligned individual maps in dataset1, 2 and fusion
    labels = ["commonKappa_VMF", "separateKappa_VMF"]
    names = ["Dataset 1", "Dataset 2", "Dataset 1+2"]
    for i, us in enumerate(U_indv):
        for j in range(len(us)):
            _plot_maps(us[j], cmap='tab20', dim=(width, width),
                       row_labels=labels[i]+names[j], save=True)

    pass

def simulation_2(K=6, width=30,
                 nsub_list=np.array([10,10]),
                 M=np.array([10,10],dtype=int),
                 num_part=2, sigma2=0.1, iter=100):
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
        grid, U, U_indv, Props, kappas, res = do_simulation_sessFusion_sess(K=K, M=M,
                                                                        nsubj_list=nsub_list,
                                                                        num_part=num_part,
                                                                        width=width,
                                                                        low=0.1, high=1.1,
                                                                        sigma2=sigma2,
                                                                        plot_trueU=False)
        res['iter'] = i
        results = pd.concat([results, res], ignore_index=True)

    # 1. Plot evaluation results
    plt.figure(figsize=(18, 10))
    crits = ['uerr_hard', 'dcbc_group', 'coserr', 'uerr_soft', 'dcbc_indiv', 'wcoserr']
    for i, c in enumerate(crits):
        plt.subplot(2, 3, i + 1)
        sb.barplot(x='dataset', y=c, hue='model_type', data=results, errorbar="se")
        plt.legend(loc='lower right')
        if 'coserr' in c:
            plt.ylim(0.8, 1)

    plt.suptitle(f'Simulation 2, K_true={K}, K_fit={K}, iter={iter}')
    plt.show()

    # plot_result(grid, Props, names = ["True", "Sess 1", "Sess 2", "Fusion"],
    #             save=True)
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
    # pass

def simulation_3(K_true=10, K=6, width=30, nsub_list=np.array([10,10]),
                 M=np.array([10,10],dtype=int), num_part=2, sigma2=0.1,
                 iter=100):
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
                                                                           low=0.1, high=1.1,
                                                                           sigma2=sigma2,
                                                                           plot_trueU=False)
        res['iter'] = i
        results = pd.concat([results, res], ignore_index=True)

    # 1. Plot evaluation results
    plt.figure(figsize=(18, 10))
    crits = ['ari_group','dcbc_group','coserr','ari_indiv','dcbc_indiv','wcoserr']
    for i, c in enumerate(crits):
        plt.subplot(2, 3, i + 1)
        sb.barplot(x='dataset', y=c, hue='model_type', data=results, errorbar="se")
        plt.legend(loc='lower right')
        if 'coserr' in c:
            plt.ylim(0.8, 1)

    plt.suptitle(f'Simulation 3, K_true={K_true}, K_fit={K}, iter={iter}')
    plt.show()

    # plot_result(grid, Props, names = ["Sess 1", "Sess 2", "Fusion"], save=True)
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
    # pass

def simulation_4(K_true=10, K=6, width=30, nsub_list=np.array([10,10]),
                 M=np.array([10,10],dtype=int), num_part=2, sigma2=0.1,
                 iter=100):
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
    res_plot = pd.DataFrame()
    for k_t in K_true:
        for k_fit in K:
            for i in range(iter):
                print(f'simulation K_true={k_t}, K_fit={k_fit}, iter {i}...')
                _, _, _, _, _, res = do_sim_diffK_fit(K_true=k_t, K=k_fit, M=M,
                                                         nsubj_list=nsub_list,
                                                         num_part=num_part,
                                                         width=width, low=0.1,
                                                         high=1.1, sigma2=sigma2,
                                                         plot_trueU=False)
                res['K_true'] = k_t
                res['K_fit'] = k_fit
                res['iter'] = i
                results = pd.concat([results, res], ignore_index=True)

            df = results.loc[(results['K_true']==k_t) & (results['K_fit']==k_fit)]
            common = df.loc[df['common_kappa'] == True]
            separate = df.loc[df['common_kappa'] == False]
            for_heatmap = pd.DataFrame(
                        {'ari_group_dif': [round(common.ari_group.mean() -
                                           separate.ari_group.mean(), 4)],
                         'ari_indiv_dif': [round(common.ari_indiv.mean() -
                                                 separate.ari_indiv.mean(), 4)],
                         'dcbc_group_dif': [round(common.dcbc_group.mean() -
                                                  separate.dcbc_group.mean(), 4)],
                         'dcbc_indiv_dif': [round(common.dcbc_indiv.mean() -
                                                  separate.dcbc_indiv.mean(), 4)],
                         'coserr_dif': [round(common.coserr.mean() - separate.coserr.mean(), 4)],
                         'wcoserr_dif': [round(common.wcoserr.mean() - separate.wcoserr.mean(), 4)],
                         'K_true': [k_t],
                         'K_fit': [k_fit]})
            res_plot = pd.concat([res_plot, for_heatmap], ignore_index=True)

    # 1. Plot evaluation results
    plt.figure(figsize=(18, 10))
    crits = ['ari_group','dcbc_group','coserr','ari_indiv','dcbc_indiv','wcoserr']
    for i, c in enumerate(crits):
        plt.subplot(2, 3, i + 1)
        result = res_plot.pivot(index='K_true', columns='K_fit', values=c+'_dif')
        sb.heatmap(result, annot=True, fmt='.2g')
        plt.title(c)

    plt.suptitle(f'Simulation 4, iter={iter}')
    plt.show()

def cal_gradient(Y):
    """Calculate the functional gradients from data
       for simulation

    Args:
        Y: raw data. shape (width, height, N)

    Returns:

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

if __name__ == '__main__':
    # original simulation by Jorn
    # nsub_list = np.array([10, 8])
    # do_simulation_sessFusion(5, nsub_list)
    # pass

    # 1. simulation - subject fusion
    # simulation_1(K=5, width=30, nsub_list=np.array([10,10]),
    #              M=np.array([40,20],dtype=int), num_part=1, sigma2=0.1)

    # 2. simulation - session fusion
    # for k in [10]:
    #     simulation_2(K=k, width=50, nsub_list=np.array([10, 10]),
    #                  M=np.array([40, 20], dtype=int), num_part=1, sigma2=0.5,
    #                  iter=10)

    # 3. simulation - session fusion (different Ks)
    # for k in [30]:
    #     simulation_3(K_true=k, K=5, width=50, nsub_list=np.array([10, 10]),
    #                  M=np.array([40, 20], dtype=int), num_part=1, sigma2=0.5,
    #                  iter=10)

    # 4. simulation - establish when underlying K >> fit K, kappa difference
    simulation_4(K_true=[5,10,20,30,40], K=[5,10,20,30,40], width=50,
                 nsub_list=np.array([10]),M=np.array([40], dtype=int),
                 num_part=1, sigma2=0.5, iter=10)

    # 3. Generate true individual maps with different parameters
    # test_Y = Y[0][0][0].T.view(30,30,-1)
    # grad = cal_gradient(test_Y)
    # plt.imshow(grad, cmap='jet')
    # plt.show()
    pass
