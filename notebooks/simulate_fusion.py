#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate datasets fusion

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
from copy import copy, deepcopy
import pandas as pd
import seaborn as sb

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           torch.FloatTensor)

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
    G = pt.zeros([map.max() + 1] * 2)
    # left-right pairs
    G[map[:, :-1], map[:, 1:]] = 1
    # right-left pairs
    G[map[:, 1:], map[:, :-1]] = 1
    # top-bottom pairs
    G[map[:-1, :], map[1:, :]] = 1
    # bottom-top pairs
    G[map[1:, :], map[:-1, :]] = 1

    labels = pt.where(G.sum(0) >= k)[0]
    base_label = labels[pt.randint(labels.shape[0], ())]
    tmp = pt.where(G[base_label] == 1)[0]
    tmp = tmp[tmp != base_label]
    neighbours = tmp[:k-1]

    return G, base_label, neighbours

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

    emissions =[]
    for i,m in enumerate(M):
        # Step 2: up the emission model and sample from it with a specific signal
        emissionT = em.MixGaussianExp(K=K, N=m, P=P, num_signal_bins=100, std_V=True)
        emissionT.sigma2 = pt.tensor(sigma2)
        # Initialize all V to be the high norm
        emissionT.V = emissionT.V * high_norm
        emissionT.num_subj=nsubj_list[i]
        emissions.append(emissionT)

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
    _, base, idx = _compute_adjacency(label_map, 3)

    # the parcels have same V magnitude in dataset1
    idx_1 = pt.cat((idx, base.view(1)))
    # the parcels have same V magnitude in dataset2
    idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
    idx_all = [idx_1, idx_2]
    print(base, idx_all)
    for i, par in enumerate(idx_all):
        # Making the V magnitude of bad parcels as low norm
        for j in range(0, len(par)):
            unit_vec = T.emissions[i].V[:, par[j]] / pt.sqrt(pt.sum(T.emissions[i].V[:, par[j]] **
                                                                   2, dim=0))
            T.emissions[i].V[:, par[j]] = unit_vec * low_norm

    # Sampling individual Us and data
    U, Y = T.sample()
    _, Y_test = T.sample(U=U)

    # Get the true signals
    signal = [None]*len(T.subj_ind)
    for i, us in enumerate(T.distribute_evidence(U)):
        signal[i] = pt.where(pt.isin(us,idx_all[i]), low_norm, high_norm)

    return T, Y, Y_test, U, signal

def make_true_model(grid, K=5, P=100,
                    nsubj_list=[10,10],
                    M=[5,5], # Number of conditions per data set
                    num_part=3,
                    theta_mu=150,
                    theta_w=20,
                    inits=None,
                    common_kappa=False,
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

    emissions =[]
    for i,m in enumerate(M):
        X = np.kron(np.ones((num_part,1)),np.eye(m))
        part_vec = np.kron(np.arange(num_part),np.ones((m,)))
        emission = em.wMixVMF(K=K, X=X, P=P, part_vec=part_vec,
                              uniform_kappa=common_kappa,
                              weighting='ones')
        emission.num_subj=nsubj_list[i]
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

    return T

def make_full_model(K=5,P=100,
                    nsubj_list=[10,10],
                    M=[5,5], # Number of conditions per data set
                    num_part=3,
                    common_kappa=False,
                    same_subj=False,
                    model_type='VMF'):
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
        elif model_type == 'wVMF_t2':
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

def do_simulation_sessFusion_sess(K=5, M=np.array([5],dtype=int),
                                  nsubj_list=None,
                                  num_part=3,
                                  width=10,
                                  low_kappa=3,
                                  high_kappa=30,
                                  arbitrary=None,
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
    pm = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    pm.random_smooth_pi(grid.Dist, theta_mu=20)
    arrangeT.theta_w = pt.tensor(20)

    T = make_full_model(K=K, P=grid.P,
                        nsubj_list=nsubj_list,
                        num_part=num_part,
                        M=M,
                        same_subj = True)
    T.arrange.logpi = pm.logpi

    if plot_trueU:
        grid.plot_maps(pt.argmax(pm.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        #plt.savefig('trueU.eps', format='eps')
        plt.show()
        plt.clf()
        grid.plot_maps(pt.exp(pm.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        #plt.savefig('true_prior.eps', format='eps')
        plt.show()

    # Initialize all kappas to be the high value
    for em in T.emissions:
        em.kappa = pt.full(em.kappa.shape, high_kappa)

    # Making ambiguous boundaries by set the same V_k for k-neighbouring parcels
    label_map = pt.argmax(T.arrange.logpi, dim=0).reshape(grid.dim)
    indices = []
    if arbitrary is None:
        # Default: 3 bad parcels in session1 - (K-3) in session2
        _, base, idx = _compute_adjacency(label_map, 3)
        # the parcels have same V in session1
        idx_1 = pt.cat((idx, base.view(1)))
        # the parcels have same V in session2
        idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
        indices.append(idx_1)
        indices.append(idx_2)
    else:
        # bad parcels per session depends on arbitrary [bad1, bad2]
        for num_p in arbitrary:
            _, base, idx = _compute_adjacency(label_map, num_p)
            # the parcels have same V in session1
            idx = pt.cat((idx, base.view(1)))
            indices.append(idx)

    print(indices)

    for i, par in enumerate(indices):
        # Making the V align to the first parcel
        for j in range(1, len(par)):
            T.emissions[i].V[:, par[j]] = T.emissions[i].V[:, par[0]]

        # Set kappas of k-neighbouring parcels to low_kappa
        for k in range(len(par)):
            if not T.emissions[i].uniform_kappa:
                T.emissions[i].kappa[par[k]] = low_kappa
            else:
                raise ValueError("The kappas of emission models need to be separate in simulation")

    # Sampling individual Us and data
    U,Y = T.sample()

    Uerrors, U_indv, Props = [], [], []
    for common_kappa in [True, False]:
        models = []
        all_ses = [x for x in np.arange(len(M))]
        em_indx = [all_ses] + [[x] for x in np.arange(len(M))] + [all_ses]

        # Initialize multiple full models: dataset1, dataset2,..., dataset1 to N
        for j in range(len(em_indx)-1):
            models.append(make_full_model(K=K,P=grid.P,
                                          M=M[em_indx[j+1]],
                                          nsubj_list=nsubj_list[em_indx[j+1]],
                                          num_part=num_part,
                                          common_kappa=common_kappa,
                                          same_subj=True))
            data = [Y[i] for i in em_indx[j+1]]
            this_sub_list = [np.arange(x) for x in nsubj_list[em_indx[j+1]]]
            models[j].initialize(data, subj_ind=this_sub_list)

        # Fitting the full models
        U_fit = []
        for i,m in enumerate(models):
            models[i],ll,theta,Uhat,first_ll = \
                models[i].fit_em_ninits(n_inits=40, first_iter=7, iter=100, tol=0.01,
                fit_emission=True, fit_arrangement=True,
                init_emission=True, init_arrangement=True,
                align = 'arrange')
            U_fit.append(Uhat)

        # Align full models to the true
        MM = [T]+models
        Prop = ev.align_models(MM)
        Props.append(Prop)

        # Option 1: Using matching_U
        U_recon_1, uerr_1 = ev.matching_U(U, U_fit[0])
        U_recon_2, uerr_2 = ev.matching_U(U, U_fit[1])
        U_recon, uerr = ev.matching_U(U, U_fit[2])
        # TODO: Option 2: Using matching greedy

        U_indv.append([U_recon_1, U_recon_2, U_recon])
        Uerrors.append([uerr_1, uerr_2, uerr])

        # Printing kappa fitting
        Kappa = np.zeros((2,4,K))
        for j,ei in enumerate(em_indx):
            for k,i in enumerate(ei):
                Kappa[i,j,:]=MM[j].emissions[k].kappa
        print(Kappa.round(1))

    return grid, U, U_indv, Uerrors, Props

def do_simulation_sessFusion_subj(K=5, M=np.array([5,5],dtype=int), nsubj_list=None,
                                  num_part=3, width=10, low_norm=3, high_norm=30,
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

    # T = make_true_model(grid, K=K, P=grid.P, nsubj_list=nsubj_list, M=M,
    #                     theta_mu=60, theta_w=2, inits=np.array([820,443,188,305,717]),
    #                     num_part=num_part)

    T, Y, Y_test, U, signal = make_true_model_GME(grid, K=K, P=grid.P, nsubj_list=nsubj_list,
                                                  M=M, theta_mu=20, theta_w=1.3, sigma2=sigma2,
                                                  inits=None)

    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.show()

    # Visualizing sampled data
    # data_visualize(Y, U)

    # Basic setting for fitting/evauation
    fitting_model = ['D1', 'D2', 'D_fusion']
    Us = [U[T.subj_ind[0], :], U[T.subj_ind[1], :], U]
    Ys = [[Y[0]], [Y[1]], Y]
    Ys_test = [[Y_test[0]], [Y_test[1]], Y_test]
    signals = [[signal[0]], [signal[1]], signal]
    sub_indices = [T.subj_ind[0], T.subj_ind[1], pt.hstack(T.subj_ind)]
    U_indv, Props, kappas = [], [], []
    results = pd.DataFrame()

    # Main loop
    for w in ['VMF', 'wVMF_l2']:
        for common_kappa in [True, False]:
            models = []
            em_indx = [[0, 1], [0], [1], [0, 1]]
            # Initialize three full models: dataset1, dataset2, dataset1 and 2
            for j, fm_name in enumerate(fitting_model):
                data = [Y[i] for i in em_indx[j + 1]]
                tdata = [Y_test[i] for i in em_indx[j + 1]]
                models.append(make_full_model(K=K,P=grid.P,M=M[em_indx[j+1]],num_part=num_part,
                                              nsubj_list=nsubj_list[em_indx[j+1]],
                                              common_kappa=common_kappa, model_type=w))
                if w == 'wVMF_t2':
                    for i, emi in enumerate(models[j].emissions):
                        emi.initialize(Ys[j][i], weight=signals[j][i]**2)
                else:
                    models[j].initialize(Ys[j])

                # Fitting the full models
                models[j],ll,theta,Uhat,first_ll = \
                    models[j].fit_em_ninits(n_inits=40, first_iter=10, iter=100, tol=0.01,
                    fit_emission=True, fit_arrangement=True,
                    init_emission=True, init_arrangement=True,
                    align = 'arrange')
                # # Use fit_em
                # models[j],ll,theta,Uhat = models[j].fit_em(iter=100, tol=0.01,
                #                                            fit_emission=True,
                #                                            fit_arrangement=True,
                #                                            first_evidence=False)

            # Align full models to the true, and store U and U_indv
            MM = [T]+models
            Prop = ev.align_models(MM, in_place=True)
            Props.append(Prop)
            UV_hard = [pt.argmax(e.Estep()[0], dim=1) for e in MM[1:]]
            UV_soft = [e.Estep()[0] for e in MM[1:]]
            U_indv.append(UV_hard)

            # evaluation starts after model alignment
            models = MM[1:]
            for j, fm_name in enumerate(fitting_model):
                # 1. Hard U reconstruction error
                uerr_hard = u_err(Us[j], UV_hard[j])
                # 2. Soft U reconstruction error
                uerr_soft = ev.u_abserr(ar.expand_mn(Us[j], K), UV_soft[j])
                # 3. non-adjusted/adjusted expected cosine error
                coserr, wcoserr = [], []
                for i, emi in enumerate(models[j].emissions):
                    coserr.append(ev.coserr(Ys_test[j][i], emi.V,
                                            UV_soft[j][models[j].subj_ind[i]],
                                            adjusted=False, soft_assign=True))
                    wcoserr.append(ev.coserr(Ys_test[j][i], emi.V,
                                             UV_soft[j][models[j].subj_ind[i]],
                                             adjusted=True, soft_assign=True))

                res = pd.DataFrame({'model_type': [f'{w}_ck={common_kappa}'],
                                    'dataset': [fm_name],
                                    'uerr_hard': [uerr_hard.mean().item()],
                                    'uerr_soft': [uerr_soft],
                                    'coserr': [pt.stack(coserr).mean().item()],
                                    'wcoserr': [pt.stack(wcoserr).mean().item()]})
                results = pd.concat([results, res], ignore_index=True)

            # Visualizing fitted results (checking results)
            # data_visualize([Y, [Y[0]], [Y[1]], Y], [U]+this_UV)

            # Printing kappa fitting
            Kappa = pt.zeros((2,3,K))
            MM = MM[1:]
            for j,ei in enumerate(em_indx[1:]):
                for k,i in enumerate(ei):
                    Kappa[i,j,:]=MM[j].emissions[k].kappa
            # print(Kappa)
            kappas.append(Kappa)

    return grid, U, U_indv, Props, pt.stack(kappas), results

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
    for i in range(2):
        print(f'simulation {i}...')
        grid, U, U_indv, Props, kappas, res = do_simulation_sessFusion_subj(K=K, M=M,
                                                                        nsubj_list=nsub_list,
                                                                        num_part=num_part,
                                                                        width=width,
                                                                        low_norm=0.1,
                                                                        high_norm=0.9,
                                                                        sigma2=sigma2,
                                                                        plot_trueU=True)

        res['iter'] = i
        results = pd.concat([results, res], ignore_index=True)

    sb.barplot(x='model_type', y='uerr_soft', hue='dataset', data=results)
    plt.show()
    plot_result(grid, Props, save=True)

    # Plot all true individual maps
    _plot_maps(U, cmap='tab20', dim=(width, width), row_labels='True', save=True)

    # Plot fitted and aligned individual maps in dataset1, 2 and fusion
    labels = ["commonKappa_VMF", "separateKappa_VMF",
              "commonKappa_wVMF", "separateKappa_wVMF"]
    names = ["Dataset 1", "Dataset 2", "Dataset 1+2"]
    for i, us in enumerate(U_indv):
        for j in range(len(us)):
            _plot_maps(us[j], cmap='tab20', dim=(width, width),
                       row_labels=labels[i]+names[j], save=True)

    pass

def simulation_2(K=6, width=30,
                 nsub_list=np.array([10,10]),
                 M=np.array([10,10],dtype=int),
                 num_part=2):
    """Simulation of common kappa vs. separate kappas across subjects
    Args:
        width: The width and height of MRF grid
        nsub_list: the list of number of subject per dataset (emission model)
        M: The number of conditions per dataset (emission model)
        num_part: the number of partitions (sessions)
    Returns:
        Simulation result plots
    """
    grid, U, U_indv, Uerrors, Props = do_simulation_sessFusion_sess(K=K, M=M,
                                                                    nsubj_list=nsub_list,
                                                                    num_part=num_part,
                                                                    width=width,
                                                                    low_kappa=3,
                                                                    high_kappa=30,
                                                                    plot_trueU=True)
    plot_uerr(Uerrors, names=['Sess 1', 'Sess 2', 'Fusion'],
              save=True)
    plot_result(grid, Props, names = ["True", "Sess 1", "Sess 2", "Fusion"],
                save=True)

    # Plot all true individual maps
    _plot_maps(U, cmap='tab20', dim=(width, width), row_labels='True', save=True)

    # Plot fitted and aligned individual maps in dataset1, 2 and fusion
    labels = ["commonKappa_", "separateKappa_"]
    names = ['Sess 1', 'Sess 2', 'Fusion']
    for i, us in enumerate(U_indv):
        for j in range(len(us)):
            _plot_maps(us[j], cmap='tab20', dim=(width, width),
                       row_labels=labels[i]+names[j], save=True)

    pass

def sample_Us(K=5, M=np.array([5,5],dtype=int), nsubj_list=None,
              num_part=3, width=10, theta_mu=[150], theta_w=[10],
              low_kappa=3, high_kappa=30, plot_trueU=False):
    #Generate grid for easier visualization
    grid = sp.SpatialGrid(width=width, height=width)
    # centroids = np.random.choice(grid.P, (K,))
    centroids = np.array([820, 443, 188, 305, 717])
    # T = make_true_model(grid, K=K, P=grid.P, nsubj_list=nsubj_list,
    #                     M=M, inits=centroids, num_part=num_part)
    T = make_true_model_GME(grid, K=K, P=grid.P, nsubj_list=nsubj_list,
                            M=M, inits=centroids)

    if plot_trueU:
        grid.plot_maps(pt.argmax(T.arrange.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.savefig('trueU.pdf', format='pdf')
        plt.show()
        plt.clf()
        grid.plot_maps(pt.exp(T.arrange.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.savefig('true_prior.pdf', format='pdf')
        plt.show()

    # Initialize all kappas to be the high value
    for em in T.emissions:
        em.kappa = pt.full(em.kappa.shape, high_kappa)

    # Step 2: Initialize the parameters of the true model
    U_all, Y_all = [], []
    for mu in theta_mu:
        T.arrange.random_smooth_pi(grid.Dist, theta_mu=mu, centroids=centroids)
        for w in theta_w:
            T.arrange.theta_w = pt.tensor(w)

            # Making ambiguous boundaries by set the same V_k for k-neighbouring parcels
            label_map = pt.argmax(T.arrange.logpi, dim=0).reshape(grid.dim)
            _, base, idx = _compute_adjacency(label_map, 3)

            # the parcels have same V in dataset1
            idx_1 = pt.cat((idx, base.view(1)))
            # the parcels have same V in dataset2
            idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
            print(base, idx_1, idx_2)
            for i, par in enumerate([[1,2,3]]):
                # Making the V align to the first parcel
                for j in range(1, len(par)):
                    T.emissions[i].V[:, par[j]] = T.emissions[i].V[:, par[0]]

                # Set kappas of k-neighbouring parcels to low_kappa
                for k in range(len(par)):
                    if not T.emissions[i].uniform_kappa:
                        T.emissions[i].kappa[par[k]] = low_kappa
                    else:
                        raise ValueError("The kappas of emission models need to"
                            " be separate in simulation")

            # Sampling individual Us and data
            U, Y = T.sample()
            U_all.append(U)
            Y_all.append(Y)
            # Plot all true individual maps
            _plot_maps(U, cmap='tab20', dim=(width, width), row_labels=f'smooth{mu}_conn{w}',
                       save=True)

    return U_all, Y_all

def cal_gradient(Y):
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
    # nsub_list = np.array([10, 8])
    # do_simulation_sessFusion(5, nsub_list)
    # pass

    # 1. simulate across subjects
    simulation_1(K=6, width=30, nsub_list=np.array([5,5]),
                 M=np.array([40,40],dtype=int), num_part=1, sigma2=0.1)

    # 2. simulate across sessions in same set of subjects
    # simulation_2()

    # 3. Generate true individual maps with different parameters
    # U,Y = sample_Us(K=5, M=np.array([5],dtype=int), nsubj_list=[5],
    #           num_part=2, width=30, theta_mu=[20], theta_w=[1],
    #           low_kappa=3, high_kappa=30, plot_trueU=True)
    # # select smooth=60, w=2 for visualizing
    # test_Y = Y[0][0][0].T.view(30,30,-1)
    # grad = cal_gradient(test_Y)
    # plt.imshow(grad, cmap='jet')
    # plt.show()
    pass