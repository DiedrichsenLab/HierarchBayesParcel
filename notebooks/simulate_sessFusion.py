#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explores the fusion of different sessions for the same subjects and the fusion of different subjects
Especially it  script of simulating the generative model training when across dataset,
and test the model recovery ability.

Author: Joern Diedrichsen, dzhi
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

def make_full_model(K=5,P=100,
                    nsubj_list=[10,10],
                    M=[5,5], # Number of conditions per data set
                    num_part=3,
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
    A = ar.ArrangeIndependent(K,P,spatial_specific=True,remove_redundancy=False)
    A.random_params()
    emissions =[]
    for i,m in enumerate(M):
        X = np.kron(np.ones((num_part,1)),np.eye(m))
        part_vec = np.kron(np.arange(num_part),np.ones((m,)))
        emission = em.MixVMF(K=K,
                    X=X,
                    P=P,
                    part_vec=part_vec,
                    uniform_kappa=common_kappa)
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
              plot=True, save=False):
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
    fig, axs = plt.subplots(1, len(D), figsize=(6*len(D), 6))

    for i, data in enumerate(D):
        A, B, C = data[0], data[1], data[2]
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
        # plt.ylim(0.6, 0.7)

    fig.suptitle('Simulation common kappa vs. separate kappas')
    plt.tight_layout()

    if save:
        plt.savefig('Uerr.eps', format='eps')
    if plot:
        plt.show()
        plt.clf()
    else:
        pass

def plot_result(grid, MM, ylabels=["common Kappa", "separate Kappas"],
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
            parcel = np.argmax(m[j, :, :], axis=0)
            grid.plot_maps(parcel, vmax=5)
            if i == 0:
                plt.title(names[j])
            if j == 0:
                plt.ylabel(ylabels[i])

    if save:
        plt.savefig('group_results.eps', format='eps')
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
        ax.imshow(U[n].reshape(dim), cmap='tab20', interpolation='nearest', vmax=vmax)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if (row_labels is not None) and (n % N == 0):
            ax.axes.yaxis.set_visible(True)
            ax.set_yticks([])
            # ax.set_ylabel(row_labels[int(n / num_sub)])

    if save:
        plt.savefig(f'{row_labels}.eps', format='eps')
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
                                width=10, low_kappa=3, high_kappa=30, plot_trueU=False):
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
    pm = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    pm.random_smooth_pi(grid.Dist, theta_mu=20)

    T = make_full_model(K=K, P=grid.P, nsubj_list=nsubj_list, M=M)
    T.arrange.logpi = pm.logpi

    if plot_trueU:
        grid.plot_maps(pt.argmax(pm.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.savefig('trueU.eps', format='eps')
        plt.show()
        plt.clf()
        grid.plot_maps(pt.exp(pm.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.savefig('true_prior.eps', format='eps')
        plt.show()

    # Initialize all kappas to be the high value
    for em in T.emissions:
        em.kappa = pt.full(em.kappa.shape, high_kappa)

    # Making ambiguous boundaries by set the same V_k for k-neighbouring parcels
    label_map = pt.argmax(T.arrange.logpi, dim=0).reshape(grid.dim)
    _, base, idx = _compute_adjacency(label_map, 3)

    # the parcels have same V in dataset1
    idx_1 = pt.cat((idx, base.view(1)))
    # the parcels have same V in dataset2
    idx_2 = pt.tensor([i for i in label_map.unique() if i not in idx_1])
    print(base, idx_1, idx_2)
    for i, par in enumerate([idx_1, idx_2]):
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

    Uerrors, U_indv, figs, Props = [], [], [], []
    for common_kappa in [True, False]:
        models = []
        em_indx = [[0, 1], [0], [1], [0, 1]]
        # Initialize three full models: dataset1, dataset2, dataset1 and 2
        for j in range(3):
            models.append(make_full_model(K=K,P=grid.P,M=M[em_indx[j+1]],
                                          nsubj_list=nsubj_list[em_indx[j+1]],
                                          common_kappa=common_kappa))
            data = [Y[i] for i in em_indx[j+1]]
            models[j].initialize(data)

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
        figs.append(U_fit)
        Props.append(Prop)

        # Calculate and plot U reconstruction error
        U1 = U[T.subj_ind[0],:]
        U2 = U[T.subj_ind[1],:]

        # Option 1: Using matching_U
        U_recon_1, uerr_1 = ev.matching_U(U1, U_fit[0])
        U_recon_2, uerr_2 = ev.matching_U(U2, U_fit[1])
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

def simulation_1(width=30,
                 nsub_list=np.array([12,8]),
                 M=np.array([10,10],dtype=int)):
    """Simulation of common kappa vs. separate kappas across subjects
    Args:
        width: The width and height of MRF grid
        nsub_list: the list of number of subject per dataset (emission model)
        M: The number of conditions per dataset (emission model)
    Returns:
        Simulation result plots
    """
    grid, U, U_indv, Uerrors, Props = do_simulation_sessFusion_subj(K=5, M=M,
                                                                    nsubj_list=nsub_list,
                                                                    width=width,
                                                                    low_kappa=3,
                                                                    high_kappa=30,
                                                                    plot_trueU=True)
    plot_uerr(Uerrors, save=True)
    plot_result(grid, Props, save=True)

    # Plot all true individual maps
    _plot_maps(U, cmap='tab20', dim=(width, width), row_labels='True', save=True)

    # Plot fitted and aligned individual maps in dataset1, 2 and fusion
    labels = ["commonKappa_", "separateKappa_"]
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

if __name__ == '__main__':
    # nsub_list = np.array([10, 8])
    # do_simulation_sessFusion(5, nsub_list)
    # pass

    # 1. simulate across subjects
    # simulation_1()

    # 2. simulate across sessions in same set of subjects
    simulation_2()
    pass
