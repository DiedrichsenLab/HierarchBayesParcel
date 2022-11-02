#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explores the fusion of different sessions for the same subjects and the fusion of different subjects
Especially it  script of simulating the generative model training when across dataset,
and test the model recovery ability.

Author: Joern Diedrichsen
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
                    common_kappa=False):
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

def do_simulation_sessFusion_dz(K=5, M=np.array([5,5],dtype=int), nsubj_list=None,
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

    if plot_trueU:
        grid.plot_maps(pt.argmax(pm.logpi, dim=0), cmap='tab20', vmax=19, grid=[1, 1])
        plt.show()
        grid.plot_maps(pt.exp(pm.logpi), cmap='jet', vmax=1, grid=(1, K), offset=1)
        plt.show()

    T = make_full_model(K=K, P=grid.P, nsubj_list=nsubj_list, M=M)
    T.arrange.logpi = pm.logpi

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

        # TODO: Calculate U reconstruction error


        # Plotting results
        names = ["True", "Dataset 1", "Dataset 2", "Dataset 1+2"]
        for i in range(len(MM)):
            plt.subplot(2,2,i+1)
            parcel = np.argmax(Prop[i,:,:],axis=0)
            grid.plot_maps(parcel,vmax=5)
            plt.title(names[i])
        plt.show()

        # Printing kappa fitting
        Kappa = np.zeros((2,4,K))
        for j,ei in enumerate(em_indx):
            for k,i in enumerate(ei):
                Kappa[i,j,:]=MM[j].emissions[k].kappa
        print(Kappa.round(1))

    pass

if __name__ == '__main__':
    # nsub_list = np.array([10, 8])
    # do_simulation_sessFusion(5, nsub_list)
    # pass

    nsub_list = np.array([12,8])
    M = np.array([10,10],dtype=int)
    do_simulation_sessFusion_dz(K=6, M=M, nsubj_list=nsub_list, width=30, low_kappa=3,
                                high_kappa=30, plot_trueU=True)
    pass
