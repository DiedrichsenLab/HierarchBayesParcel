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

def make_full_model(K=5,P=100,
                    nsubj_list=[10,10],
                    M=[5,5], # Number of conditions per data set
                    num_part=3):
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
                    uniform_kappa=False)
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


if __name__ == '__main__':
    nsub_list = np.array([10,8])
    do_simulation_sessFusion(5,nsub_list)
    pass
