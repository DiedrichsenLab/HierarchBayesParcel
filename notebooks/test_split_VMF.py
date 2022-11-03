#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test what happens in estimation
when a VMF model is split in multiple sessions
"""
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import Functional_Fusion.matrix as matrix
# for testing and evaluating models
import evaluation as ev
import arrangements as ar 
import emissions as em

def make_joined(K=5,P=100,
                n_cond = 4,
                n_sess = 2,
                n_part = 2,
                n_subj = 10):
    N = n_cond * n_sess * n_part
    U=np.kron(np.ones(int(P/K),),np.arange(K))
    U=np.kron(np.ones((n_subj,1)),U.reshape(1,-1))
    sess_vec = np.kron(np.arange(n_sess),np.ones(n_part*n_cond,))
    # Different conditions across sessions 
    c = np.kron(np.arange(n_cond),np.ones(n_part,))
    cond_vec = np.kron(np.ones(n_sess,),c)
    cond_vec = cond_vec + sess_vec*n_cond 
    part_vec = np.kron(np.ones(n_sess * n_cond,),np.arange(n_part))

    # Build emission model
    X = matrix.indicator(cond_vec)
    emM = em.MixVMF(K=K,P=P,part_vec = part_vec, X=X,uniform_kappa=True)
    emM.kappa = 15
    Y = emM.sample(pt.tensor(U))
    return Y,U,sess_vec,cond_vec,part_vec,emM

def simulate_split(K=5,P=100,
                n_cond = 3,
                n_sess = 2,
                n_part = 2,
                n_subj = 10,
                uniform_kappa=True):
    Y,U,sess_vec,cond_vec,part_vec,emT = make_joined(K,P,n_cond,
                                            n_sess,n_part,n_subj)
    emM =[]
    # Build overall emission model
    X = matrix.indicator(cond_vec)
    emM.append(em.MixVMF(K=K,P=P,part_vec = part_vec, X=X,uniform_kappa=uniform_kappa))
    emM[0].initialize(Y)
    # Build emission model
    for s in range(2):
        X = matrix.indicator(cond_vec[sess_vec==s])
        emM.append(em.MixVMF(K=K,P=P,
            part_vec = part_vec[sess_vec==s], 
            X=X,
            uniform_kappa=uniform_kappa))
        emM[-1].initialize(Y[:,sess_vec==s,:])
    
    LL=[]
    for M in emM:
        M.Mstep(ar.expand_mn(U,K=K))
        LL.append(M.Estep())
    pass

    # Plot result of simulation 
    plt.figure(figsize=(18,5))
    plt.subplot(2,3,1)
    plt.imshow(emM[0].V,vmin=-0.7,vmax=0.7)

    plt.subplot(2,3,2)
    V1 = np.r_[emM[1].V,emM[2].V]
    V1  = V1 / np.sqrt(np.sum(V1**2,axis=0))
    plt.imshow(V1,vmin=-0.7,vmax=0.7)

    plt.subplot(2,3,3)
    V2 = np.r_[emM[1].V*emM[1].kappa,emM[2].V*emM[2].kappa]
    V2  = V2 / np.sqrt(np.sum(V2**2,axis=0))
    plt.imshow(V2,vmin=-0.7,vmax=0.7)

    plt.subplot(2,3,4)
    U1 = pt.softmax(LL[0][0,:,:10],dim=0)
    plt.imshow(U1,vmin=0,vmax=1)
    plt.subplot(2,3,5)
    U2 = pt.softmax(LL[1][0,:,:10] + LL[2][0,:,:10],dim=0)
    plt.imshow(U2,vmin=0,vmax=1)
    pass

if __name__ == '__main__':
    simulate_split(uniform_kappa=False)
    


    pass