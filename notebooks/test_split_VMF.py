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
import generativeMRF.evaluation as ev
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em

def sim_gaussian(M,N,kappa=5,mean=1):
    mu = pt.ones((M,1))*mean
    eps = pt.randn((M,N))/np.sqrt(kappa)
    Y = mu + eps
    Y = Y / pt.sqrt(pt.sum(Y ** 2, dim=0))
    y = pt.sum(Y,dim=1)/N
    r_norm = pt.sqrt(pt.sum(y ** 2, dim=0))
    V = y / r_norm
    kappa = (r_norm * M - r_norm**3) / (1 - r_norm**2)
    return kappa


def make_joined(K=5,P=100,
                n_cond = 4,
                n_sess = 2,
                n_part = 2,
                n_subj = 10,
                W = pt.tensor([0,0.1,0.5,0.8,1])):
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

    # Set V to something informative
    Vi = [] # List of individual V's
    for i in range(n_sess):
        Vi.append(pt.randint(0,2, (n_cond,K))*2-1)

    if W is None:
        V = pt.concat(Vi,dim=0)
    else:
        V = pt.concat([Vi[0]*W,Vi[1]*(1-W)],dim=0)
    V = V / pt.sqrt(pt.sum(V ** 2, dim=0))

    emM.V = V
    emM.kappa = 10
    Y = emM.sample(pt.tensor(U))
    return Y,U,sess_vec,cond_vec,part_vec,emM

def simulate_split(K=5,P=100,
                n_cond = 10,
                n_sess = 2,
                n_part = 2,
                n_subj = 100,
                W=None):
    Y,U,sess_vec,cond_vec,part_vec,emT = make_joined(K,P,n_cond,
                                            n_sess,n_part,n_subj,W)
    emM =[]
    # Build overall emission model
    X = matrix.indicator(cond_vec)
    emM.append(em.MixVMF(K=K,P=P,part_vec = part_vec, X=X,uniform_kappa=True))
    emM[0].initialize(Y)

    # Build emission model with Uniform / non-uniform kappa
    for uk in [True,False]:
        for s in range(n_sess):
            X = matrix.indicator(cond_vec[sess_vec==s])
            emM.append(em.MixVMF(K=K,P=P,
                part_vec = part_vec[sess_vec==s],
                X=X,
                uniform_kappa=uk))
            emM[-1].initialize(Y[:,sess_vec==s,:])
    LL=[]
    for M in emM:
        M.Mstep(ar.expand_mn(U,K=K))
        LL.append(M.Estep())
    pass

    # Plot result of simulation
    plt.figure(figsize=(18,5))


    plt.subplot(2,4,1)
    plt.imshow(emT.V,vmin=-0.7,vmax=0.7)
    plt.title(f'True k={emT.kappa:0.2f}')

    plt.subplot(2,4,2)
    plt.imshow(emM[0].V,vmin=-0.7,vmax=0.7)
    plt.title(f'Joint k={emM[0].kappa:0.2f}')

    plt.subplot(2,4,3)
    V2=[]
    for i in range(n_sess):
        V2.append(emM[i+1].V*emM[i+1].kappa)
    V2 = np.vstack(V2)
    V2  = V2 / np.sqrt(np.sum(V2**2,axis=0))
    plt.imshow(V2,vmin=-0.7,vmax=0.7)
    plt.title(f'k={emM[1].kappa:0.2f}\nk={emM[2].kappa:0.2f}')

    plt.subplot(2,4,4)
    V2=[]
    for i in range(n_sess):
        V2.append(emM[i+1+n_sess].V*emM[i+1+n_sess].kappa)
    V2 = np.vstack(V2)
    V2  = V2 / np.sqrt(np.sum(V2**2,axis=0))
    plt.imshow(V2,vmin=-0.7,vmax=0.7)

    str=''
    for i in range(n_sess):
        str += 'k='
        for k in range(K):
            str += f' {emM[n_sess+1+i].kappa[k]:0.1f}'
        str += '\n'
    plt.title(str)

    plt.subplot(2,4,6)
    U1 = pt.softmax(LL[0][0,:,:5],dim=0)
    plt.imshow(U1,vmin=0,vmax=1)

    plt.subplot(2,4,7)
    U2 = pt.softmax(LL[1][0,:,:5] + LL[2][0,:,:5],dim=0)
    plt.imshow(U2,vmin=0,vmax=1)

    plt.subplot(2,4,8)
    U2 = pt.softmax(LL[3][0,:,:5] + LL[4][0,:,:5],dim=0)
    plt.imshow(U2,vmin=0,vmax=1)
    pass

if __name__ == '__main__':
    # simulate_split(n_cond=500,n_sess=3)
    M=np.arange(31)+1
    kappa = np.zeros(M.shape)
    k=2.0
    m=1
    for i,m in enumerate(M):
        kappa[i] = sim_gaussian(m,400,kappa=k,mean=m)

    plt.plot(M,kappa,'r')
    plt.plot(M,k*M*(m**2),'k')


    pass