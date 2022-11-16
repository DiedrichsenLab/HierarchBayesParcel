#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test weigted VMF model

"""
# general import packages
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sb

# for testing and evaluating models
from full_model import FullMultiModel
from arrangements import ArrangeIndependent, expand_mn, sample_multinomial
from emissions import MixGaussianExp, MixVMF, wMixVMF
import evaluation as ev

def simulate_from_GME(K=5, P=100, N=40, num_sub=10,
                sigma2 = 1,
                signal_distrib='discrete',
                signal_param=[0.5,0.4,0.1]):
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
    # Step 1: Set the arrangement model and sample from it 
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
    arrangeT.random_params()
    U = arrangeT.sample(num_subj=num_sub)

    # Step 2: up the emission model and sample from it with a specific signal 
    emissionT = MixGaussianExp(K=K, N=N, P=P, num_signal_bins=100, std_V=True)
    emissionT.sigma2 = pt.tensor(sigma2)
    
    if signal_distrib=='discrete':
        num_cat = len(signal_param)
        pi = pt.tensor(signal_param).reshape(-1,1).expand(-1,P)
        signal = sample_multinomial(pi,
                            shape=(num_sub,num_cat,P),compress=True)
        Y, signal = emissionT.sample(U, signal,return_signal=True)
    elif signal_distrib=='exp':
        emissionT.beta = pt.tensor(signal_param)
        Y, signal = emissionT.sample(U, return_signal=True)
    else:
        raise(NameError('Unknown signal distribution'))

    return Y,U,signal,arrangeT,emissionT


def build_model(model_type,Y,signal,arrangeT,uniform_kappa=True):
    """Builds and initilizes the full model used for fitting 

    Args:
        model_type (_type_): _description_
        Y (_type_): _description_
        signal (_type_): _description_
        arrangeT (_type_): _description_
        uniform_kappa (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    num_sub,N,P = Y.shape
    if model_type == 'VMF':
        emissionM = MixVMF(K=arrangeT.K, N = N, P=arrangeT.P, X=None, uniform_kappa=uniform_kappa)
        emissionM.initialize(Y)
    elif model_type == 'wVMF_t2':
        emissionM = wMixVMF(K=arrangeT.K, N = N, P=arrangeT.P, X=None, uniform_kappa=uniform_kappa)
        emissionM.initialize(Y,weight=signal**2)
    else: 
        raise(NameError('Unknown model type'))
    emissionM.random_params()
    fm = FullMultiModel(arrangeT,[emissionM])
    fm.initialize()
    return fm

def do_sim(K=5,N=10,num_sim=10): 
    # different types of fitting models 
    model_type = ['VMF','wVMF_t2'] # ,'wVMF_y2','wVMF_y']

    # Generate training data 
    results = pd.DataFrame()
    for n in range(num_sim):
        print(f'simulation {n}')
        Y,U,signal,arrangeT,emissionT = simulate_from_GME(K=K,
                    N=N,
                    num_sub=10,
                    sigma2=0.3)

        # Get independent test data
        Y_test = emissionT.sample(U,signal=signal)

        for mt in model_type:
            fm = build_model(mt,Y,signal,arrangeT)
            fm, ll, theta, U_hat = fm.fit_em(iter=100, tol=0.0001,
                                    fit_arrangement=False,
                                    first_evidence=False)
            uerr = ev.u_abserr(expand_mn(U,K), U_hat)
            coserr = ev.coserr(Y_test, 
                                fm.emissions[0].V,
                                U_hat,
                                adjusted=False, 
                                soft_assign=True)
            wcoserr = ev.coserr(Y_test, 
                                fm.emissions[0].V,
                                U_hat,
                                adjusted=True, 
                                soft_assign=True)
            res=pd.DataFrame({'model_type':[mt],
                             'uerr':[uerr],
                             'coserr':[coserr.mean().item()],
                             'wcoserr':[wcoserr.mean().item()]})
            results  = pd.concat([results,res],ignore_index=True)
    return results

def plot_results(results):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    sb.barplot(data=results,x='model_type',y='uerr')
    plt.subplot(1,3,2)
    sb.barplot(data=results,x='model_type',y='coserr')
    plt.subplot(1,3,3)
    sb.barplot(data=results,x='model_type',y='wcoserr')

if __name__ == '__main__':
    # simulate_split(n_cond=500,n_sess=3)
    results = do_sim()
    plot_results(results)
    pass