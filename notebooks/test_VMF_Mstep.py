#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests different M-step methods for the VMF distribution
In this script, we follow a very simple example where there is only one possible parcel
and we can estimate V and kappa directly in one M-step.
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
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar

class MixVMF1(em.MixVMF):
    """Original version of the van Mises-Fisher Mixture Model"""
    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are Vs (unit norm projected on
            the N-1 sphere) and kappa (concentration value).

        Args:
            U_hat: the expected log likelihood from the arrangement model

        Returns:
            Update all the object's parameters
        """
        if type(U_hat) is np.ndarray:
            U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

        # JU is the number of observations (voxels x num_part) in each subject
        JU = pt.sum(self.num_part * U_hat,dim=2)   # (num_sub, K)

        # Calculate YU = \sum_i\sum_k<u_i^k>y_i # (num_sub, N, K)
        YU = pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat, 1, 2))

        # 1. Updating the V_k
        if 'V' in self.param_list:
            if self.subjects_equal_weight:
                self.V=pt.nanmean(YU / JU.unsqueeze(1),dim=0)
            else:
                self.V = pt.sum(YU,dim=0) / JU.sum(dim=0, keepdim=True)
            v_norm = pt.sqrt(pt.sum(self.V**2, dim=0))
            v_norm[v_norm == 0] = pt.nan # Avoid division by zero
            self.V = self.V / v_norm

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2),
        # where r_bar = ||V_k||/N*Uhat
        if 'kappa' in self.param_list:
            if (not self.subject_specific_kappa) and (not self.parcel_specific_kappa):
                yu = pt.sum(YU,dim=0)
                r_bar =pt.sum(pt.sqrt(pt.sum(yu**2, dim=0)))/ pt.sum(JU)
            elif self.parcel_specific_kappa and (not self.subject_specific_kappa):
                yu = pt.sum(YU,dim=0)
                r_bar=pt.sqrt(pt.sum(yu**2, dim=0))/ pt.sum(JU,dim=0)
            elif self.subject_specific_kappa and (not self.parcel_specific_kappa):
                r_bar = pt.sum(pt.sqrt(pt.sum(YU**2, dim=1)),dim=1)/pt.sum(JU,dim=1)
            else:
                r_bar=pt.sqrt(pt.sum(YU**2, dim=1))/ JU
            r_bar[r_bar > 0.99] = 0.99
            self.kappa = (r_bar * self.M - r_bar**3) / (1 - r_bar**2)

class MixVMF2(em.MixVMF):
    """Newer version of the van Mises-Fisher Mixture Model, with use of estimated V-estimation"""
    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are Vs (unit norm projected on
            the N-1 sphere) and kappa (concentration value).

        Args:
            U_hat: the expected log likelihood from the arrangement model

        Returns:
            Update all the object's parameters
        """
        if type(U_hat) is np.ndarray:
            U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

        # JU is the number of observations (voxels x num_part) in each subject
        JU = pt.sum(self.num_part * U_hat,dim=2)   # (num_sub, K)

        # Calculate YU = \sum_i\sum_k<u_i^k>y_i # (num_sub, N, K)
        YU = pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat, 1, 2))

        # 1. Updating the V_k
        if 'V' in self.param_list:
            if self.subjects_equal_weight:
                self.V=pt.nanmean(YU / JU.unsqueeze(1),dim=0)
            else:
                self.V = pt.sum(YU,dim=0) / JU.sum(dim=0, keepdim=True)
            v_norm = pt.sqrt(pt.sum(self.V**2, dim=0))
            v_norm[v_norm == 0] = pt.nan # Avoid division by zero
            self.V = self.V / v_norm

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2),
        # where r_bar = ||V_k||/N*Uhat
        if 'kappa' in self.param_list:
            if (not self.subject_specific_kappa) and (not self.parcel_specific_kappa):
                v = pt.sum(YU, dim=0) / pt.sum(JU,dim=0)
                r_bar =pt.mean(pt.sum(v*self.V, dim=0))
            elif self.parcel_specific_kappa and (not self.subject_specific_kappa):
                yu = pt.sum(YU,dim=0)
                r_bar=pt.sqrt(pt.sum(yu**2, dim=0))/ pt.sum(JU,dim=0)
            elif self.subject_specific_kappa and (not self.parcel_specific_kappa):
                v = YU/JU.unsqueeze(1)
                r_bar = pt.mean(pt.sum(v*self.V,dim=1),dim=1)
            else:
                r_bar=pt.sqrt(pt.sum(YU**2, dim=1))/ JU
            r_bar[r_bar > 0.99] = 0.99
            self.kappa = (r_bar * self.M - r_bar**3) / (1 - r_bar**2)


def make_dataset(K=1, P=20,
                 num_cond=6,
                 num_part=10,
                 num_subj=8,
                 sig_subj = 0,
                 cov_voxel=0.1,
                 kappa=[10]*8,
                 V=None):
    """Generates a true dataset for the VMF model"""

    cond_vec=np.kron(np.ones(num_part),np.arange(num_cond))
    part_vec=np.kron(np.arange(num_part),np.ones(num_cond))
    X = np.kron(np.ones((num_part,1)),np.eye(num_cond))
    em_true = em.MixVMF(K=K, P=P, part_vec =part_vec,X=X, num_subj=1)
    u = pt.randint(low=0,high=K,size=(num_subj,P))
    U = ar.expand_mn(u,K)
    em_true.random_params()
    Y = pt.zeros((num_subj,num_cond*num_part,P))

    # generate V-matrices for each subject
    V = np.random.normal(0,sig_subj,(num_subj,em_true.M,em_true.K))  + em_true.V.numpy()
    V = V / np.sqrt(np.sum(V**2,axis=1,keepdims=True))

    for s in range(U.shape[0]):
        em_true.V = pt.tensor(V[s])
        em_true.kappa= pt.tensor(kappa[s])
        Y[s]=em_true.sample(u[s:s+1,:])[0]
    return Y,em_true,U,part_vec,X

def fit_emissions(emissions,Y,U,true_kappa=None):
    """Fits the emissions to the data and returns the kappa parameters for each subject"""
    D=[]
    n_subj =Y.shape[0]
    for i,em in enumerate(emissions):
        em.initialize(Y)
        em.Mstep(U)
        D.append(pd.DataFrame({'model':[i]*n_subj,
                          'true_kappa':true_kappa,
                          'subj':np.arange(n_subj),
                          'kappa':em.kappa.numpy()}))

    D = pd.concat(D)
    return D

def sim_basic():
    """ Checks if the M-step is working correctly for basic data"""
    P = 100
    K = 1
    num_sim=10
    kappa = np.array([0,1,2,3,4,5,6,7.0])
    num_subj = len(kappa)
    D=[]
    for n in range(num_sim):
        Y,em_true,U,part_vec,X = make_dataset(K =1, P=P,kappa=kappa)
        em1 = MixVMF1(K=K,P=P,part_vec=part_vec,X=X,num_subj=num_subj)
        em2 = MixVMF2(K=K,P=P,part_vec=part_vec,X=X,subject_specific_kappa=True,num_subj=num_subj)
        D.append(fit_emissions([em1,em2],Y,U,true_kappa=kappa))
    D = pd.concat(D)
    return D

def sim_voxel_dependence(P = 5,
                        P_factor = 20,
                        K = 1,
                        num_sim=10,
                        num_subj=8,
                        num_part = 2,
                        kappa= 3):
    kappa = np.array([kappa]*num_subj)
    D=[]
    for n in range(num_sim):
        Y,em_true,U,part_vec,X = make_dataset(K =1, P=P,kappa=kappa,num_part=num_part)
        Y = pt.tile(Y,(1,1,P_factor))
        U = pt.tile(U,(1,1,P_factor))
        em1 = MixVMF1(K=K,P=P*P_factor,part_vec=part_vec,X=X,num_subj=num_subj)
        em2 = MixVMF1(K=K,P=P*P_factor,part_vec=part_vec,X=X,subject_specific_kappa=True,num_subj=num_subj)
        em3 = MixVMF2(K=K,P=P*P_factor,part_vec=part_vec,X=X,num_subj=num_subj)
        em4 = MixVMF2(K=K,P=P*P_factor,part_vec=part_vec,X=X,subject_specific_kappa=True,num_subj=num_subj)
        D.append(fit_emissions([em1,em2,em3,em4],Y,U,true_kappa=kappa))
    D = pd.concat(D)
    return D

def sim_subject_differences():
    """ Checks if the M-step is working correctly for basic data"""
    P = 100
    K = 1
    num_sim=5
    num_subj=8
    kappa = np.array([5]*num_subj)
    D=[]
    for sig_s in [0,0.3]:
        for n in range(num_sim):
            Y,em_true,U,part_vec,X = make_dataset(K =1, P=P,kappa=kappa,sig_subj=sig_s)
            em1 = MixVMF1(K=K,P=P,part_vec=part_vec,X=X,num_subj=num_subj)
            em2 = MixVMF1(K=K,P=P,part_vec=part_vec,X=X,subject_specific_kappa=True,num_subj=num_subj)
            em3 = MixVMF2(K=K,P=P,part_vec=part_vec,X=X,num_subj=num_subj)
            em4 = MixVMF2(K=K,P=P,part_vec=part_vec,X=X,subject_specific_kappa=True,num_subj=num_subj)
            d=fit_emissions([em1,em2,em3,em4],Y,U,true_kappa=kappa)
            d['sig_subj'] = [sig_s]*d['subj'].shape[0]
            D.append(d)
    D = pd.concat(D)
    return D

if __name__=='__main__':
    # D=sim_subject_differences()
    # sb.barplot(data=D,x='sig_subj',y='kappa',hue='model')
    D = sim_voxel_dependence()
    sb.barplot(data=D,x='model',y='kappa')
    pass