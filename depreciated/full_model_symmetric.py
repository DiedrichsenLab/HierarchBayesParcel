#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/18/2022
Full Model class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
from torch.utils.data import DataLoader, TensorDataset
import generativeMRF.emissions as emi
import generativeMRF.arrangements as arr
import warnings
from copy import copy,deepcopy
from generativeMRF.full_model import FullMultiModel

class FullModel:
    """The full generative model contains single arrangement model and
       single emission model for training by given dataset
    """
    def __init__(self, arrange,emission):
        self.arrange = arrange
        self.emission = emission
        self.nparams = self.arrange.nparams + self.emission.nparams
        DeprecationWarning('Full Model will be removed in future verisons - use FullMultiModel')

    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y

    def Estep(self, Y=None, signal=None, separate_ll=False):
        if Y is not None:
            self.emission.initialize(Y)

        if signal is not None:  # for GMM with signal strength
            emloglik = self.emission.Estep(signal=signal)
        else:
            emloglik = self.emission.Estep()

        Uhat, ll_a = self.arrange.Estep(emloglik)
        ll_e = (Uhat * emloglik).sum()
        if separate_ll:
            return Uhat,[ll_e,ll_a.sum()]
        else:
            return Uhat,ll_a.sum()+ll_e

    def fit_em(self, Y, iter=30, tol=0.01, seperate_ll=False,
               fit_emission=True, fit_arrangement=True, signal=None):
        """ Run the EM-algorithm on a full model
        this demands that both the Emission and Arrangement model
        have a full Estep and Mstep and can calculate the likelihood, including the partition function

        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector

        """
        # Initialize the tracking
        ll = np.zeros((iter, 2))
        theta = np.zeros((iter, self.nparams))
        if signal is not None:
            self.emission.initialize(Y, signal=signal)
        else:
            self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = self.get_params()

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood
            ll_E = pt.sum(Uhat * emloglik, dim=(1, 2))
            ll[i, 0] = pt.sum(ll_A)
            ll[i, 1] = pt.sum(ll_E)
            # Check convergence:
            # This is what is here before. It ignores whether likelihood increased or decreased!
            # if i == iter - 1 or ((i > 1) and (np.abs(ll[i, :].sum() - ll[i - 1, :].sum()) < tol)):
            if i == iter - 1:
                break
            elif i > 1:
                dl = ll[i, :].sum() - ll[i - 1, :].sum()  # Change in logliklihood
                if dl < 0:
                    warnings.warn(f'Likelihood decreased - terminating on iteration {i}')
                    break
                elif dl < tol:
                    break

            # Updates the parameters
            if fit_emission:
                self.emission.Mstep(Uhat)
            if fit_arrangement:
                self.arrange.Mstep()

        if seperate_ll:
            return self, ll[0:i+1], theta[0:i+1,:], Uhat
        else:
            return self, ll[0:i+1].sum(axis=1), theta[0:i+1,:], Uhat

    def fit_sml(self, Y, iter=60, stepsize= 0.8, seperate_ll=False, estep='sample'):
        """ Runs a Stochastic Maximum likelihood algorithm on a full model.
        The emission model is still assumed to have E-step and Mstep.
        The arrangement model is has a postive and negative phase estep,
        and a gradient M-step. The arrangement likelihood is not necessarily
        FUTURE EXTENSIONS:
        * Sampling of subjects from training set
        * initialization of parameters
        * adaptitive stopping criteria
        * Adaptive stepsizes
        * Gradient acceleration methods
        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations
            stepsize (double): Fixed step size for MStep
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
        """
        # Initialize the tracking
        ll = np.zeros((iter,2))
        theta = np.zeros((iter, self.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            print(f'start: {i}')
            # Track the parameters
            theta[i, :] = self.get_params()

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            if estep=='sample':
                Uhat,_ = self.arrange.Estep(emloglik)
                if hasattr(self.arrange, 'Eneg'):
                    self.arrange.Eneg(emission_model=self.emission)

            elif estep=='ssa':
                Uhat,ll_A = self.arrange.epos_ssa(emloglik)
                self.arrange.eneg_ssa()

            # Compute the expected complete logliklihood
            ll_E = pt.sum(Uhat * emloglik, dim=(1, 2))
            ll[i, 0] = pt.sum(ll_E)

            # Run the Mstep
            self.emission.Mstep(Uhat)
            self.arrange.Mstep()

        if seperate_ll:
            return self, ll[0:i + 1], theta[0:i + 1, :], Uhat
        else:
            return self, ll[0:i + 1].sum(axis=1), theta[0:i + 1, :], Uhat

    def ELBO(self,Y):
        """Evidence lower bound of the data under the full model
        Args:
            Y (nd-array): numsubj x N x P array of data
        Returns:
            ELBO (nd-array): Evidence lower bound - should be relatively tight
            Uhat (nd-array): numsubj x K x P array of expectations
            ll_E (nd-array): emission logliklihood of data (numsubj,)
            ll_A (nd-array): arrangement logliklihood of data (numsubj,)
            lq (nd-array): <log q(u)> under q: Entropy
        """
        self.emission.initialize(Y)
        emloglik=self.emission.Estep()
        try:
            Uhat,ll_A,QQ = self.arrange.Estep(emloglik,return_joint=True)
            lq = np.sum(np.log(QQ)*QQ,axis=(1,2))
        except:
            # Assume independence:
            Uhat,ll_A = self.arrange.Estep(emloglik)
            lq = np.sum(np.log(Uhat)*Uhat,axis=(1,2))
            # This is the same as:
            # Uhat2 = Uhat[0,:,0]*Uhat[0,:,1].reshape(-1,1)
            # l_test = np.sum(np.log(Uhat2)*Uhat2)
        ll_E = np.sum(emloglik*Uhat,axis=(1,2))
        ELBO = ll_E + ll_A - lq
        return  ELBO, Uhat, ll_E,ll_A,lq


    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model
        Returns:
            theta (ndarrap)
        """
        return np.concatenate([self.arrange.get_params(),self.emission.get_params()])

    def get_param_indices(self,name):
        """Return the indices for the full model theta vector

        Args:
            name (str): Parameter name in the format of 'arrange.logpi'
                        or 'emission.V'
        Returns:
            indices (np.ndarray): 1-d numpy array of indices into the theta vector
        """
        names = name.split(".")
        if (len(names)==2) and (names[0] in vars(self)):
            ind=vars(self)[names[0]].get_param_indices(names[1])
            if names[0]=='emission':
                ind=ind+self.arrange.nparams
            return ind
        else:
            raise NameError('Parameter name needs to be arrange.param or emission.param')


class FullMultiModelSymmetric(FullMultiModel):
    """ Full generative model contains arrangement model and multiple
       emission models for training across dataset
    """
    def __init__(self, arrange, emission, indx_full, indx_reduced, same_parcels=False):
        """Constructor
        Args:
            arrange: the arrangement model with P_sym nodes
            emission: the list of emission models, each one with P nodes
            indx_full (ndarray): 2 x P_sym array of indices mapping data to nodes
            indx_reduced (ndarray): P-vector of indices mapping nodes to data
            same_parcels (bool): are the means of parcels the same or different across hemispheres?
        """
        super().__init__(arrange, emission)
        self.same_parcels = same_parcels
        if type(indx_full) is np.ndarray:
            indx_full = pt.tensor(indx_full, dtype=pt.get_default_dtype()).long()

        if type(indx_reduced) is np.ndarray:
            indx_reduced = pt.tensor(indx_reduced, dtype=pt.get_default_dtype()).long()

        self.indx_full = indx_full
        self.indx_reduced = indx_reduced
        self.P_sym = arrange.P
        self.K_sym = arrange.K
        self.K = self.emissions[0].K
        self.P = self.emissions[0].P

        if indx_full.shape[1]!=self.P_sym:
            raise(NameError('index_full must be of size 2 x P_sym'))
        if indx_reduced.shape[0]!=self.emissions[0].P:
            raise(NameError('index_reduced must be of size P (same as emissions)'))
        if not same_parcels:
            if self.K_sym*2 != self.K:
                raise(NameError('K in emission models must be twice K in arrangement model'))
        DeprecationWarning('FullMultiModelSymmetric will be removed in future verisons - use FullMultiModel \
            and a symmetric arrangement model. Old models can be translated with full_model.update_symmetric()')


    def remap_evidence(self,Uhat):
        """Placeholder function of remapping evidence from an
        arrangement space to a emission space (here it doesn't do anything)
        Args:
            Uhat (ndarray): tensor of estimated arrangement
        Returns:
            Uhat (ndarray): tensor of estimated arrangements
        """
        if Uhat.ndim == 3:
            if self.same_parcels:
                Umap = Uhat[:,:,self.indx_reduced]
            else:
                Umap = pt.zeros((Uhat.shape[0],self.K,self.P))
                Umap[:,:self.K_sym,self.indx_full[0]]=Uhat
                Umap[:,self.K_sym:,self.indx_full[1]]=Uhat
        elif Uhat.ndim==2:
            if self.same_parcels:
                Umap = Uhat[:,self.indx_reduced]
            else:
                Umap = pt.zeros((self.K,self.P))
                Umap[:self.K_sym,self.indx_full[0]]=Uhat
                Umap[self.K_sym:,self.indx_full[1]]=Uhat
        DeprecationWarning('FullMultiModelSymmetric will be removed in future verisons - use FullMultiModel \
            and a symmetric arrangement model. Old models can be translated with full_model.update_symmetric()')
        return Umap

    def collect_evidence(self,emloglik):
        """Collects evidence over the different data sets
        and across the two hemispheres

        Args:
            emloglik (list): List of emissionlogliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        emloglik_comb = super().collect_evidence(emloglik)  # concatenate all emloglike
        # Combine emission log-likelihoods across left and right sides
        if self.same_parcels:
            emloglik_comb = emloglik_comb[:,:,self.indx_full[0]] + emloglik_comb[:,:,self.indx_full[1]]
        else:
            emloglik_comb = emloglik_comb[:,:self.K_sym,self.indx_full[0]] + emloglik_comb[:,self.K_sym:,self.indx_full[1]]
        return emloglik_comb

    def distribute_evidence(self,Uhat):
        """Splits the evidence to the different emission models
        and expands it from the reduced representation to a
        full representation across both sides of the brain

        Args:
            Uhat (ndarray): ndarrays of estimated arrangement
        Returns:
            Usplit (list): List of Uhats (per emission model)
        """
        # Map the evidence back to the orginal space
        Uhat_full = self.remap_evidence(Uhat)
        # Distribute over data sets
        Usplit = emloglik_comb = super().distribute_evidence(Uhat_full)
        return Usplit


def update_symmetric_model(model):
    """ Updates a symmetric model from a 
    FullMultiModelSymmetric + ArrangementIndependent
    FullMultiModel + ArrangementIndependentSymmetric 
    Args:
        model (FullMultiModelSymmetric)
    Returns:
        new_model (FullMultiModel)
    """
    if type(model) is FullMultiModel:
        if type(model.arrange) is arr.ArrangeIndependent:
            raise(NameError('Warning: Model is not not symmetric'))
        elif type(model.arrange) is arr.ArrangeIndependentSymmetric:
            print('Warning: Model is already updated')
            return model
    elif type(model) is FullMultiModelSymmetric:
        new_arrange = arr.ArrangeIndependentSymmetric(model.K,
            model.indx_full,
            model.indx_reduced,
            model.same_parcels,
            model.arrange.spatial_specific,
            model.arrange.rem_red)
        new_arrange.logpi = model.arrange.logpi
        new_model = FullMultiModel(new_arrange,model.emissions)
        new_model.nsubj = model.nsubj
        new_model.n_emission = model.n_emission
        new_model.nsubj_list = model.nsubj_list
        new_model.subj_ind = model.subj_ind
        if hasattr(model,'ds_weight'):
            new_model.ds_weight = model.ds_weight

    return new_model
