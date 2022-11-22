#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/18/2022
Full Model class

Author: DZHI, jdiedrichsen
"""
import numpy as np
import torch as pt
import generativeMRF.emissions as emi
import generativeMRF.arrangements as arr
import warnings
from copy import copy,deepcopy

class FullModel:
    """The full generative model contains single arrangement model and
       single emission model for training by given dataset
    """
    def __init__(self, arrange,emission):
        self.arrange = arrange
        self.emission = emission
        self.nparams = self.arrange.nparams + self.emission.nparams

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


class FullMultiModel:
    """The full generative model contains arrangement model and multiple
       emission models for training across dataset
    """
    def __init__(self, arrange, emission):
        """Constructor
        Args:
            arrange: the arrangement model
            emission: the list of emission models
        """
        self.arrange = arrange
        self.emissions = emission
        self.n_emission = len(self.emissions)
        self.nparams = self.arrange.nparams + sum([i.nparams for i in self.emissions])
        self.K = self.arrange.K
        self.P = self.arrange.P

    def initialize(self,Y=None,subj_ind='separate'):
        """ Initializes the model for fitting.
        If Y or subj_ind is given, it replaces the existing.
        If set to None, the old existing will be used.

        Args:
            Y (list): List of (numsubj x N x numvoxel) arrays of data
            subj_ind (list): List of unique subject indicators OR
                'separate': sets seperate subjs for each data set OR
                None: Don't change anything
        """
        self.n_emission = len(self.emissions)
        self.nparams = self.arrange.nparams + sum([i.nparams for i in self.emissions])
        self.K = self.emissions[0].K
        self.P = self.emissions[0].P

        self.nsubj_list = []

        if subj_ind is not None:
            self.subj_ind = []

        # Got through emission models and initialize
        sbj = 0
        for i,em in enumerate(self.emissions):
            if Y is not None:
                em.initialize(Y[i])
            self.nsubj_list.append(em.num_subj)
            if isinstance(subj_ind,(list,np.ndarray)):
                if len(subj_ind[i])!=self.nsubj_list[i]:
                    raise(NameError(f"length of subj_ind[{i}] does not match number of subjects: {self.n_subj_list[i]}"))
                else:
                    self.subj_ind.append(subj_ind[i])
            elif subj_ind == 'separate':
                self.subj_ind.append(pt.arange(sbj,self.nsubj_list[i]+sbj))
                sbj = sbj + self.nsubj_list[i]
            elif subj_ind is not None:
                raise(NameError("subj_ind needs to be an array/list, 'separate', or None"))
        # Overall number of unique subjects
        self.nsubj = max([max(i) for i in self.subj_ind]).item()+1
        self.ds_weight = pt.ones((self.n_emission,)) # Experimental dataset weighting

    def clear(self):
        """Clears the data from all emission models
        and temporary statistics from the arrangement model
        """
        self.arrange.clear()
        for em in self.emissions:
            em.clear()
        if hasattr(self,'self_ind'):
            delattr(self,'self_ind')

    def remap_evidence(self,Uhat):
        """Placeholder function of remapping evidence from an
        arrangement space to a emission space (here it doesn't do anything)

        Args:
            Uhat (ndarray): tensor of estimated arrangement
        Returns:
            Uhat (ndarray): tensor of estimated arrangements
        """
        return Uhat

    def collect_evidence(self,emloglik):
        """Collects evidence over the different data sets
        For subjects that are in multiple datasets, it sums the
        log evidence.

        Args:
            emloglik (list): List of emissionlogliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        if not hasattr(self,'subj_ind'):
            raise(NameError('subj_ind not found. Call model.initialize() first.'))
        emlog = pt.zeros(self.nsubj,self.K,self.P)
        for i,eml in enumerate(emloglik):
            emlog[self.subj_ind[i]]+=eml*self.ds_weight[i]
        return emlog

    def distribute_evidence(self,Uhat):
        """Splits the evidence to the different emission models

        Args:
            Uhat (pt.tensor): tensor of estimated or arrangement
        Returns:
            Usplit (list): List of Uhats (per emission model)
        """
        if not hasattr(self,'subj_ind'):
            raise(NameError('subj_ind not found. Call model.initialize() first.'))
        Usplit = []
        for i,s in enumerate(self.subj_ind):
            Usplit.append(Uhat[s])
        return Usplit

    def marginal_prob(self):
        """Convenience function that returns
        marginal probability for the full model

        Returns:
            Prob (pt.tensor): KxP marginal probabilities
        """
        return self.remap_evidence(self.arrange.marginal_prob())

    def set_num_subj(self,num_subj=None):
        """Sets the number of subjects for simulations
        Args:
            num_subj: list of subjects number. i.e [2, 3, 4] for
                            Separate subejcts per dataset OR
                      list of subject indices [[0,1,2],[0,1,2],[2,3,4]]
        """
        # Default - assign 10 subjects per dataset
        if num_subj is None:
            num_subj = [10] * len(self.emissions)

        # Independent subjects in all data sets
        if isinstance(num_subj[0],int):
            for i,em in enumerate(self.emissions):
                em.num_subj = num_subj[i]
            self.initialize(Y=None,subj_ind='separate')
        # Specific overlap between data sets
        elif isinstance(num_subj[0],(list,np.ndarray,pt.tensor)):
            for i,em in enumerate(self.emissions):
                em.num_subj = len(num_subj[i])
            self.initialize(Y=None,subj_ind=num_subj)
        else:
            raise NameError('num_subj needs to be a list of ints or a list of array/lists')

    def sample(self, num_subj=None):
        """Take in the number of subjects to sample for each emission model
        Args:
            num_subj: list of subjects number. i.e [2, 3, 4] Or
                      list of subject indices [[0,1,2],[0,1,2],[2,3,4]]
        Returns:
            U: the true Us of all subjects concatenated vertically,
               shape(num_subs, P)
            Y: data sampled from emission models, shape (num_subs, N, P)
        """
        # If number of subject is given or current nsubj in the model is None,
        # then we overwrite the
        # subjects data per each emission model
        if (num_subj is not None) or (self.nsubj is None):
            self.set_num_subj(num_subj)

        U = self.arrange.sample(self.nsubj)
        Y = []

        for em, Us in enumerate(self.distribute_evidence(U)):
            this_Y = self.emissions[em].sample(Us)
            Y.append(this_Y)
        return U, Y

    def Estep(self, separate_ll=False):
        """E step for full model. Run a full process of EM procedure once
           on both arrangement and emission models.
        Args:
            separate_ll: if True, return separte ll_A and ll_E
        Returns:
            Uhat: the prediction U_hat
            ll: the log-likelihood for arrangement and emission models
        """
        # Run E-step
        emloglik = [e.Estep() for e in self.emissions]

        # Collect the evidence and broadcast to arrangement mode
        emloglik_comb = self.collect_evidence(emloglik)  # combine the log-liklihoods
        Uhat, ll_a = self.arrange.Estep(emloglik_comb)
        ll_e = (Uhat * emloglik_comb).sum()
        if separate_ll:
            return Uhat, [ll_e, ll_a.sum()]
        else:
            return Uhat, ll_a.sum()+ll_e

    def fit_em(self,iter=30, tol=0.01, seperate_ll=False,
            fit_emission=True,
            fit_arrangement=True,
            first_evidence=True):
        """ Run the EM-algorithm on a full model
            this demands that both the Emission and Arrangement model
            have a full Estep and Mstep and can calculate the likelihood,
            including the partition function
        Args:
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
            fit_emission (list / array of bools): If True, fit emission model.
                    Otherwise, freeze it
            fit_arrangement: If True, fit the arrangement model.
                    Otherwise, freeze it
            first_evidence (bool or list of bool): Determines whether evidence
                    is passed from emission models to arrangement model on the
                    first iteration. Usually set to True. If a list of bools, it determines this for each emission model seperately. To improve alignment between emission models from random starting values, only pass evidence from one or none of the emission models. 

        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
            Uhat (pt.tensor): (n_subj,K,P) matrix of estimates - note that this is in the
                space of arrangement model - call distribute_evidence(Uhat) to get this
                in the space of emission model
        """
        if not hasattr(fit_emission, "__len__"):
            fit_emission = [fit_emission]*len(self.emissions)
        if not hasattr(first_evidence, "__len__"):
            first_evidence = [first_evidence]*len(self.emissions)

        # Initialize the tracking
        ll = pt.zeros((iter, 2))
        theta = pt.zeros((iter, self.nparams))

        # Run number of iterations
        for i in range(iter):
            # Track the parameters
            theta[i, :] = self.get_params()

            # Get the (approximate) posterior p(U|Y)
            # emloglik = [e.Estep() for e in self.emissions]
            # Pass emlogliks immediately to collect evidence function,
            # rathen than save a local variable `emloglik` to waste memory
            emloglik_comb = self.collect_evidence([e.Estep() for e in self.emissions])  # Combine subjects

            # If first iteration, only pass the desired emission models (pretraining)
            if i==0:
                emloglik_c = deepcopy([e.Estep() for e in self.emissions])
                for j,emLL in enumerate(emloglik_c):
                    if not first_evidence[j]:
                        emLL[:,:,:]=0
                Uhat, ll_A = self.arrange.Estep(self.collect_evidence(emloglik_c))
            # Otherwise pass all evidence to arrangement model:
            else:
                Uhat, ll_A = self.arrange.Estep(emloglik_comb)

            # Compute the expected complete logliklihood
            ll_E = pt.sum(Uhat * emloglik_comb, dim=(1, 2))
            ll[i, 0] = pt.sum(ll_A)
            ll[i, 1] = pt.sum(ll_E)
            if pt.isnan(ll[i,:].sum()):
                raise(NameError('Likelihood returned a NaN'))
            # Check convergence:
            # This is what was here before. It ignores whether likelihood increased or decreased!
            # if i == iter - 1 or ((i > 1) and (np.abs(ll[i, :].sum() - ll[i - 1, :].sum()) < tol)):
            if i==iter-1:
                break
            elif i>1:
                dl = ll[i,:].sum()-ll[i-1,:].sum() # Change in logliklihood
                # Check if likelihood decreases more than tolerance
                if dl<-tol:
                    warnings.warn(f'Likelihood decreased - terminating on iteration {i}')
                    break
                elif dl<tol:
                    break

            # Updates the parameters
            Uhat_split = self.distribute_evidence(Uhat)
            if fit_arrangement:
                self.arrange.Mstep()
            for em, Us in enumerate(Uhat_split):
                if fit_emission[em]:
                    self.emissions[em].Mstep(Us)

        if seperate_ll:
            return self, ll[0:i+1], theta[0:i+1, :], Uhat
        else:
            return self, ll[0:i+1].sum(axis=1), theta[0:i+1, :], Uhat

    def fit_em_ninits(self, n_inits=20, first_iter=7, iter=30, tol=0.01,
                      fit_emission=True, fit_arrangement=True,
                      init_emission=True, init_arrangement=True,
                      align = 'arrange'):
        """Run the EM-algorithm on a full model starting with n_inits multiple
           random initialization values and escape from local maxima by selecting
           the model with the highest likelihood after first_iter.
           This demands that both the Emission and
           Arrangement model have a full Estep and Mstep and can calculate the
           likelihood, including the partition function
        Args:
            n_inits: the number of random inits
            first_iter: the first few iterations for the random inits to find
                        the inits parameters with maximal likelihood
            iter: the number of iterations for full EM process
            tol: Tolerance on overall likelihood (def: 0.01)
            fit_emission (list): If True, fit emission model. Otherwise, freeze it
            fit_arrangement: If True, fit arrangement model. Otherwise, freeze it
            init_emission (list or bool): Randomly initialize emission models before fitting?
            init_arrangement (bool): Randomly initialize arrangement model before fitting?
            align: (None,'arrange', or int): Alignment one first step is performed
                None: Not performed - Emission models may not get aligned
                'arrange': from the arrangement model only (works only if spatially
                        non-flat (i.e. random) initialization)
                int: from emission model with number i. Works with spatially flat
                        initialization of arrangement model
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of best full model as function of iteration
                the initial iterations are included
            theta (ndarray): History of the parameter vector
            Uhat: the predicted U (probabilistic)
            first_lls: the log-likelihoods for the n_inits random parameters
                       after first_iter runs
        """
        max_ll = pt.tensor([-pt.inf])
        first_lls = pt.full((n_inits,first_iter), pt.nan)
        # Set the first passing of evidence based on alignment pretraining:
        if align is None:
            first_ev = [True]*len(self.emissions)
        elif align=='arrange':
            first_ev = [False]*len(self.emissions)
        else:
            first_ev = [False]*len(self.emissions)
            first_ev[align]=True

        for i in range(n_inits):
            # Making the new set of emission models with random initializations
            fm = deepcopy(self)
            if init_arrangement:
                fm.arrange.random_params()
            if init_emission:
                for em in fm.emissions:
                    em.random_params()

            fm, this_ll, theta, _ = fm.fit_em(first_iter, tol=tol, seperate_ll=False,
                                           fit_emission=fit_emission,
                                           fit_arrangement=fit_arrangement,
                                           first_evidence=first_ev)
            first_lls[i,:len(this_ll)]=this_ll
            if this_ll[-1] > max_ll[-1]:
                max_ll = this_ll
                self = fm
                best_theta = theta
        self, ll, theta, U_hat = self.fit_em(iter=iter-first_iter, tol=tol, seperate_ll=False,
                                           fit_emission=fit_emission,
                                           fit_arrangement=fit_arrangement,
                                           first_evidence=True)
        ll_n = pt.cat([max_ll,ll[1:]])
        theta_n = pt.cat([best_theta,theta[1:, :]])
        return self, ll_n, theta_n, U_hat, first_lls

    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model
        Returns:
            theta (ndarrap)
        """
        emi_params = [i.get_params() for i in self.emissions]
        return pt.cat([self.arrange.get_params(), pt.cat(emi_params)])

    def get_param_indices(self, name):
        """Return the indices for the full model theta vector
        Args:
            name (str): Parameter name in the format of 'arrange.logpi'
                        or 'emissions.<X>.V' where <X> is the index of
                        emission model. For example 'emissions.0.V' will
                        return the Vs from self.emissions[0]
        Returns:
            indices (np.ndarray): 1-d numpy array of indices into the theta vector
        """
        names = name.split(".")
        if (len(names) == 2) and (names[0] == 'arrange'):
            ind = vars(self)[names[0]].get_param_indices(names[1])
            return ind
        elif (len(names) == 3) and (names[0] == 'emissions'):
            ems_nparams = np.cumsum([i.nparams for i in self.emissions])
            nparams_offset = np.insert(ems_nparams, 0, 0) + self.arrange.nparams
            ind = vars(self)[names[0]][int(names[1])].get_param_indices(names[2])
            return ind + nparams_offset[int(names[1])]
        else:
            raise NameError('Parameter name needs to be arrange.<param> '
                            'or emissions.<X>.<param>, where <X> is the index '
                            'of emission model and <param> is the param name. i.g '
                            'emissions.0.V')

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

