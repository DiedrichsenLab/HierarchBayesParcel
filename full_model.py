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
            # Check convergence
            # ll_decrease_flag = (i > 0) and (ll[i, :].sum() - ll[i-1, :].sum() < 0)
            # if i == iter-1 or ((i > 0) and (ll[i,:].sum() - ll[i-1,:].sum() < tol)) or ll_decrease_flag:
            if i == iter - 1 or ((i > 1) and (np.abs(ll[i, :].sum() - ll[i - 1, :].sum()) < tol)):
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
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = np.concatenate([self.arrange.get_params(),self.emission.get_params()])

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            if estep=='sample':
                Uhat,ll_A = self.arrange.epos_sample(emloglik,num_chains=self.arrange.epos_numchains)
                self.arrange.eneg_sample(num_chains=self.arrange.eneg_numchains)
            elif estep=='ssa':
                Uhat,ll_A = self.arrange.epos_ssa(emloglik)
                self.arrange.eneg_ssa()

            # Compute the expected complete logliklihood
            ll_E = np.sum(Uhat * emloglik,axis=(1,2))
            ll[i,0]=np.sum(ll_A)
            ll[i,1]=np.sum(ll_E)

            # Run the Mstep
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(stepsize)

        if seperate_ll:
            return self,ll[:i+1,:], theta[:i+1,:]
        else:
            return self,ll[:i+1,:].sum(axis=1), theta[:i+1,:]

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
        self.nparams = self.arrange.nparams + sum([i.nparams for i in self.emissions])

    def clear(self):
        """Clears the data from all emission models
        """
        for em in self.emissions:
            em.clear()

    def collect_evidence(self,emloglik):
        """Collects evidence over the different data sets
        This is a function here to make inheritance easier

        Args:
            emloglik (list): List of emissionlogliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        return pt.cat(emloglik, dim=0)

    def distribute_evidence(self,Uhat):
        """Splits the evidence to the different emission models
        This is a function here to make inheritance easier

        Args:
            Uhat (ndarray): ndarrays of estimated arrangement
        Returns:
            Usplit (list): List of Uhats (per emission model)
        """
        Usplit = pt.split(Uhat, self.nsub_list, dim=0)
        return Usplit

    def sample(self, num_subj=None):
        """Take in the number of subjects to sample for each emission model
        Args:
            num_subj: list of subjects number. i.g [2, 3, 4]
        Returns:
            U: the true Us of all subjects concatenated vertically,
               shape(num_subs, P)
            Y: data sampled from emission models, shape (num_subs, N, P)
        """
        if num_subj is None:
            # If number of subject not given, then generate 10
            # subjects data per each emission model
            num_subj = [10] * len(self.emissions)

        U = self.arrange.sample(sum(num_subj))
        Y = []
        for em, Us in enumerate(pt.split(U, num_subj, dim=0)):
            this_Y = self.emissions[em].sample(Us)
            Y.append(this_Y)
        return U, Y

    def Estep(self, Y=None, signal=None, separate_ll=False):
        """E step for full model. Run a full process of EM procedure once
           on both arrangement and emission models.
        Args:
            Y: data
            signal: the ground truth signal strength if applied.
            separate_ll: if True, return separte ll_A and ll_E
        Returns:
            Uhat: the prediction U_hat
            ll: the log-likelihood for arrangement and emission models
        """
        if Y is not None:
            for n, em in enumerate(self.emissions):
                em.initialize(Y[n])

        # Comment: Remove this special case and deal with signal over initialization
        # of that particular emission model.
        if signal is not None:  # for GMM with signal strength
            emloglik = [e.Estep(signal=signal) for e in self.emissions]
        else:
            emloglik = [e.Estep() for e in self.emissions]

        # Collect the evidence and broadcast to arrangement mode
        emloglik_comb = self.collect_evidence(emloglik)  # concatenate all emloglike
        Uhat, ll_a = self.arrange.Estep(emloglik_comb)
        ll_e = (Uhat * emloglik_comb).sum()
        if separate_ll:
            return Uhat, [ll_e, ll_a.sum()]
        else:
            return Uhat, ll_a.sum()+ll_e

    def fit_em(self, Y=None, iter=30, tol=0.01, seperate_ll=False,
               fit_emission=True, fit_arrangement=True,first_evidence=True):
        """ Run the EM-algorithm on a full model
            this demands that both the Emission and Arrangement model
            have a full Estep and Mstep and can calculate the likelihood,
            including the partition function
        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
            fit_emission (list / array of bools): If True, fit emission model.
                    Otherwise, freeze it
            fit_arrangement: If True, fit the arrangement model.
                    Otherwise, freeze it
            first_evidence (bool or list of bool): Determines whether evidence
                    is passed from emission models to arrangement model on the
                    first iteration. Usually set to True. However, to improve alignment
                    between emission models from random starting values, you may want to start from
                    False.  Setting one of the emission models to True can be thought
                    of as a very short pretraining phase
                    with that model alone.

        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
        """
        if not hasattr(fit_emission, "__len__"):
            fit_emission = [fit_emission]*len(self.emissions)
        if not hasattr(first_evidence, "__len__"):
            first_evidence = [first_evidence]*len(self.emissions)

        # Initialize the tracking
        ll = np.zeros((iter, 2))
        theta = np.zeros((iter, self.nparams))

        # Intialize the emission model
        if Y is not None:
            for n, em in enumerate(self.emissions):
                em.initialize(Y[n])

        # Get the number of subjects per emission model
        self.nsub_list = []
        for n, em in enumerate(self.emissions):
            self.nsub_list.append(em.num_subj)

        for i in range(iter):
            # Track the parameters
            theta[i, :] = self.get_params()

            # Get the (approximate) posterior p(U|Y)
            emloglik = [e.Estep() for e in self.emissions]
            emloglik_comb = self.collect_evidence(emloglik)  # Combine subjects

            # If first iteration, only pass the desired emission models (pretraining)
            if i==0:
                emloglik_c = deepcopy(emloglik)
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
            if np.isnan(ll[i,:].sum()):
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

    def fit_em_ninits(self, Y=None, n_inits=20, first_iter=7, iter=30, tol=0.01,
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
            Y: data
            n_inits: the number of random inits
            first_iter: the first few iterations for the random inits to find
                        the inits parameters with maximal likelihood
            iter: the number of iterations for full EM process
            tol: Tolerance on overall likelihood (def: 0.01)
            fit_emission (list): If True, fit emission model. Otherwise, freeze it
            fit_arrangement: If True, fit arrangement model. Otherwise, freeze it
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
        max_ll = np.array([-np.inf])
        first_lls = np.full((n_inits,first_iter),np.nan)
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

            fm, this_ll, theta, _ = fm.fit_em(Y, iter=first_iter, tol=tol, seperate_ll=False,
                                           fit_emission=fit_emission,
                                           fit_arrangement=fit_arrangement,
                                           first_evidence=first_ev)
            first_lls[i,:len(this_ll)]=this_ll
            if this_ll[-1] > max_ll[-1]:
                max_ll = this_ll
                self = fm
                best_theta = theta
        self, ll, theta, U_hat = self.fit_em(Y, iter=iter-first_iter, tol=tol, seperate_ll=False,
                                           fit_emission=fit_emission,
                                           fit_arrangement=fit_arrangement,
                                           first_evidence=True)
        ll_n = np.r_[max_ll,ll[1:]]
        theta_n = np.r_[best_theta,theta[1:, :]]
        return self, ll_n, theta_n, U_hat, first_lls

    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model
        Returns:
            theta (ndarrap)
        """
        emi_params = [i.get_params() for i in self.emissions]
        return np.concatenate([self.arrange.get_params(), pt.cat(emi_params)])

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
    def __init__(self, arrange, emission,indx_full,indx_reduced,same_parcels=False):
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

    def collect_evidence(self,emloglik):
        """Collects evidence over the different data sets
        and across the two hemispheres

        Args:
            emloglik (list): List of emissionlogliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        emloglik_comb = pt.cat(emloglik, dim=0)  # concatenate all emloglike
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
        if self.same_parcels:
            Uhat_full = Uhat[:,:,self.indx_reduced]
        else:
            Uhat_full = pt.zeros((Uhat.shape[0],self.K,self.P))
            Uhat_full[:,:self.K_sym,self.indx_full[0]]=Uhat
            Uhat_full[:,self.K_sym:,self.indx_full[1]]=Uhat
        # Distribute over data sets
        Usplit = pt.split(Uhat_full, self.nsub_list, dim=0)
        return Usplit


    def sample(self, num_subj=None):
        """Sample data from each emission model (different subjects)
        Args:
            num_subj: list of subjects numbers per emission model e.g [2, 3, 4]
        Returns:
            U: the true Us of all subjects concatenated vertically,
               shape(num_subs, P)
            Y: data sampled from emission models, shape (num_subs, N, P)
        """
        if num_subj is None:
            # If number of subject not given, then generate 10
            # subjects data per each emission model
            num_subj = [10] * len(self.emissions)

        U = self.arrange.sample(sum(num_subj))
        U = U[:,self.index_reduced]
        # Label all right sided parcels higher
        if not self.same_parcels:
            U[:,self.indx_full[1]]=U[:,self.indx_full[1]]+self.K_sym
        Y = []
        for em, Us in enumerate(pt.split(U, num_subj, dim=0)):
            this_Y = self.emissions[em].sample(Us)
            Y.append(this_Y)
        return U, Y

