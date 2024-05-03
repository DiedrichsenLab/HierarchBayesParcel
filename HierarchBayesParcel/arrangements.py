#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Arrangement models class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
from torch import exp,log
from HierarchBayesParcel.model import Model
import pandas as pd
import warnings

class ArrangementModel(Model):
    """ Abstract arrangement model class
    """

    def __init__(self, K, P):
        self.K = K  # Number of states
        self.P = P  # Number of nodes
        self.tmp_list = []
    
    def map_to_full(self,Uhat):
        """ Placeholder

        Args:
            Uhat (ndarray): tensor of estimated arrangement

        Returns:
            Uhat (ndarray): tensor of estimated arrangements
        """
        return Uhat

    def map_to_arrange(self, emloglik):
        """ Maps emission log likelihoods to the internal size of the representation: Empty

        Args:
            emloglik (list): List of emission logliklihoods

        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        return emloglik

class ArrangeIndependent(ArrangementModel):
    """ Independent arrangement model
    """

    def __init__(self, K=3, P=100, spatial_specific=True,
                 remove_redundancy=False):
        """Constructor for the independent arrangement model

        Args:
            K (int): Number of different parcels
            P (int): Number of voxels / vertices
            spatial_specific (bool): Use a spatially specific
                model (default True)
            remove_redundancy (bool): Code with K or K-1
                probabilities parameters?
        """
        super().__init__(K, P)
        self.spatial_specific = spatial_specific
        if spatial_specific:
            pi = pt.ones(K, P) / K
        else:
            pi = pt.ones(K, 1) / K
        self.logpi = pt.log(pi)
        # Remove redundancy in parametrization
        self.rem_red = remove_redundancy
        if self.rem_red:
            self.logpi = self.logpi - self.logpi[-1, :]
        self.set_param_list(['logpi'])
        self.tmp_list = ['estep_Uhat']

    def random_params(self):
        """ Sets prior parameters to random starting values
        """
        self.logpi = pt.normal(0, 1, size=self.logpi.shape)

    def Estep(self, emloglik, gather_ss=True):
        """ Estep for the spatial arrangement model

        Args:
            emloglik: (pt.tensor) emission log likelihood
                log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss: (bool) Gather Sufficient statistics
                for M-step (default = True)

        Returns:
            Uhat (pt.tensor): posterior p(U|Y) a
                numsubj x K x P matrix
            ll_A (pt.tensor): Expected log-liklihood of
                the arrangement model
        """
        if type(emloglik) is np.ndarray:
            emloglik = pt.tensor(emloglik, dtype=pt.get_default_dtype())
        logq = emloglik + self.logpi
        Uhat = pt.softmax(logq, dim=1)
        if gather_ss:
            self.estep_Uhat = Uhat
        # The log likelihood for arrangement model
        # p(U|theta_A) is sum_i sum_K Uhat_(K)*log pi_i(K)
        pi = pt.softmax(self.logpi,dim=0)
        lpi = pt.nan_to_num(pt.log(pi),neginf=0) # Prevent underflow
        ll_A = pt.sum(Uhat * lpi)
        if pt.isnan(ll_A):
            raise (NameError('likelihood is nan'))
        return Uhat, ll_A

    def Mstep(self):
        """ M-step for the spatial arrangement model. Update
            the pi for arrangement model uses the epos_Uhat
            statistic that is put away from the last e-step.
        """
        # Averarging over subjects
        pi = pt.mean(self.estep_Uhat, dim=0)
        if not self.spatial_specific:
            pi = pi.mean(dim=1).reshape(-1, 1)
        self.logpi = log(pi)
        if self.rem_red:
            self.logpi = self.logpi - self.logpi[-1, :]
        self.logpi = pt.nan_to_num(self.logpi)

    def sample(self, num_subj=10):
        """ Samples a number of subjects from the prior.
        In this i.i.d arrangement model we assume each node has
        no relation with other nodes, which means it equals to
        sample from the prior pi.

        Args:
            num_subj (int): the number of subjects to sample

        Returns:
            U (pt.tensor): the sampled data for subjects
        """
        U = pt.zeros(num_subj, self.P)
        pi = pt.softmax(self.logpi, dim=0)
        for i in range(num_subj):
            if self.spatial_specific:
                U = sample_multinomial(pi, shape=(num_subj,self.K,self.P),
                                       compress=True)
            else:
                pi=pi.expand(self.K, self.P)
                U = sample_multinomial(pi, shape=(num_subj,self.K,self.P),
                                       compress=True)
        return U

    def marginal_prob(self):
        """ Returns marginal probabilty for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        return pt.softmax(self.logpi, dim=0)


class ArrangeIndependentSymmetric(ArrangeIndependent):
    """ Independent arrangement model with symmetry constraint.
        It has two sizes:
        P and K (number of nodes / parcels for arrangement model)
        P_full and K_full (number of location / parcels for data)
    """
    def __init__(self, K,
                 indx_full,
                 indx_reduced,
                 same_parcels=False,
                 spatial_specific=True,
                 remove_redundancy=False):
        """ Constructor for the independent arrangement model

        Args:
            K (int): Number of different parcels
            indx_full (ndarray/tensor): 2 x P array of data-indices
                for each node (L/R)
            indx_reduced (ndarray): P_full - vector of node-indices
                for each data location
            same_parcels (bool): are the mean functional profiles of parcels the same
                or different across hemispheres?
            spatial_specific (bool): Use a spatially specific model
                (default True)
            remove_redundancy (bool): Code with K probabilities
                with K or K-1 parameters?
        """
        if type(indx_full) is np.ndarray:
            indx_full = pt.tensor(
                indx_full, dtype=pt.get_default_dtype()).long()

        if type(indx_reduced) is np.ndarray:
            indx_reduced = pt.tensor(
                indx_reduced, dtype=pt.get_default_dtype()).long()

        self.indx_full = indx_full
        self.indx_reduced = indx_reduced

        self.P_full = indx_reduced.shape[0]
        self.P = indx_full.shape[1]
        self.K_full = K
        if not same_parcels:
            self.K = int(K / 2)
        self.same_parcels = same_parcels
        super().__init__(self.K, self.P, spatial_specific, remove_redundancy)

    def map_to_full(self,Uhat):
        """ Remapping evidence from an arrangement space to a
        emission space (here it doesn't do anything)

        Args:
            Uhat (ndarray): tensor of estimated arrangement

        Returns:
            Uhat (ndarray): tensor of estimated arrangements
        """
        if Uhat.ndim == 3:
            if self.same_parcels:
                Umap = Uhat[:, :, self.indx_reduced]
            else:
                Umap = pt.zeros((Uhat.shape[0], self.K_full, self.P_full))
                Umap[:, :self.K, self.indx_full[0]] = Uhat
                Umap[:, self.K:, self.indx_full[1]] = Uhat
        elif Uhat.ndim == 2:
            if self.same_parcels:
                Umap = Uhat[:, self.indx_reduced]
            else:
                Umap = pt.zeros((self.K_full, self.P_full))
                Umap[:self.K, self.indx_full[0]] = Uhat
                Umap[self.K:, self.indx_full[1]] = Uhat
        return Umap

    def map_to_arrange(self, emloglik):
        """ Maps emission log likelihoods to the internal size of the
        representation

        Args:
            emloglik (list): List of emission logliklihoods

        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        if self.same_parcels:
            emloglik_comb = emloglik[:, :, self.indx_full[0]
                                     ] + emloglik[:, :, self.indx_full[1]]
        else:
            emloglik_comb = emloglik[:, :self.K, self.indx_full[0]
                                     ] + emloglik[:, self.K:, self.indx_full[1]]
        return emloglik_comb

    def Estep(self, emloglik, gather_ss=True):
        """ Estep for the spatial arrangement model

        Args:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss (bool):
                Gather Sufficient statistics for M-step (default = True)

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Expected log-liklihood of the arrangement model
        """
        emloglik = self.map_to_arrange(emloglik)
        Uhat, ll_A = super().Estep(emloglik, gather_ss)
        Uhat = self.map_to_full(Uhat)
        Uhat = Uhat / Uhat.sum(dim=1, keepdim=True)
        return Uhat, ll_A

    def sample(self, num_subj=10):
        """ Samples a number of subjects from the prior.
        In this i.i.d arrangement model we assume each node has
        no relation with other nodes, which means it equals to
        sample from the prior pi.

        Args:
            num_subj (int): the number of subjects to sample

        Returns:
            U (pt.tensor): the sampled data for subjects
        """
        U = super.sample(num_subj)
        U = self.map_to_full(U)
        return U

    def marginal_prob(self):
        """ Returns marginal probability for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        P = self.map_to_full(pt.softmax(self.logpi, dim=0))
        P = P/P.sum(dim=0)
        return P


class ArrangeIndependentSeparateHem(ArrangeIndependentSymmetric):
    """Independent arrangement model without symmetry constraint, 
    but like a symmetric model, it keeps the parcels (and emission models)
    for the left and right hemishere separate. P is the same to the full data,
    K (parcels for arrangement model) K_full (parcels for data)

    Args:
        K (int):
            Number of different parcels
        indx_hem (ndarray/tensor): 
            1 x P array of indices for the hemisphere
            -1 - Left; 0 - Midline; 1 - Right
        spatially_specific (bool):  
            Use a spatially specific model (default True)
        remove_redundancy (bool): 
            Code with K probabilities with K or K-1 parameters? 
    """

    def __init__(self, K,
                 indx_hem,
                 spatial_specific=True,
                 remove_redundancy=False):

        if type(indx_hem) is np.ndarray:
            indx_hem = pt.tensor(
                indx_hem, dtype=pt.get_default_dtype()).long()

        self.indx_hem = indx_hem

        self.P = indx_hem.shape[1]

        self.K = int(K / 2)
        super().__init__(K, indx_full=indx_hem,
                         indx_reduced=indx_hem.T,
                         spatial_specific=spatial_specific,
                         remove_redundancy=remove_redundancy)

    def map_to_full(self, Uhat):
        """ remapping evidence from an
        arrangement space to emission space

        Args:
            Uhat (ndarray): tensor of estimated arrangement
        Returns:
            Umap (ndarray): tensor of estimated arrangements
        """
        left = (self.indx_hem == -1).squeeze()
        right = (self.indx_hem == 1).squeeze()
        midline = (self.indx_hem == 0).squeeze()
        if Uhat.ndim == 3:
            Umap = pt.zeros((Uhat.shape[0], self.K_full, self.P))
            # left hemisphere
            Umap[:, :self.K, left] = Uhat[:, :, left]
            # right hemisphere
            Umap[:, self.K:, right] = Uhat[:, :, right]
            # Map midline to both
            Umap[:, :self.K, midline] = Uhat[:, :,midline] / 2
            Umap[:, self.K:, midline] = Uhat[:, :,midline] / 2
        elif Uhat.ndim == 2:
            Umap = pt.zeros((self.K_full, self.P))
            Umap[:self.K, left] = Uhat[:, left]
            Umap[self.K:, right] = Uhat[:, right]
            # Map midline to both
            Umap[:self.K, midline] = Uhat[:, midline] / 2
            Umap[self.K:, midline] = Uhat[:, midline] / 2
        return Umap

    def map_to_arrange(self, emloglik):
        """ Maps emission log likelihoods to the internal size of the
        representation

        Args:
            emloglik (list): List of emission logliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        emloglik_comb = pt.zeros((emloglik.shape[0], self.K, self.P_full))
        # left hemisphere
        emloglik_comb[:, :, self.indx_hem[0, :] == -1] \
            = emloglik[:, :self.K, self.indx_hem[0, :] == -1]
        # right hemisphere
        emloglik_comb[:, :, self.indx_hem[0, :] == 1] \
            = emloglik[:, self.K:, self.indx_hem[0, :] == 1]

        # midline from left
        emloglik_comb[:, :, self.indx_hem[0, :] == 0] \
            = emloglik[:, :self.K, self.indx_hem[0, :] == 0]
        # midline from right
        emloglik_comb[:, :, self.indx_hem[0, :] == 0] \
            += emloglik[:, self.K:, self.indx_hem[0, :] == 0]

        return emloglik_comb


class PottsModel(ArrangementModel):
    """ Potts models (Markov random field on multinomial variable) with K
        possible states. Potential function is determined by linkages
        parameterization is joint between all linkages, although it could
        be split into different parameter functions
    """

    def __init__(self, W, K=3, remove_redundancy=True):
        self.W = W
        self.K = K  # Number of states
        self.P = W.shape[0]
        self.theta_w = 1  # Weight of the neighborhood relation - inverse temperature param
        self.rem_red = remove_redundancy
        pi = pt.ones((K, self.P)) / K
        self.logpi = log(pi)
        if remove_redundancy:
            self.logpi = self.logpi - self.logpi[-1, :]
        # Inference parameters for persistence CD alogrithm via sampling
        self.epos_U = None
        self.eneg_U = None
        self.fit_theta_w = True  # Update smoothing parameter in Mstep
        self.update_order = None
        self.nparams = 10
        self.set_param_list(['logpi', 'theta_w'])

    def random_smooth_pi(self, Dist, theta_mu=1,centroids=None):
        """ Defines pi (prior over parcels) using a Ising model with K centroids
            Needs the Distance matrix to define the prior probability
        """
        if centroids is None:
            centroids = np.random.choice(self.P, (self.K,))
        d2 = Dist[centroids, :]**2
        pi = exp(-d2 / (2 * theta_mu))
        pi = pi / pi.sum(dim=0)
        self.logpi = log(pi)
        if self.rem_red:
            self.logpi = self.logpi - self.logpi[-1, :]

    def potential(self,y):
        """ Returns the potential functions for the log-linear form of the model

        Args:
            y (ndarray): 2d array (NxP) of network states

        Returns:
            phi (ndarray): 2d array (NxP) of potential functions
        """
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        # Potential on states
        N = y.shape[0]  # Number of observations
        phi = pt.zeros((self.numparam, N))
        for i in range(N):
            S = pt.eq(y[i, :], y[i, :].reshape((-1, 1)))
            phi[0, i] = pt.sum(S * self.W)
        return (phi)

    def loglike(self,U):
        """ Returns the energy term of the network up to a constant
        the loglikelihood of the state

        Args:
            U (ndarray): 2d array (NxP) of network states

        Returns:
            ll (ndarray)): 1d array (N,) of likelihoods
        """
        N, P = U.shape
        la = pt.empty((N,))
        lp = pt.empty((N,))
        for n in range(N):
            phi = np.equal(U[n, :], U[n, :].reshape((-1, 1)))
            la[n] = pt.sum(self.theta_w * self.W * phi)
            lp[n] = pt.sum(self.logpi(U[n,:],range(self.P)))

        return(la + lp)

    def cond_prob(self, U, node, bias):
        """Returns the conditional probabity vector for node x, given U

        Args:
            U (ndarray): Current state of the network
            node (int): Number of node to get the conditional prov for
            bias (pt.tensor): (1,P) Log-Bias term for the node

        Returns:
            p (pt.tensor): (K,) vector of conditional probabilities
                for the node
        """
        x = np.arange(self.K)
        # Find all the neighbors for node x (precompute!)
        ind = np.where(self.W[node,:] > 0)
        nb_x = U[ind] # Neighbors to node x
        same = np.equal(x,nb_x.reshape(-1, 1))
        loglik = self.theta_w * pt.sum(same, dim=0) + bias

        return(pt.softmax(loglik, dim=0))

    def calculate_neighbours(self):
        """Calculate Neighbourhood
        """
        self.neighbours = np.empty((self.P,), dtype=object)
        for p in range(self.P):
            # Find all the neighbors for node x (precompute!)
            self.neighbours[p] = pt.where(self.W[p, :] != 0)[0]

    def sample_gibbs(self, U0=None, num_chains=None, bias=None,
                     iter=5, return_hist=False, track=None):
        """ Samples a number of gibbs-chains simulatenously
        using the same bias term

        Args:
            U0 (nd-array): Initial starting point (num_chains x P).
                Default None - and will be initialized by the bias term alone
            num_chains (int): If U0 not provided, number of chains to initialize
            bias (nd-array): Bias term (in log-probability (K,P)).
                 Defaults to None. Assumed to be the same for all the chains
            iter (int): Number of iterations. Defaults to 5.
            return_hist (bool): Return the history as a second return argument?

        Returns:
            U (nd-array): A (num_chains,P) array of integers
            Uhist (nd-array): Full sampling path - (iter,num_chains,P)
                array of integers (optional, only if return_all = True)

        Notes:
            This probably can be made more efficient by doing some of the
            sampling un bulk?
        """
        # Check for initialization of chains
        if U0 is None:
            prob = pt.softmax(bias, dim=0)
            U0 = sample_multinomial(prob, shape=(num_chains, self.K, self.P),
                                    compress=True)
        else:
            num_chains = U0.shape[0]

        if return_hist:
            if track is None:
                # Initilize array of full history of sample
                Uhist = pt.zeros((iter, num_chains, self.P), dtype=np.int16)
            else:
                Uhist = pt.zeros((iter, self.K))
        if not hasattr(self, 'neighbours'):
            self.calculate_neighbours()

        # Start the chains
        U = U0
        u = pt.arange(self.K).reshape(self.K, 1)

        for i in range(iter):
            if return_hist:
                if track is None:
                    Uhist[i, :, :] = U
                else:
                    for k in range(self.K):
                        Uhist[i, k] = pt.mean(U[:, track] == k)
            # Now loop over noes for all chains at the same tim

            for p in np.arange(self.P-1,-1,-1):
                nb_u = U[:,self.neighbours[p]] # Neighbors to node x
                nb_u = nb_u.reshape(num_chains,1,-1)
                same = pt.eq(nb_u,u)
                loglik = self.theta_w * pt.sum(same,dim=2) + bias[:,p].reshape(1,self.K)
                prob = pt.softmax(loglik,dim=1)
                U[:,p]=sample_multinomial(prob,kdim=1,compress=True)

        return (U,Uhist) if return_hist else U

    def sample(self, num_subj=10, burnin=20):
        """ Samples new subjects from prior: wrapper for sample_gibbs

        Args:
            num_subj (int): Number of subjects. Defaults to 10.
            burnin (int): Number of . Defaults to 20.

        Returns:
            U (pt.tensor): Labels for all subjects (numsubj x P) array
        """
        U = self.sample_gibbs(bias = self.logpi, iter = burnin,
            num_chains = num_subj)

        return U

    def Estep(self, emloglik):
        """ Positive phase of getting p(U|Y) for the spatial arrangement model
        Gets the expectations.

        Args:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E),
                a numsubj x K x P matrix

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Unnormalized log-likelihood of the arrangement model for
                each subject. Note that this does not contain the partition function
        """
        numsubj, K, P = emloglik.shape
        bias = emloglik + self.logpi
        self.epos_U = pt.empty((numsubj, self.epos_numchains, P))
        for s in range(numsubj):
            self.epos_U[s,:,:] = self.sample_gibbs(num_chains=self.epos_numchains,
                                                   bias=bias[s], iter=self.epos_iter)

        # Get Uhat from the sampled examples
        self.epos_Uhat = pt.empty((numsubj, self.K, self.P))
        for k in range(self.K):
            self.epos_Uhat[:, k, :] = pt.sum(
                self.epos_U == k, dim=1) / self.epos_numchains

        # Get the sufficient statistics for the potential functions
        self.epos_phihat = pt.zeros((numsubj,))
        for s in range(numsubj):
            phi = pt.zeros((self.P, self.P))
            for n in range(self.epos_numchains):
                phi = phi + \
                    np.equal(self.epos_U[s, n, :],
                             self.epos_U[s, n, :].reshape((-1, 1)))
            self.epos_phihat[s] = pt.sum(self.W * phi) / self.epos_numchains

        # The log likelihood for arrangement model p(U|theta_A) is not trackable-
        # So we can only return the unormalized potential functions
        if P > 2:
            ll_Ae = self.theta_w * self.epos_phihat
            ll_Ap = pt.sum(self.epos_Uhat * self.logpi, dim=(1, 2))
            if self.rem_red:
                Z = exp(self.logpi).sum(dim=0)  # Local partition function
                ll_Ap = ll_Ap - pt.sum(log(Z))
            ll_A = ll_Ae + ll_Ap
        else:
            # Calculate Z in the case of P=2
            pp=exp(self.logpi[:,0] + self.logpi[:,1].reshape((-1, 1))
                   + np.eye(self.K) * self.theta_w)
            Z = pt.sum(pp) # full partition function
            ll_A = self.theta_w * self.epos_phihat \
                   + pt.sum(self.epos_Uhat*self.logpi,dim=(1,2)) - log(Z)

        return self.epos_Uhat, ll_A

    def eneg_sample(self, num_chains=None, iter=5):
        """ Negative phase of the learning: uses persistent contrastive divergence
        with sampling from the spatial arrangement model (not clampled to data)
        Uses persistence across negative smapling steps

        Args:
            num_chains (int, optional): Number of chains to use. Defaults to None.
            iter (int, optional): Number of iterations. Defaults to 5.

        Returns:
            eneg_Uhat (pt.tensor): Labels for all subjects (numsubj x P) array
        """
        if self.eneg_U is None:
            self.eneg_U = self.sample_gibbs(num_chains=num_chains,
                                            bias=self.logpi, iter=iter)
            # For tracking history: ,return_hist=True,track=0
        else:
            if (num_chains != self.eneg_U.shape[0]):
                raise NameError('num_chains needs to stay constant')
            self.eneg_U = self.sample_gibbs(self.eneg_U,
                                            bias=self.logpi, iter=iter)

        # Get Uhat from the sampled examples
        self.eneg_Uhat = pt.empty((self.K, self.P))
        for k in range(self.K):
            self.eneg_Uhat[k, :] = pt.sum(self.eneg_U == k, dim=0) / num_chains

        # Get the sufficient statistics for the potential functions
        phi = pt.zeros((self.P, self.P))
        for n in range(num_chains):
            phi = phi + \
                np.equal(self.eneg_U[n, :], self.eneg_U[n, :].reshape((-1, 1)))
        self.eneg_phihat = pt.sum(self.W * phi) / num_chains
        return self.eneg_Uhat

    def Mstep(self, stepsize=0.1):
        """ Gradient update for SML or CD algorithm

        Args:
            stepsize (float): Stepsize for the update of the parameters
        """
        # Update logpi
        if self.rem_red:
            # The - pi can be dropped here as we follow the
            # difference between pos and neg anyway
            gradpos_logpi = self.epos_Uhat[:,:-1,:].mean(dim=0)
            gradneg_logpi = self.eneg_Uhat[:-1,:]
            self.logpi[:-1,:] = self.logpi[:-1,:] \
                                + stepsize * (gradpos_logpi - gradneg_logpi)
        else:
            gradpos_logpi = self.epos_Uhat.mean(dim=0)
            gradneg_logpi = self.eneg_Uhat
            self.logpi = self.logpi + stepsize * (gradpos_logpi - gradneg_logpi)

        if self.fit_theta_w:
            grad_theta_w = self.epos_phihat.mean() - self.eneg_phihat
            self.theta_w = self.theta_w + stepsize* grad_theta_w

    def marginal_prob(self):
        """ Returns marginal probabilty for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        # TODO: need find a smarter way to computer marginals
        # U = self.sample(num_subj=1, burnin=20)
        # N, P = U.shape
        # la = pt.empty((self.K, P))
        # lp = pt.empty((N,))
        # for n in range(N):
        #     phi = pt.eq(U[n, :], U[n, :].reshape((-1, 1)))
        #     local = pt.sum(self.theta_w * self.W * phi, dim=0, keepdim=True)
        #     local += self.logpi
        # return (la + lp)
        return pt.softmax(self.logpi, dim=0)


class mpRBM(ArrangementModel):
    """ multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses Contrastive-Divergence k for learning

    Notes:
        Outer nodes (U):
            The outer (most peripheral nodes) are
            categorical with K possible categories.
            There are three different representations:
            a) N x nv: integers between 0 and K-1 (u)
            b) N x K x nv : indicator variables or probabilities (U)
            c) N x (K * nv):  Vectorized version of b- with all nodes
            of category 1 first, etc,
            If not otherwise noted, we will use presentation b).

        Hidden nodes (h):
            In this version we will use binary hidden nodes -
            so to get the same capacity as a mmRBM, one would need to
            set the number of hidden nodes to nh
    """

    def __init__(self, K, P, nh):
        """ Constructor for the mpRBM class
        Args:
            K (int): Number of parcels
            P (int): Number of brain voxels
            nh (int): Number of hidden nodes
        """
        super().__init__(K, P)
        self.K = K
        self.P = P
        self.nh = nh
        self.W = pt.randn(nh, P * K)
        self.bh = pt.randn(nh)
        self.bu = pt.randn(K, P)
        self.eneg_U = None
        self.Etype = 'prob'
        self.alpha = 0.01
        self.epos_iter = 5
        self.set_param_list(['W', 'bh', 'bu'])
        self.tmp_list = ['epos_U', 'epos_Eh', 'eneg_U', 'eneg_Eh']

    def sample_h(self, U):
        """ Sample hidden nodes given an activation state
            of the outer nodes

        Args:
            U (pt.tensor): Indicator or probability tensor
                of outer layer, shape (N, K, P)

        Returns:
            p_h (pt.tensor): probability of the hidden nodes,
                shape (N, nh)
            sample_h (pt.tensor): 0/1 values of discretely
                sampled hidde nodes, shape (N, nh)
        """
        wv = pt.mm(U.reshape(U.shape[0], -1), self.W.t())
        activation = wv + self.bh
        p_h = pt.sigmoid(activation)
        sample_h = pt.bernoulli(p_h)

        return p_h, sample_h

    def sample_U(self, h):
        """ Returns a sampled U as a unpacked indicator variable

        Args:
            h (pt.tensor): Hidden states

        Returns:
            p_u (pt.tensor): Probability of each node [N,K,nv] array
            sample_U (pt.tensor): One-hot encoding of random sample [N,K,nv] array
        """
        N = h.shape[0]
        wh = pt.mm(h, self.W).reshape(N,self.K,self.P)
        p_u = pt.softmax(wh + self.bu,1)
        sample = sample_multinomial(p_u,kdim=1)

        return p_u, sample

    def sample(self, num_subj, iter=10):
        """Draw new subjects from the model

        Args:
            num_subj (int): Number of subjects
            iter (int): Number of iterations until burn in

        Returns:
            u (pt.tensor): Sampled subjects
        """
        p = pt.ones(self.K)
        u = pt.multinomial(p, num_subj * self.P, replacement=True)
        u = u.reshape(num_subj, self.P)
        U = expand_mn(u, self.K)
        for i in range(iter):
            _, h = self.sample_h(U)
            _, U = self.sample_U(h)
        u = compress_mn(U)

        return u

    def Estep(self, emloglik, gather_ss=True, iter=None):
        """ Positive Estep for the multinomial boltzman model
            Uses mean field approximation to posterior to U and hidden parameters.

        Args:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss (bool):
                Gather Sufficient statistics for M-step (default = True)

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Nan - returned for consistency
        """
        if type(emloglik) is np.ndarray:
            emloglik = pt.tensor(emloglik, dtype=pt.get_default_dtype())
        if iter is None:
            iter = self.epos_iter
        N = emloglik.shape[0]
        Uhat = pt.softmax(emloglik + self.bu, dim=1)  # Start with hidden = 0
        for i in range(iter):
            wv = pt.mm(Uhat.reshape(N, -1), self.W.t())
            Eh = pt.sigmoid(wv + self.bh)
            wh = pt.mm(Eh, self.W).reshape(N, self.K, self.P)
            Uhat = pt.softmax(wh + self.bu + emloglik, 1)
        if gather_ss:
            if self.Etype=='vis':
                # This is incorrect, but a understandable and information error
                self.epos_U = pt.softmax(emloglik,dim=1)
            elif self.Etype=='prob':
                # This is correct and isthe olny version that should be used.
                self.epos_U = Uhat
            self.epos_Eh = Eh

        return Uhat, pt.nan

    def marginal_prob(self):
        """ Returns marginal probabilty for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        # If not given, then initialize:
        if self.eneg_U is None:
            U, Eh = self.Eneg()
        pi = pt.mean(self.eneg_U, dim=0)
        return pi

    def Mstep(self):
        """ Performs gradient step on the parameters given
            the learning rate self.alpha
        """
        N = self.epos_Eh.shape[0]
        M = self.eneg_Eh.shape[0]
        epos_U=self.epos_U.reshape(N,-1)
        eneg_U=self.eneg_U.reshape(M,-1)
        self.W += self.alpha * (pt.mm(self.epos_Eh.t(),epos_U)
                                - N / M * pt.mm(self.eneg_Eh.t(),eneg_U))
        self.bu += self.alpha * (pt.sum(self.epos_U,0)
                                 - N / M * pt.sum(self.eneg_U,0))
        self.bh += self.alpha * (pt.sum(self.epos_Eh,0)
                                 - N / M * pt.sum(self.eneg_Eh, 0))


class mpRBM_pCD(mpRBM):
    """ Multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses persistent Contrastive-Divergence k for learning
    """
    def __init__(self, K, P, nh, eneg_iter=3, eneg_numchains=77):
        """ Constructor for the mpRBM_pCD class
        Args:
            K (int): Number of parcels
            P (int): Number of brain voxels
            nh (int): Number of hidden units
            eneg_iter (int): Number of iterations for negative phase
            eneg_numchains (int): Number of chains for energy minimization
        """
        super().__init__(K, P,nh)
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains

    def Eneg(self, U=None):
        """ Negative phase of the persistent contrastive divergence algorithm
        Args:
            U (pt.tensor): Initial values for the chains (default = None)

        Returns:
            EU (pt.tensor): Sampled values for the chains
            Eh (pt.tensor): Sampled values for the hidden units
        """
        if (self.eneg_U is None):
            U = pt.empty(self.eneg_numchains, self.K, self.P).uniform_(0, 1)
        else:
            U = self.eneg_U
        for i in range(self.eneg_iter):
            Eh, h = self.sample_h(U)
            EU, U = self.sample_U(h)
        self.eneg_Eh = Eh
        self.eneg_U = EU
        return self.eneg_U, self.eneg_Eh


class mpRBM_CDk(mpRBM):
    """ Multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses persistent Contrastive-Divergence k for learning
    """
    def __init__(self, K, P, nh, eneg_iter=1):
        """ Constructor for the mpRBM_CDk class

        Args:
            K (int): Number of parcels
            P (int): Number of brain voxels
            nh (int): Number of hidden units
            eneg_iter (int): Number of iterations for negative phase
        """
        super().__init__(K, P,nh)
        self.eneg_iter = eneg_iter

    def Eneg(self,U):
        """ Negative phase of the persistent contrastive divergence algorithm

        Args:
            U (pt.tensor): Initial values for the chains (default = None)

        Returns:
            EU (pt.tensor): Sampled values for the chains
            Eh (pt.tensor): Sampled values for the hidden units
        """
        for i in range(self.eneg_iter):
            Eh, h = self.sample_h(U)
            EU, U = self.sample_U(h)
        self.eneg_Eh = Eh
        self.eneg_U = EU
        return self.eneg_U, self.eneg_Eh


class cmpRBM(mpRBM):
    """ Convolutional multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses variational stochastic maximum likelihood for learning
    """
    def __init__(self, K, P, nh=None, Wc=None, theta=None,
                 eneg_iter=10, epos_iter=10, eneg_numchains=77,
                 momentum=False, wd=0):
        """ Constructor for the cmpRBM class

        Args:
            K (int): number of classes
            P (int): number of brain locations
            nh (int): number of hidden multinomial nodes
            Wc (tensor): 2d/3d-tensor for connecticity weights
            theta (tensor): 1d vector of parameters
            eneg_iter (int): HOw many iterations for each negative step. Defaults to 3.
            eneg_numchains (int): How many chains. Defaults to 77.
        """
        self.K = K
        self.P = P
        self.Wc  = Wc
        self.bu = pt.randn(K,P)
        self.momentum = momentum
        self.wd = wd
        if Wc is None:
            if nh is None:
                raise NameError('Provide Connectivty kernel (Wc)'
                                ' matrix or number of hidden nodes (nh)')
            self.nh = nh
            self.W = pt.randn(nh,P) * 0.1
            self.theta = None
            self.set_param_list(['bu', 'W'])
        else:
            if Wc.ndim == 2:
                self.Wc = Wc.view(Wc.shape[0], Wc.shape[1], 1)
            self.nh = Wc.shape[0]

            if theta is None:
                # self.theta = pt.abs(pt.randn((self.Wc.shape[2],)))
                self.theta = pt.distributions.uniform.Uniform(0, 3).sample((self.Wc.shape[2],))
            else:
                self.theta = pt.tensor(theta, dtype=pt.get_default_dtype())
                if self.theta.ndim ==0:
                    self.theta = self.theta.view(1)
            self.W = (self.Wc * self.theta).sum(dim=2)
            self.set_param_list(['bu', 'theta'])
        self.gibbs_U = None  # samples from the hidden layer for negative phase
        self.alpha = 0.01

        if self.momentum:
            self.MOMENTUM_COEF = 0.6
            self.velocity_W = 0
            self.velocity_bu = 0

        self.epos_iter = epos_iter
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains
        self.use_tempered_transition = False
        self.fit_bu = True
        self.fit_W = True
        self.tmp_list = ['epos_Uhat', 'epos_Hhat', 'eneg_U', 'eneg_H']

    def sample_h(self, U):
        """ Sample hidden nodes given an activation state of the outer nodes

        Args:
            U (NxKxP tensor): Indicator or probability tensor of outer layer
        Returns:
            p_h: (N x nh tensor): probability of the hidden nodes
            sample_h (N x nh tensor): 0/1 values of discretely sampled hidde nodes
        """
        wv = pt.matmul(U, self.W.t())
        # activation = wv + self.b
        # p_h = pt.sigmoid(activation)
        # sample_h = pt.bernoulli(p_h)
        p_h = pt.softmax(wv, 1)
        sample_h = sample_multinomial(p_h, kdim=1)
        return p_h, sample_h

    def sample_U(self, h, emloglik=None):
        """ Returns a sampled U as a unpacked indicator variable
        Args:
            h (tensor): Hidden states (NxKxnh)
        Returns:
            p_u (tensor): Probability of each node [N,K,P] array
            sample_U (tensor): One-hot encoding of random sample [N,K,P] array
        """
        N = h.shape[0]
        act = pt.matmul(h, self.W) + self.bu
        if emloglik is not None:
            act += emloglik
        p_u = pt.softmax(act ,1)
        sample = sample_multinomial(p_u, kdim=1)

        return p_u, sample

    def marginal_prob(self):
        """ Returns marginal probabilty for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        if self.gibbs_U is None:
            return pt.softmax(self.bu, 0)
        else:
            pi = pt.mean(self.gibbs_U, dim=0)
        return pi

    def unnormalized_prob(self, U, H):
        """ Calculate the unnormalized probability of the model given U and H

        Args:
            U (pt.Tensor): The indicator tensor of the outer layer
            H (pt.Tensor): The indicator tensor of the hidden layer

        Returns:
            unnormalized_prob (pt.Tensor): The unnormalized probability
                of the model
        """
        bias = (U * self.bu).sum((1, 2))
        connection = (H @ self.W * U).sum((1, 2))
        return pt.exp(bias + connection).mean()

    def tempered_transition(self, U, betas, emission_model):
        """ Sample from the model using tempered transitions.

        Args:
            U (pt.Tensor): The initial value of U
            betas (list[float]): The temperature coefficient for
                each intermediate distribution, where betas[-1] = 1
            emission_model (EmissionModel): The emission model to
                use for sampling

        Returns:
            U (pt.Tensor): The sampled value of U
            H (pt.Tensor): The sampled value of H

        References:
            https://www.cs.cmu.edu/~rsalakhu/papers/trans.pdf
        """
        # save the current parameters
        true_W = self.W.detach().clone()
        true_bu = self.bu.detach().clone()
        accept = 0.0
        _, H = self.sample_U(U)
        U_p = U
        while (accept < np.random.uniform(0.0, 1.0)):
            accept = 1.0
            for beta in reversed(betas):
                accept /= self.unnormalized_prob(U, H)
                self.W = true_W * beta
                self.bu = true_bu * beta
                accept *= self.unnormalized_prob(U, H)

                Y = emission_model.sample(compress_mn(U_p))
                emloglik = emission_model.Estep(Y)
                _, H = self.sample_h(U_p)
                U_p, U = self.sample_U(H, emloglik)

            # inverse transition T tilde (Gibbs sample in reverse)
            for beta in betas[1:]:
                Y = emission_model.sample(compress_mn(U_p))
                emloglik = emission_model.Estep(Y)
                U_p, U = self.sample_U(H, emloglik)
                _, H = self.sample_h(U_p)

                accept /= self.unnormalized_prob(U, H)
                self.W = true_W * beta
                self.bu = true_bu * beta
                accept *= self.unnormalized_prob(U, H)
            accept = min(1, accept)
        # reload the true parameters
        self.W = true_W
        self.bu = true_bu
        return U, H

    def Estep(self, emloglik,gather_ss=True,iter=None):
        """ Positive Estep for the multinomial boltzman model
            Uses mean field approximation to posterior to U and hidden parameters.

        Args:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss (bool):
                Gather Sufficient statistics for M-step (default = True)

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Nan - returned for consistency
        """
        if type(emloglik) is np.ndarray:
            emloglik = pt.tensor(emloglik, dtype=pt.get_default_dtype())
        if iter is None:
            iter = self.epos_iter
        N = emloglik.shape[0]
        Uhat = pt.softmax(emloglik + self.bu, dim=1)  # Start with hidden = 0
        for i in range(iter):
            wv = pt.matmul(Uhat,self.W.t())
            Hhat = pt.softmax(wv,1)
            # Hsamples = sample_multinomial_pt(Hhat, kdim=1)
            # wh = pt.matmul(Hsamples, self.W)
            wh = pt.matmul(Hhat, self.W)
            Uhat = pt.softmax(wh + self.bu + emloglik, 1)
        if gather_ss:
            self.epos_Uhat = Uhat
            self.epos_Hhat = Hhat

        # The unnormalized log likelihood
        ll_A = pt.sum(Uhat * self.bu) \
               + pt.sum(self.W * pt.matmul(pt.transpose(Uhat, 1, 2), Hhat))
        if pt.isnan(ll_A):
            raise ValueError('likelihood is nan')

        return Uhat, ll_A

    def Eneg(self, iter=None, use_chains=None, emission_model=None):
        """ Negative phase of the E-step for the multinomial boltzman model

        Args:
            iter (int): Number of iterations to run the negative phase
            use_chains (list): Which chains to use for the negative phase
            emission_model (object): Emission model to use for the negative phase

        Returns:
            eneg_U (pt.tensor): Sampled U from the negative phase
            eneg_H (pt.tensor): Sampled H from the negative phase
        """
        # If no iterations specified - use standard
        if iter is None:
            iter = self.eneg_iter
        # If no markov chain are initialized, start them off
        if self.gibbs_U is None:
            p = pt.softmax(self.bu,0)
            # Old sample
            self.gibbs_U = sample_multinomial(p, shape=(self.eneg_numchains,self.K,self.P),
                                              kdim=0, compress=False)
            # sample using pytorch
            # self.gibbs_U = sample_multinomial_pt(p, num_subj=self.eneg_numchains, kdim=0)

        # Grab the current chains
        if use_chains is None:
            use_chains = pt.arange(self.eneg_numchains)

        U = self.gibbs_U[use_chains]
        # sampling using tempered transitions
        if self.use_tempered_transition:
            U0 = U.detach().clone()
            U, H = self.tempered_transition(U0, np.linspace(0.9, 1.0, iter),
                                            emission_model)
        else:
            # standard gibbs sampling
            for i in range(iter):
                Y = emission_model.sample(compress_mn(U))
                emloglik = emission_model.Estep(Y)
                ph, H = self.sample_h(U)
                pu, U = self.sample_U(H, emloglik)

        # TODO: For the last update of the hidden units, it is common
        #  to use the probability instead of sampling a multinomial value
        #  to avoid unnecessary sampling noise.
        #  Refererce: 'A Practical Guide to Training Restricted Boltzmann Machines'
        #  https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
        self.eneg_H = ph
        self.eneg_U = pu
        # Persistent: Keep the new gibbs samples around
        self.gibbs_U[use_chains]=U

        return self.eneg_U,self.eneg_H

    def Mstep(self):
        """ Performs gradient step on the parameters
            given the learning rate self.alpha
        """
        N = self.epos_Hhat.shape[0]
        M = self.eneg_H.shape[0]
        # Update the connectivity
        if self.fit_W:
            gradW = pt.matmul(pt.transpose(self.epos_Hhat,1,2),
                              self.epos_Uhat).sum(dim=0)/N
            gradW -= pt.matmul(pt.transpose(self.eneg_H,1,2),
                               self.eneg_U).sum(dim=0)/M
            # If we are dealing with component Wc:
            if self.Wc is not None:
                if self.momentum:
                    self.velocity_W = self.MOMENTUM_COEF * self.velocity_W \
                                      + self.alpha * gradW
                    gradW = self.velocity_W

                gradW = gradW.view(gradW.shape[0],gradW.shape[1],1)
                weights = pt.sum(self.Wc,dim=(0,1))
                gradT   = pt.sum(gradW*self.Wc,dim=(0,1))/weights
                if self.momentum:
                    self.theta += gradT - self.alpha * self.wd * self.theta
                else:
                    self.theta += self.alpha * gradT - self.alpha * self.wd * self.theta

                self.W = (self.Wc * self.theta).sum(dim=2)
            else:
                if self.momentum:
                    self.velocity_W = self.MOMENTUM_COEF * self.velocity_W \
                                      + self.alpha * gradW
                    self.W += self.velocity_W - self.alpha * self.wd * self.velocity_W
                else:
                    self.W += self.alpha * gradW - self.alpha * self.wd * gradW

        # Update the bias term
        if self.fit_bu:
            gradBU =   1 / N * pt.sum(self.epos_Uhat,0)
            gradBU -=  1 / M * pt.sum(self.eneg_U,0)

            if self.momentum:
                self.velocity_bu = self.MOMENTUM_COEF * self.velocity_bu \
                                 + self.alpha * gradBU
                # self.velocity_bu = self.velocity_bu - self.alpha * self.wd * self.bu
                self.bu += self.velocity_bu
            else:
                self.bu += self.alpha * gradBU
                # self.bu += self.alpha * gradBU - self.alpha * self.wd * self.bu


class wcmDBM(mpRBM):
    """ wcmDBM: weighted convolutional multinomial Deep Boltzman Machine
        for learning of brain parcellations for probabilistic
        input Uses variational stochastic maximum likelihood for learning
    """
    def __init__(self, K, P, nh=None, Wc=None, theta=None, eneg_iter=10,
                 epos_iter=10, eneg_numchains=77, momentum=True):
        """ Constructor for the wcmDBM class

        Args:
            K (int): number of classes
            P (int): number of brain locations
            nh (int): number of hidden multinomial nodes
            Wc (tensor): 2d/3d-tensor for connecticity weights
            theta (tensor): 1d vector of parameters
            eneg_iter (int): HOw many iterations for each negative step.
                Defaults to 3.
            eneg_numchains (int): How many chains. Defaults to 77.
        """
        self.K = K
        self.P = P
        self.Wc = Wc
        self.bu = pt.randn(K, P)
        self.momentum = momentum
        if Wc is None:
            if nh is None:
                raise NameError('Provide Connectivty kernel (Wc)'
                                ' matrix or number of hidden nodes (nh)')
            self.nh = nh
            self.W = pt.randn(nh, P) * 0.1
            self.theta = None
            self.set_param_list(['bu', 'W'])
        else:
            assert Wc.ndim == 2, 'Currently only support Wc is a 2d tensor'
            self.Wc = Wc.to_sparse_csr() if not Wc.is_sparse else Wc
            self.Wc_ind = self.Wc.to_sparse_coo().indices().to('cpu')
            self.nh = Wc.shape[0]

            if theta is None:
                theta = pt.abs(pt.randn((1,))).item()
                self.theta = pt.tensor(theta, dtype=pt.get_default_dtype())
            else:
                self.theta = pt.tensor(theta, dtype=pt.get_default_dtype())
                assert self.theta.ndim == 0, 'theta must be a scalar'

            if self.Wc.layout == pt.sparse_coo:
                self.W = self.Wc * self.theta
            elif self.Wc.layout == pt.sparse_csr:
                self.W = pt.clone(self.Wc)
                self.W.values().mul_(self.theta)

            self.Wc_value_coo = self.Wc.to_sparse_coo().values().to('cpu')
            self.Wc = 1 # Remove Wc
            self.set_param_list(['bu', 'theta'])

        self.gibbs_U = None  # samples from the hidden layer for negative phase
        self.alpha = 0.1

        if self.momentum:
            self.MOMENTUM_COEF = 0.9
            self.velocity_W = 0
            self.velocity_bu = 0

        self.epos_iter = epos_iter
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains
        self.fit_bu = True
        self.fit_W = True
        self.use_tempered_transition = False
        self.pretrain = False

        # Assembly dump list
        # TODO: keep the gibbs chain in the model for pCD?
        self.tmp_list = ['epos_Hhat', 'eneg_H']
        if Wc is not None:
            self.tmp_list += ['W', 'Wc_ind', 'Wc_value_coo']
        if self.momentum:
            self.tmp_list += ['velocity_W', 'velocity_bu']

    def random_params(self):
        """ Sets prior parameters to random starting values
        """
        self.bu = pt.randn(self.K, self.P)
        if self.Wc is not None:
            # theta = pt.abs(pt.randn((1,))).item()
            self.theta = pt.distributions.uniform.Uniform(0, 5).sample()
        else:
            self.W = pt.randn(self.nh, self.P) * 0.1

    def sample_h(self, U):
        """Sample hidden nodes given an activation state of the outer nodes

        Args:
            U (NxKxP tensor): Indicator or probability tensor of outer layer
        Returns:
            p_h: (N x nh tensor): probability of the hidden nodes
            sample_h (N x nh tensor): 0/1 values of discretely sampled hidde nodes
        """
        wv = pt.matmul(U,self.W.t())
        # activation = wv + self.b
        # p_h = pt.sigmoid(activation)
        # sample_h = pt.bernoulli(p_h)
        p_h = pt.softmax(wv,1)
        sample_h = sample_multinomial(p_h, kdim=1)
        return p_h, sample_h

    def sample_U(self, h, emloglik = None):
        """ Returns a sampled U as a unpacked indicator variable

        Args:
            h tensor: Hidden states (NxKxnh)
        Returns:
            p_u: Probability of each node [N,K,P] array
            sample_U: One-hot encoding of random sample [N,K,P] array
        """
        N = h.shape[0]
        act = pt.matmul(h, self.W) + self.bu
        if emloglik is not None and not self.pretrain:
            act += emloglik
        p_u = pt.softmax(act ,1)
        sample = sample_multinomial(p_u,kdim=1)
        return p_u, sample

    def marginal_prob(self):
        """ Returns marginal probabilty for every node under the model

        Returns:
            pi (pt.tensor): marginal probability under the model
        """
        if self.gibbs_U is None:
            return pt.softmax(self.bu,0)
        else:
            pi  = pt.mean(self.gibbs_U,dim=0)
        return pi

    def free_energy(self, emloglik):
        """ Calculate the free energy of the current model states
        TODO: finish the free energy function

        Args:
            emloglik (tensor): log likelihood of the emission model

        Returns:
            free_energy (tensor): free energy of the current model states
        """
        if self.epos_Uhat is None:
            Uhat = pt.softmax(emloglik + self.bu, dim=1)
        else:
            Uhat = self.epos_Uhat

        v_sample = sample_multinomial(Uhat, kdim=1)
        wv = pt.matmul(v_sample, self.W.t())
        vWh = pt.matmul(self.gibbs_U, self.W.t())
        vWh = vWh + self.bu

    def unnormalized_prob(self, U, H):
        """ Calculate the unnormalized probability of the model given U and H

        Args:
            U (pt.Tensor): The indicator tensor of the outer layer
            H (pt.Tensor): The indicator tensor of the hidden layer

        Returns:
            unnormalized_prob (pt.Tensor): The unnormalized probability
                of the model
        """
        bias = (U * self.bu).sum((1, 2))
        connection = (H @ self.W * U).sum((1, 2))
        return pt.exp(bias + connection).mean()

    def tempered_transition(self, U, betas, emission_model):
        """ Sample from the model using tempered transitions.

        Args:
            U (pt.Tensor): The initial value of U
            betas (list[float]): The temperature coefficient for
                each intermediate distribution, where betas[-1] = 1
            emission_model (EmissionModel): The emission model to
                use for sampling

        Returns:
            U (pt.Tensor): The sampled value of U
            H (pt.Tensor): The sampled value of H

        References:
            https://www.cs.cmu.edu/~rsalakhu/papers/trans.pdf
        """
        # save the current parameters
        true_W = self.W.detach().clone()
        true_bu = self.bu.detach().clone()
        accept = 0.0
        _, H = self.sample_U(U)
        U_p = U
        while (accept < np.random.uniform(0.0, 1.0)):
            accept = 1.0
            for beta in reversed(betas):
                accept /= self.unnormalized_prob(U, H)
                self.W = true_W * beta
                self.bu = true_bu * beta
                accept *= self.unnormalized_prob(U, H)

                Y = emission_model.sample(compress_mn(U_p))
                emloglik = emission_model.Estep(Y)
                _, H = self.sample_h(U_p)
                U_p, U = self.sample_U(H, emloglik)

            # inverse transition T tilde (Gibbs sample in reverse)
            for beta in betas[1:]:
                Y = emission_model.sample(compress_mn(U_p))
                emloglik = emission_model.Estep(Y)
                U_p, U = self.sample_U(H, emloglik)
                _, H = self.sample_h(U_p)

                accept /= self.unnormalized_prob(U, H)
                self.W = true_W * beta
                self.bu = true_bu * beta
                accept *= self.unnormalized_prob(U, H)
            accept = min(1, accept)
        # reload the true parameters
        self.W = true_W
        self.bu = true_bu
        return U, H

    def Estep(self, emloglik, gather_ss=True, iter=None):
        """ Positive Estep for the multinomial boltzman model using
            mean field approximation to posterior to U and hidden parameters.

        Args:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss (bool):
                Gather Sufficient statistics for M-step (default = True)
            iter (int): the number of iterations to run the mean field

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Nan - returned for consistency
        """
        if type(emloglik) is np.ndarray:
            emloglik=pt.tensor(emloglik,dtype=pt.get_default_dtype())
        if iter is None:
            iter = self.epos_iter
        N=emloglik.shape[0]
        Uhat = pt.softmax(emloglik + self.bu,dim=1) # Start with hidden = 0
        if self.pretrain:
            # single pass for just the top RBM
            wv = pt.matmul(Uhat, self.W.t())
            Hhat = pt.softmax(wv, 1)
        else:
            # mean-field approximation for the entire network
            try:
                # First try if cuda can handle this large tensor
                for i in range(iter):
                    wv = pt.matmul(Uhat, self.W.t())
                    # wv = pt.stack([pt.sparse.mm(self.W, Uhat[j].transpose(0, 1))
                    #           for j in range(N)]).transpose(1,2)
                    Hhat = pt.softmax(wv, 1)
                    wh = pt.matmul(Hhat, self.W)
                    Uhat = pt.softmax(wh + self.bu + emloglik, 1)
            except BaseException as e:
                # If the given tensor too large, we convert to cpu computation
                print(e)
                print('We have to Fall back to CPU computation.')
                # First convert these large tensor to cpu
                Uhat = Uhat.to('cpu')
                self.W = self.W.to('cpu')
                # mean-field approximation for the entire network
                for i in range(iter):
                    wv = pt.matmul(Uhat, self.W.t())
                    Hhat = pt.softmax(wv, 1)
                    wh = pt.matmul(Hhat, self.W)
                    Uhat = pt.softmax(wh + self.bu.to('cpu')
                                      + emloglik.to('cpu'), 1)

                # Move back to cuda after cpu compuation
                Uhat = Uhat.to('cuda')
                self.W = self.W.to('cuda')
            else:
                pass

        if gather_ss:
            self.epos_Uhat = Uhat
            self.epos_Hhat = Hhat
        return Uhat, pt.nan

    def Eneg(self, iter=None, use_chains=None, emission_model=None):
        """ Negative Estep for the multinomial boltzman model using

        Args:
            iter (int): the number of iterations to run the mean field
            use_chains (list[int]): the chains to use for the negative step
            emission_model (EmissionModel): the emission model to use for

        Returns:
            eneg_U (pt.tensor): the sampled U values
            eneg_H (pt.tensor): the sampled H values
        """
        # If no iterations specified - use standard
        if iter is None:
            iter = self.eneg_iter
        # If no markov chain are initialized, start them off
        if self.gibbs_U is None:
            p = pt.softmax(self.bu,0)
            self.gibbs_U = sample_multinomial(p,
                                              shape=(self.eneg_numchains,
                                                     self.K, self.P),
                                              kdim=0,
                                              compress=False)
        # Grab the current chains
        if use_chains is None:
            use_chains = pt.arange(self.eneg_numchains)

        U = self.gibbs_U[use_chains]

        if self.pretrain:
            # gibbs sample for only the top RBM
            for i in range(iter):
                _, H = self.sample_h(U)
                _, U = self.sample_U(H)
        else:
            # sampling using tempered transitions
            if self.use_tempered_transition:
                U0 = U.detach().clone()
                U, H = self.tempered_transition(U0, np.linspace(0.9, 1.0, iter),
                                                emission_model)
            else:
                # standard gibbs sampling
                for i in range(iter):
                    Y = emission_model.sample(compress_mn(U))
                    emloglik = emission_model.Estep(Y)
                    ph, H = self.sample_h(U)
                    pu, U = self.sample_U(H, emloglik)

        self.eneg_H = ph
        self.eneg_U = pu
        # Persistent: Keep the new gibbs samples around
        self.gibbs_U[use_chains] = U
        return self.eneg_U, self.eneg_H

    def Mstep(self):
        """ Performs gradient step on the parameters
            given the learning rate self.alpha
        """
        N = self.epos_Hhat.shape[0]
        M = self.eneg_H.shape[0]
        # Update the connectivity
        if self.fit_W:
            try:
                gradW = pt.matmul(pt.transpose(self.epos_Hhat,1,2),
                                  self.epos_Uhat).sum(dim=0)/N
                gradW -= pt.matmul(pt.transpose(self.eneg_H,1,2),
                                   self.eneg_U).sum(dim=0)/M
            except BaseException as e:
                print('No enough cuda memory for calculating the gradW,'
                      ' we have to back to cpu computation.')

                gradW = pt.matmul(pt.transpose(self.epos_Hhat, 1, 2).to('cpu'),
                                  self.epos_Uhat.to('cpu')).sum(dim=0) / N
                gradW -= pt.matmul(pt.transpose(self.eneg_H, 1, 2).to('cpu'),
                                   self.eneg_U.to('cpu')).sum(dim=0) / M

                # gradW = pt.zeros(self.W.shape, device='cpu')
                # for i in range(N):
                #     gradW += pt.matmul(pt.transpose(self.epos_Hhat, 1, 2)[i].to('cpu'),
                #                       self.epos_Uhat[i].to('cpu'))
                # gradW = gradW / N
                #
                # for j in range(M):
                #     gradW -= pt.matmul(pt.transpose(self.eneg_H,1,2)[j].to('cpu'),
                #                    self.eneg_U[j].to('cpu'))/M
            else:
                pass
            # Convert gradW to sparse COO on cuda
            # values = gradW[self.Wc_ind[0], self.Wc_ind[1]]
            # gradW = pt.sparse_coo_tensor(self.Wc_ind, values,
            #                              size=gradW.shape).coalesce()

            # If we are dealing with component Wc:
            if self.Wc is not None:
                if self.momentum:
                    self.velocity_W = self.MOMENTUM_COEF * self.velocity_W \
                                      + self.alpha * gradW
                    gradW = self.velocity_W

                # weights = pt.sparse.sum(self.Wc)
                # gradT   = pt.sparse.sum(gradW * self.Wc) / weights
                gradT = gradW[self.Wc_ind[0], self.Wc_ind[1]].sum() \
                        / self.Wc_value_coo.sum()
                if self.momentum:
                    self.theta += gradT.to('cuda')
                else:
                    self.theta += self.alpha * gradT

                # Update W only to its value, without creating new sparse tensor to save memory
                self.W.values().mul_(0).add_(self.Wc_value_coo.to('cuda')*self.theta)
                # self.W = pt.sparse_csr_tensor(self.W.crow_indices(),
                #                               self.W.col_indices(),
                #                               self.Wc_value_coo.to('cuda') * self.theta,
                #                               size=self.W.shape)

            else:
                if self.momentum:
                    self.velocity_W = self.MOMENTUM_COEF * self.velocity_W \
                                      + self.alpha * gradW
                    self.W += self.velocity_W
                else:
                    self.W += self.alpha * gradW

        # Update the bias term
        if self.fit_bu:
            gradBU =   1 / N * pt.sum(self.epos_Uhat,0)
            gradBU -=  1 / M * pt.sum(self.eneg_U,0)

            if self.momentum:
                self.velocity_bu = self.MOMENTUM_COEF * self.velocity_bu \
                                 + self.alpha * gradBU
                self.bu += self.velocity_bu
            else:
                self.bu += self.alpha * gradBU

    def sample(self, num_subj, Wc, iter=10):
        """ Samples from the model

        Args:
            num_subj (int): Number of subjects to sample
            iter (int): Number of iterations to run the sampler

        Returns:
            samples (dict): Dictionary of samples
        """
        Uhat = pt.randn((num_subj, self.K, self.P))
        Uhat = pt.softmax(Uhat, dim=1)
        W = Wc * self.theta

        for i in range(iter):
            wv = pt.matmul(Uhat, W.t())
            Hhat = pt.softmax(wv, 1)
            wh = pt.matmul(Hhat, W)
            Uhat = pt.softmax(wh + self.bu, 1)

        return Uhat

####################################################################
## Belows are the helper functions for spatial arrangement models ##
####################################################################
def sample_multinomial_pt(p, num_subj=1, kdim=0, compress=False):
    """ Samples from a multinomial distribution using pytorch built in multinomial distribution sampler

    Args:
        p (pt.tensor): Tensor of probabilities
        num_subj (int): Number of subjects to sample
        kdim (int): Dimension of K
        compress (bool): Whether to compress the output or not

    Returns:
        samples (pt.tensor): Samples from the multinomial distribution
    """
    K = p.shape[kdim]
    if p.dim() == 2:
        # p is K x P matrix
        sample = pt.multinomial(p.t(), num_subj, replacement=True).t()
    elif p.dim() == 3:
        # p is num_subjects x K x P matrix
        sample = pt.stack([pt.multinomial(this_p.t(), 1, replacement=True).reshape(-1)
                           for this_p in p])
    else:
        raise ValueError("p must be 2 or 3 dimensional")

    if compress:
        return sample
    else:
        # No compress, return indicator coding
        return expand_mn(sample, K)

def sample_multinomial(p,shape=None,kdim=0,compress=False):
    """ Samples from a multinomial distribution. fast smpling from matrix probability without looping

    Args:
        p (tensor): Tensor of probilities, which sums to 1
            on the dimension kdim
        shape (tuple): Shape of the output data (in uncompressed form)
            Smaller p will be broadcasted to target shape
        kdim (int): Number of dimension of p that indicates the different
            categories (default 0)
        compress (bool): Return as int (True) or indicator (false)

    Returns:
        samples (tensor): Samples either in indicator coding (compress = False)
            or as ints (compress = False)
    """
    if shape is None:
        shape = p.shape
    out_kdim = len(shape) - p.dim() + kdim
    K = p.shape[kdim]
    shapeR = list(shape)
    shapeR[out_kdim] = 1  # Set the probability dimension to 1
    r = pt.empty(shapeR).uniform_(0, 1)
    cdf_v = p.cumsum(dim=kdim)
    sample = (r < cdf_v).to(pt.get_default_dtype())
    if compress:
        return sample.argmax(dim=out_kdim)
    else:
        for k in np.arange(K - 1, 0, -1):
            a = sample.select(out_kdim, k)  # Get view of slice
            a -= sample.select(out_kdim, k - 1)
        return sample


def expand_mn(u,K):
    """ Expands a N x P multinomial vector to an N x K x P tensor of indictor variables

    Args:
        u (2d-tensor): N x nv matrix of samples from [int]
        K (int): Number of categories

    Returns:
        U (3d-tensor): N x K x nv matrix of indicator variables
            [default float]
    """
    N = u.shape[0]
    P = u.shape[1]
    U = pt.zeros(N, K, P)
    U[np.arange(N).reshape((N, 1)), u, np.arange(P)] = 1
    return U


def expand_mn_1d(U, K):
    """ Expands a P long multinomial vector to an K x P tensor of indictor variables

    Args:
        U (1d-tensor): P vector of samples from [int]
        K (int): Number of categories

    Returns:
        U (2d-tensor): K x P matrix of indicator variables
         [default float]
    """
    if type(U) is np.ndarray:
        U = pt.tensor(U, dtype=pt.long)
        
    P = U.shape[0]
    U_return = pt.zeros(K, P)
    U_return[U, pt.arange(P)] = 1

    return U_return


def compress_mn(U):
    """ Compresses a N x K x P tensor of indictor variables to a N x P multinomial tensor

    Args:
        U (3d-tensor): N x K x P matrix of indicator variables
    Returns:
        u (2d-tensor): N x P matrix of category labels [int]
    """
    u = U.argmax(1)
    return u

def build_arrangement_model(U, prior_type='prob', atlas=None, sym_type='asym',
                            model_type='independent'):
    """ Builds an arrangment model based on a set of probability 

    Args:
        U (tensor or ndarray): 
            A K x P matrix of group probability
        prior_type (str):
            the type of prior, either 'prob' or 'logpi' (default: 'prob')
            if 'prob', the input is the marginal probability K x P matrix,
            which the columns sum to 1. If 'logpi', the input is the group
            prior in log-space
        atlas (object): 
            the atlas object for the arrangement model for symmetric models
        sym_type (str): 
            the symmetry type of the arrangement model (default: 'asym')
        model_type (str):
            the arrangement model type (default: 'independent')

    Returns:
        ar_model (object): the arrangement model object
    """
    # convert tdata to tensor
    if type(U) is np.ndarray:
        U = pt.tensor(U, dtype=pt.get_default_dtype())
    K,P = U.shape

    # Check if the marginal has nan or zero values
    if prior_type == 'prob':
        # if it has nan values - give flat distribution
        if pt.isnan(U).any().item():
            nan_voxl = pt.sum(pt.any(pt.isnan(U), dim=0)).item()
            warnings.warn(f'The marginal probability has {nan_voxl} voxels '
                          f'NaN value - replacing with flat distribution')
            U = U.nan_to_num(1/K)
        # if it has zero values - add small value to avoid -inf
        if pt.eq(U, 0).any().item():
            epsilon = 1e-8
            zero_voxl = pt.sum(pt.any(U == 0, dim=0)).item()
            warnings.warn(f'The marginal probability has {zero_voxl} voxels'
                          f' zero values - adding small value to avoid -inf')
            while pt.any(pt.isinf(U.log())).item():
                U += epsilon
        # Convert to log space for building arrangement model
        U = pt.log(U)
    elif prior_type == 'logpi':
        pass
    else:
        raise ValueError(f'Unknown prior_type: {prior_type}, it must be '
                         f'either prob or logpi!')

    # Build arrangement model by given model and symmetry type
    if model_type == 'independent':
        if sym_type == 'sym':
            ar_model = ArrangeIndependentSymmetric(K,
                                                   atlas.indx_full,
                                                   atlas.indx_reduced,
                                                   same_parcels=False,
                                                   spatial_specific=True,
                                                   remove_redundancy=False)
            # Warn if the input is not symmetric
            if not pt.allclose(U[:ar_model.K, ar_model.indx_full[0]], U[ar_model.K:, ar_model.indx_full[1]]):
                warnings.warn('The input probability is not symmetric, '
                              'but the model is symmetric! Check you are importing the correct probabilities')
            U = U.unsqueeze(0)
            U = ar_model.map_to_arrange(U)
            U = U.squeeze(0)
            non_vermal = ar_model.indx_full[0] != ar_model.indx_full[1]
            # U.sum(dim=0)[non_vermal]
            U[:,non_vermal] = U[:,non_vermal] / 2
            # assert(pt.allclose(U.sum(dim=0), pt.ones(U.shape[1])))
            ar_model.name = 'indp_ym'
        elif sym_type == 'asym':
            ar_model = ArrangeIndependent(K, P,
                                          spatial_specific=True,
                                          remove_redundancy=False)
            ar_model.name = 'indp_asym'

        # Attach the logpi to the model
        ar_model.logpi = U
    else:
        raise NameError(f'Unknown model_type:{model_type} - Currently only '
                        f'support independent arrangement model.')

    return ar_model
