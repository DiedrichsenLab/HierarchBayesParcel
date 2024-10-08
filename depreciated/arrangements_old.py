# Example Models
import os  # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
from decimal import Decimal
from torch import exp,log,sqrt
from model import Model
import torch as pt
from arrangements import *
import sys

class PottsModel_old(PottsModel):
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states
    Potential function is determined by linkages
    parameterization is joint between all linkages, although it could be split
    into different parameter functions
    """

    def potential(self,y):
        """
        returns the potential functions for the log-linear form of the model
        """
        if y.ndim==1:
            y=y.reshape((-1,1))
        # Potential on states
        N = y.shape[0] # Number of observations
        phi = pt.zeros((self.numparam,N))
        for i in range(N):
           S = np.equal(y[i,:],y[i,:].reshape((-1,1)))
           phi[0,i]=pt.sum(S*self.W)
        return(phi)

    def loglike(self,U):
        """Returns the energy term of the network
        up to a constant the loglikelihood of the state

        Params:
            U (ndarray): 2d array (NxP) of network states
        Returns:
            ll (ndarray)): 1d array (N,) of likelihoods
        """
        N,P = U.shape
        la = pt.empty((N,))
        lp = pt.empty((N,))
        for n in range(N):
            phi=np.equal(U[n,:],U[n,:].reshape((-1,1)))
            la[n] = pt.sum(self.theta_w * self.W * phi)
            lp[n] = pt.sum(self.logpi(U[n,:],range(self.P)))
        return(la + lp)

    def cond_prob(self,U,node,bias):
        """Returns the conditional probabity vector for node x, given U

        Args:
            U (ndarray): Current state of the network
            node (int): Number of node to get the conditional prov for
            bias (pt.tensor): (1,P) Log-Bias term for the node
        Returns:
            p (pt.tensor): (K,) vector of conditional probabilities for the node
        """
        x = np.arange(self.K)
        ind = np.where(self.W[node,:]>0) # Find all the neighbors for node x (precompute!)
        nb_x = U[ind] # Neighbors to node x
        same = np.equal(x,nb_x.reshape(-1,1))
        loglik = self.theta_w * pt.sum(same,dim=0) + bias
        return(pt.softmax(loglik,dim=0))

    def calculate_neighbours(self):
        """Calculate Neighbourhood
        """
        self.neighbours=np.empty((self.P,),dtype=object)
        for p in range(self.P):
            self.neighbours[p]= np.where(self.W[p,:]!=0)[0] # Find all the neighbors for node x (precompute!)


    def sample_gibbs(self,U0 = None, num_chains=None, bias = None, iter=5, return_hist=False, track=None):
        """Samples a number of gibbs-chains simulatenously
        using the same bias term

        Args:
            U0 (nd-array): Initial starting point (num_chains x P):
                Default None - and will be initialized by the bias term alone
            num_chains (int): If U0 not provided, number of chains to initialize
            bias (nd-array): Bias term (in log-probability (K,P)).
                 Defaults to None. Assumed to be the same for all the chains
            iter (int): Number of iterations. Defaults to 5.
            return_hist (bool): Return the history as a second return argument?
        Returns:
            U (nd-array): A (num_chains,P) array of integers
            Uhist (nd-array): Full sampling path - (iter,num_chains,P) array of integers (optional, only if return_all = True)
        Comments:
            This probably can be made more efficient by doing some of the sampling un bulk?
        """
        # Check for initialization of chains
        if U0 is None:
            prob = pt.softmax(bias,axis=0)
            U0 = sample_multinomial(prob,shape=(num_chains,self.K,self.P), compress=True)
        else:
            num_chains = U0.shape[0]

        if return_hist:
            if track is None:
                # Initilize array of full history of sample
                Uhist = pt.zeros((iter,num_chains,self.P),dtype=np.int16)
            else:
                Uhist = pt.zeros((iter,self.K))
        if not hasattr(self,'neighbours'):
            self.calculate_neighbours()

        # Start the chains
        U = U0
        u = pt.arange(self.K).reshape(self.K,1)

        for i in range(iter):
            if return_hist:
                if track is None:
                    Uhist[i,:,:]=U
                else:
                    for k in range(self.K):
                        Uhist[i,k]=pt.mean(U[:,track]==k)
            # Now loop over chains: This loop can maybe be replaced
            for p in np.arange(self.P-1,-1,-1):
                nb_u = U[:,self.neighbours[p]] # Neighbors to node x
                nb_u = nb_u.reshape(num_chains,1,-1)
                same = pt.eq(nb_u,u)
                loglik = self.theta_w * pt.sum(same,dim=2) + bias[:,p].reshape(1,self.K)
                prob = pt.softmax(loglik,dim=1)
                U[:,p]=sample_multinomial(prob,kdim=1,compress=True)
        if return_hist:
            return U,Uhist
        else:
            return U

    def sample(self,num_subj = 10,burnin=20):
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

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Unnormalized log-likelihood of the arrangement model for each subject. Note that this does not contain the partition function
        """
        numsubj, K, P = emloglik.shape
        bias = emloglik + self.logpi
        self.epos_U = pt.empty((numsubj,self.epos_numchains,P))
        for s in range(numsubj):
            self.epos_U[s,:,:] = self.sample_gibbs(num_chains=self.epos_numchains, bias = bias[s],iter=self.epos_iter)

        # Get Uhat from the sampled examples
        self.epos_Uhat = pt.empty((numsubj,self.K,self.P))
        for k in range(self.K):
            self.epos_Uhat[:,k,:]=pt.sum(self.epos_U==k,dim=1)/self.epos_numchains

        # Get the sufficient statistics for the potential functions
        self.epos_phihat = pt.zeros((numsubj,))
        for s in range(numsubj):
            phi = pt.zeros((self.P,self.P))
            for n in range(self.epos_numchains):
                phi=phi+np.equal(self.epos_U[s,n,:],self.epos_U[s,n,:].reshape((-1,1)))
            self.epos_phihat[s] = pt.sum(self.W * phi)/self.epos_numchains

        # The log likelihood for arrangement model p(U|theta_A) is not trackable-
        # So we can only return the unormalized potential functions
        if P>2:
            ll_Ae = self.theta_w * self.epos_phihat
            ll_Ap = pt.sum(self.epos_Uhat*self.logpi,dim=(1,2))
            if self.rem_red:
                Z = exp(self.logpi).sum(dim=0) # Local partition function
                ll_Ap = ll_Ap - pt.sum(log(Z))
            ll_A=ll_Ae+ll_Ap
        else:
            # Calculate Z in the case of P=2
            pp=exp(self.logpi[:,0]+self.logpi[:,1].reshape((-1,1))+np.eye(self.K)*self.theta_w)
            Z = pt.sum(pp) # full partition function
            ll_A = self.theta_w * self.epos_phihat + pt.sum(self.epos_Uhat*self.logpi,dim=(1,2)) - log(Z)
        return self.epos_Uhat,ll_A

    def Estep_new(self, emloglik):
        """ Positive phase of getting p(U|Y) for the spatial arrangement model
        Gets the expectations.

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Unnormalized log-likelihood of the arrangement model for each subject. Note that this does not contain the partition function
        """
        numsubj, K, P = emloglik.shape
        bias = emloglik + self.logpi
        self.epos_U[s,:,:] = self.sample_gibbs3(self.epos_numchains, bias,self.epos_iter)

        # Get Uhat from the sampled examples
        self.epos_Uhat = pt.empty((numsubj,self.K,self.P))
        for k in range(self.K):
            self.epos_Uhat[:,k,:]=pt.sum(self.epos_U==k,dim=1)/self.epos_numchains

        # Get the sufficient statistics for the potential functions
        self.epos_phihat = pt.zeros((numsubj,))
        for s in range(numsubj):
            phi = pt.zeros((self.P,self.P))
            for n in range(self.epos_numchains):
                phi=phi+np.equal(self.epos_U[s,n,:],self.epos_U[s,n,:].reshape((-1,1)))
            self.epos_phihat[s] = pt.sum(self.W * phi)/self.epos_numchains

        # The log likelihood for arrangement model p(U|theta_A) is not trackable-
        # So we can only return the unormalized potential functions
        if P>2:
            ll_Ae = self.theta_w * self.epos_phihat
            ll_Ap = pt.sum(self.epos_Uhat*self.logpi,dim=(1,2))
            if self.rem_red:
                Z = exp(self.logpi).sum(dim=0) # Local partition function
                ll_Ap = ll_Ap - pt.sum(log(Z))
            ll_A=ll_Ae+ll_Ap
        else:
            # Calculate Z in the case of P=2
            pp=exp(self.logpi[:,0]+self.logpi[:,1].reshape((-1,1))+np.eye(self.K)*self.theta_w)
            Z = pt.sum(pp) # full partition function
            ll_A = self.theta_w * self.epos_phihat + pt.sum(self.epos_Uhat*self.logpi,dim=(1,2)) - log(Z)
        return self.epos_Uhat,ll_A

    def eneg_sample(self,num_chains=None,iter=5):
        """Negative phase of the learning: uses persistent contrastive divergence
        with sampling from the spatial arrangement model (not clampled to data)
        Uses persistence across negative smapling steps
        """
        if self.eneg_U is None:
            self.eneg_U = self.sample_gibbs(num_chains=num_chains,
                    bias = self.logpi,iter=iter)
            # For tracking history: ,return_hist=True,track=0
        else:
            if (num_chains != self.eneg_U.shape[0]):
                raise NameError('num_chains needs to stay constant')
            self.eneg_U = self.sample_gibbs(self.eneg_U,
                    bias = self.logpi,iter=iter)

        # Get Uhat from the sampled examples
        self.eneg_Uhat = pt.empty((self.K,self.P))
        for k in range(self.K):
            self.eneg_Uhat[k,:]=pt.sum(self.eneg_U==k,dim=0)/num_chains

        # Get the sufficient statistics for the potential functions
        phi = pt.zeros((self.P,self.P))
        for n in range(num_chains):
            phi=phi+np.equal(self.eneg_U[n,:],self.eneg_U[n,:].reshape((-1,1)))
        self.eneg_phihat = pt.sum(self.W * phi)/num_chains
        return self.eneg_Uhat

    def epos_meanfield(self, emloglik,iter=5):
        """ Positive phase of getting p(U|Y) for the spatial arrangement model
        Using meanfield approximation. Note that this implementation is not accurate
        As it simply uses the Uhat from the other node

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        numsubj, K, P = emloglik.shape
        bias = emloglik + self.logpi
        self.epos_Uhat = pt.softmax(bias,dim=1)
        h = pt.empty((iter+1,))
        for i in range(iter):
            h[i]=self.epos_Uhat[0,0,0]
            for p in range(P): # Serial updating across all subjects
                nEng = self.theta_w*pt.sum(self.W[:,p]*self.epos_Uhat,dim=2)
                nEng = nEng + bias[:,:,p]
                self.epos_Uhat[:,:,p]=pt.softmax(nEng,dim=1)
        h[i+1]=self.epos_Uhat[0,0,0]
        return self.epos_Uhat, h

    def epos_jta(self, emloglik,order=None):
        """ This implements a closed-form Estep using a Junction-tree (Hugin)
        Algorithm. Uses a sequential elimination algorithm to result in a factor graph
        Uses the last node as root.

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        numsubj, K, P = emloglik.shape
        # Construct a linear Factor graph containing the factor (u1,u2),
        #(u2,u3),(u3,u4)....
        Phi = pt.zeros((numsubj,K,P)) # Messages for the forward pass
        Phis = pt.zeros((numsubj,K,P)) # Messages for the backward pass
        for s in range(numsubj):
            Psi = pt.zeros((P-1,K,K))
            # Initialize the factors
            Psi[0,:,:]=Psi[0,:,:]+self.logpi[:,0].reshape(-1,1)
            for p in range(P-1):
                Psi[p,:,:]=Psi[p,:,:]+self.logpi[:,p+1]
            Psi=Psi+np.eye(K)*self.theta_w
            pass
            # Now pass the evidence to the factors
            Psi[0,:,:]=Psi[0,:,:]+emloglik[s,:,0].reshape(-1,1)
            for p in range(P-1):
                Psi[p,:,:]=Psi[p,:,:]+emloglik[s,:,p+1]
            pass
            # Do the forward pass
            for p in np.arange(0,P-1):
                pp=exp(Psi[p,:,:])
                pp = pp / pt.sum(pp) # Normalize
                Phi[s,:,p+1]=np.log(pp.sum(dim=0))
                if p<P-2:
                    Psi[p+1,:,:]=Psi[p+1,:,:]+Phi[s,:,p+1].reshape(-1,1) # Update the next factors
            pass
            # Do the backwards pass
            Phis[s,:,P-1]=Phi[s,:,P-1]
            for p in np.arange(P-2,-1,-1):
                pp=exp(Psi[p,:,:])
                pp = pp / pt.sum(pp) # Normalize
                Phis[s,:,p]=np.log(pp.sum(dim=1))
                if p>0:
                    Psi[p-1,:,:]=Psi[p-1,:,:]+Phis[s,:,p]-Phi[s,:,p] # Update the factors
            pass

        return exp(Phis)

    def epos_ssa_chain(self, emloglik):
        """ This implements a closed-form Estep using the Schafer Shenoy algorithm on a chain
        This is identical to the JTA (Hugin) algorithm, but does not use the intermediate Phi factors.

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        numsubj, K, P = emloglik.shape
        # Construct a linear Factor graph containing the factor (u1,u2),
        #(u2,u3),(u3,u4)....
        self.epos_Uhat = pt.zeros((numsubj, K, P))
        self.epos_phihat = pt.zeros((numsubj,))
        for s in range(numsubj):
            Psi = pt.zeros((P-1,K,K)) # These are the original potentials
            muL = pt.zeros((P-1,K)) # Incoming messages from the left side
            muR = pt.zeros((P-1,K)) # Incoming messages from the right side
            # Initialize the factors
            Psi[0,:,:]=Psi[0,:,:]+self.logpi[:,0].reshape(-1,1)
            for p in range(P-1):
                Psi[p,:,:]=Psi[p,:,:]+self.logpi[:,p+1]
            Psi=Psi+np.eye(K)*self.theta_w
            # Now pass the evidence to the factors
            muL[0,:]=muL[0,:]+emloglik[s,:,0]
            for p in range(P-1):
                muR[p,:]=muR[p,:]+emloglik[s,:,p+1]
            # Do the forward pass
            for p in np.arange(0,P-2):
                pp=exp(Psi[p,:,:]+muL[p,:].reshape(-1,1)+muR[p,:])
                pp = pp / pt.sum(pp) # Normalize
                muL[p+1,:]=np.log(pp.sum(dim=0))
            # Do the backwards pass
            for p in np.arange(P-2,0,-1):
                pp=exp(Psi[p,:,:]+muR[p,:])
                pp = pp / pt.sum(pp) # Normalize
                muR[p-1,:]=muR[p-1,:]+np.log(pp.sum(dim=1))
            pass
            # Finally get the normalized potential
            for p in np.arange(0,P-1):
                pp=exp(Psi[p,:,:]+muL[p,:].reshape(-1,1)+muR[p,:])
                pp =pp / pt.sum(pp)
                # For the sufficent statistics, we need to consider each potential twice in the
                # Sum of the log-liklihoof sum_{i} sum_{j \neq i} <u_i u_j>
                self.epos_phihat[s] = self.epos_phihat[s] + 2 * pp.trace()
                if p==0:
                    self.epos_Uhat[s,:,0]=pp.sum(dim=1)
                self.epos_Uhat[s,:,p+1]=pp.sum(dim=0)
        # Get the postive likelihood: Could we use standardization here?
        ll_Ae = self.theta_w * self.epos_phihat
        ll_Ap = pt.sum(self.epos_Uhat*self.logpi,dim=(1,2))
        if self.rem_red:
            Z = exp(self.logpi).sum(dim=0) # Local partition function
            ll_Ap = ll_Ap - pt.sum(log(Z))
        ll_A=ll_Ae+ll_Ap

        return self.epos_Uhat, ll_A

    def eneg_ssa_chain(self):
        """ This implements a closed-form Estep using the Schafer Shenoy algorithm.
        This is identical to the JTA (Hugin) algorithm, but does not use the intermediate Phi factors.
        Calculates the expectation under the current verson of the model, without input data

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        # Construct a linear Factor graph containing the factor (u1,u2),
        #(u2,u3),(u3,u4)....
        P = self.P
        K = self.K
        Psi = pt.zeros((P-1,K,K)) # These are the original potentials
        muL = pt.zeros((P-1,K)) # Incoming messages from the left side
        muR = pt.zeros((P-1,K)) # Incoming messages from the right side
        self.eneg_Uhat = pt.zeros((K, P))
        self.eneg_phihat = 0
        # Initialize the factors
        Psi[0,:,:]=Psi[0,:,:]+self.logpi[:,0].reshape(-1,1)
        for p in range(P-1):
            Psi[p,:,:]=Psi[p,:,:]+self.logpi[:,p+1]
        Psi=Psi+np.eye(K)*self.theta_w
        # Do the forward pass
        for p in np.arange(0,P-2):
            pp=exp(Psi[p,:,:]+muL[p,:].reshape(-1,1))
            pp = pp / pt.sum(pp) # Normalize
            muL[p+1,:]=np.log(pp.sum(dim=0))
        # Do the backwards pass
        for p in np.arange(P-2,0,-1):
            pp=exp(Psi[p,:,:]+muR[p,:])
            pp = pp / pt.sum(pp) # Normalize
            muR[p-1,:]=muR[p-1,:]+np.log(pp.sum(dim=1))
        pass
        # Finally get the normalized potential
        for p in np.arange(0,P-1):
            pp=exp(Psi[p,:,:]+muL[p,:].reshape(-1,1)+muR[p,:])
            pp =pp / pt.sum(pp)
            self.eneg_phihat = self.eneg_phihat + 2 * pp.trace()
            if p==0:
                self.eneg_Uhat[:,0]=pp.sum(dim=1)
            self.eneg_Uhat[:,p+1]=pp.sum(dim=0)
        return self.eneg_Uhat


    def epos_ssa(self, emloglik, update_order=None):
        """ Implements general Shenoy-Shaefer algorithm
        Constructing a general clique tree from the connectivity matrix

        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        numsubj, K, P = emloglik.shape
        self.epos_Uhat = pt.zeros((numsubj, K, P))
        self.epos_phihat = pt.zeros((numsubj,))

        # Construct general factor graph containing the factor (u1,u2),
        if hasattr(self,'inE'):
            inP = self.inE
            outP = self.ouE
            num_edge = len(inP)
        else:
            num_edge = (self.W>0).sum()
            inP,outP = np.where(self.W>0)
        if update_order is None:
            update_order = np.tile(np.arange(len(inP)),2)
        for s in range(numsubj):
            muIn = pt.zeros((K,num_edge)) # Incoming messages from nodes to Edges
            muOut = pt.zeros((K,num_edge)) # Outgoing messages from Edges to nodes
            mu = pt.zeros((K,self.P)) # Message on each potential
            phi_trace = pt.zeros((num_edge,)) # Outgoing messages from Edges to nodes
            bias = emloglik[s,:,:]+self.logpi
            for e in update_order:
                # Update the state of the input node
                muIn[:,e] = bias[:,inP[e]] + pt.sum(muOut[:,(inP[e]==outP) & (inP !=outP[e])],dim=1)
                E = np.eye(K)*self.theta_w + muIn[:,e].reshape(-1,1)
                pp = exp(E)
                pp = pp/ pp.sum()
                muOut[:,e]=log(pp.sum(dim=0))
            # Calcululate sufficient stats: This could be cut in half...
            for e in range(num_edge):
                opp = (inP[e]==outP) & (outP[e]==inP)
                E = np.eye(K)*self.theta_w + muIn[:,e].reshape(-1,1) + muIn[:,opp].reshape(1,-1)
                pp = exp(E)
                pp = pp/ pp.sum()
                self.epos_Uhat[s,:,outP[e]]=pp.sum(dim=0)
                phi_trace[e]=pp.trace()
                self.epos_phihat[s] = self.epos_phihat[s] + pp.trace()
        # Get the postive likelihood: Could we use standardization here?
        ll_Ae = self.theta_w * self.epos_phihat
        ll_Ap = pt.sum(self.epos_Uhat*self.logpi,dim=(1,2))
        if self.rem_red:
            Z = exp(self.logpi).sum(dim=0) # Local partition function
            ll_Ap = ll_Ap - pt.sum(log(Z))
        ll_A=ll_Ae+ll_Ap

        return self.epos_Uhat, ll_A

    def eneg_ssa(self, update_order=None):
        """ Implements general Shenoy-Shaefer algorithm: Negative step
        Constructing a general clique tree from the connectivity matrix

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        P = self.P
        K = self.K

        self.eneg_Uhat = pt.zeros(K, P)
        self.eneg_phihat = 0

        # Construct general factor graph containing the factor (u1,u2),
        if hasattr(self,'inE'):
            inP = self.inE
            outP = self.ouE
            num_edge = len(inP)
        else:
            num_edge = (self.W>0).sum()
            inP,outP = np.where(self.W>0)
        if update_order is None:
            update_order=np.arange(num_edge)
        muIn = pt.zeros(K,num_edge) # Incoming messages from nodes to Edges
        muOut = pt.zeros(K,num_edge) # Outgoing messages from Edges to nodes
        bias = self.logpi
        for e in update_order:
            # Update the state of the input node
            muIn[:,e] = bias[:,inP[e]] + pt.sum(muOut[:,(inP[e]==outP) & (inP !=outP[e])],dim=1)
            E = np.eye(K)*self.theta_w + muIn[:,e].reshape(-1,1)
            pp = exp(E)
            pp = pp/ pp.sum()
            muOut[:,e]=log(pp.sum(dim=0))
        # Calcululate sufficient stats: This could be cut in half...
        for e in range(num_edge):
            opp = (inP[e]==outP) & (outP[e]==inP)
            E = np.eye(K)*self.theta_w + muIn[:,e].reshape(-1,1) + muIn[:,opp].reshape(1,-1)
            pp = exp(E)
            pp = pp/ pp.sum()
            self.eneg_Uhat[:,outP[e]]=pp.sum(dim=0)
            self.eneg_phihat = self.eneg_phihat + pp.trace()
        return self.eneg_Uhat


    def Mstep(self,stepsize = 0.1):
        """ Gradient update for SML or CD algorithm
        Parameters:
            stepsize (float):
                Stepsize for the update of the parameters
        """
        # Update logpi
        if self.rem_red:
            # The - pi can be dropped here as we follow the difference between pos and neg anyway
            gradpos_logpi = self.epos_Uhat[:,:-1,:].mean(dim=0)
            gradneg_logpi = self.eneg_Uhat[:-1,:]
            self.logpi[:-1,:] = self.logpi[:-1,:] + stepsize * (gradpos_logpi - gradneg_logpi)
        else:
            gradpos_logpi = self.epos_Uhat.mean(dim=0)
            gradneg_logpi = self.eneg_Uhat
            self.logpi = self.logpi + stepsize * (gradpos_logpi - gradneg_logpi)
        if self.fit_theta_w:
            grad_theta_w = self.epos_phihat.mean() - self.eneg_phihat
            self.theta_w = self.theta_w + stepsize* grad_theta_w
        return

class PottsModelDuo(PottsModel):
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states, but only 2 nodes
    Closed-form solututions for checking the approximations
    """
    def __init__(self,K=3,remove_redundancy=True):
        W = pt.tensor([[0,1],[1,0]])
        super().__init__(W,K,remove_redundancy)

    def Estep(self,emloglik,return_joint = False):
        numsubj, K, P = emloglik.shape
        logq = emloglik + self.logpi
        self.estep_Uhat = pt.empty((numsubj,K,P))
        Uhat2 = pt.empty((numsubj,K,K))
        for s in range(numsubj):
            pp=exp(logq[s,:,0]+logq[s,:,1].reshape((-1,1))+np.eye(self.K)*self.theta_w)
            Z = pt.sum(pp) # full partition function
            pp = pp/Z
            self.estep_Uhat[s] = np.c_[pp.sum(dim=0),pp.sum(dim=1)]
            Uhat2[s]=pp

        # The log likelihood for arrangement model p(U|theta_A) is sum_i sum_K Uhat_(K)*log pi_i(K)
        p,pp = self.marginal_prob(return_joint=True)
        ll_A = pt.sum(Uhat2 * log(pp),dim=(1,2))
        if return_joint:
            return self.estep_Uhat, ll_A,Uhat2
        else:
            return self.estep_Uhat, ll_A

    def marginal_prob(self,return_joint=False):
        pp=exp(self.logpi[:,0]+self.logpi[:,1].reshape((-1,1))+np.eye(self.K)*self.theta_w)
        Z = pt.sum(pp) # full partition function
        pp=pp/Z
        p1 = pp.sum(dim=0)
        p2 = pp.sum(dim=1)
        if return_joint:
            return np.c_[p1,p2],pp
        else:
            return np.c_[p1,p2]



def loglik2prob(loglik,dim=0):
    """Safe transformation and normalization of
    logliklihood

    Args:
        loglik (ndarray): Log likelihood (not normalized)
        axis (int): Number of axis (or axes), along which the probability is being standardized
    Returns:
        prob (ndarray): Probability
    """
    if (dim==0):
        ml,_=pt.max(loglik,dim=0)
        loglik = loglik-ml+10
        prob = np.exp(loglik)
        prob = prob/pt.sum(prob,dim=0)
    else:
        a = pt.tensor(loglik.shape)
        a[dim]=1 # Insert singleton dimension
        ml,_=pt.max(loglik,dim=0)
        loglik = loglik-ml.reshape(a)+10
        prob = pt.exp(loglik)
        prob = prob/pt.sum(prob,dim=1).reshape(a)
    return prob

def sample_multinomial_old(p_u,N=1,compress=False):
    """Samples from a multinomial distribution
    Fast smpling from matrix probability without looping

    Args:
        p_u (tensor): 1d (K), 2d- (KxP) or 3d-tensor (NxKxP)
        N ([int]): If provided for 1d-sample give
        compress: Return as int (True) or indicator (false)
    Returns: Samples either in indicator coding (compress = False)
            or as ints (compress = False)
    """
    if p_u.dim() == 1:
        K = p_u.shape[0]
        r = pt.empty(1,N).uniform_(0,1)
        cdf_v = p_u.cumsum(dim=0)
        sample = (r < cdf_v).float()
        for k in np.arange(K-1,0,-1):
            sample[k,:]-=sample[k-1,:]
        dim =0
    elif p_u.dim() == 2:
        K,P = p_u.shape
        r = pt.empty(N,1,P).uniform_(0,1)
        cdf_v = p_u.cumsum(dim=0)
        sample = (r < cdf_v).float()
        for k in np.arange(K-1,0,-1):
            sample[:,k,:]-=sample[:,k-1,:]
        dim = 1
    elif p_u.dim() == 3:
        N,K,P = p_u.shape
        r = pt.empty(N,1,P).uniform_(0,1)
        cdf_v = p_u.cumsum(dim=1)
        sample = (r < cdf_v).float()
        for k in np.arange(K-1,0,-1):
            sample[:,k,:]-=sample[:,k-1,:]
        dim = 1
    if compress:
        return sample.argmax(dim=dim)
    else:
        return sample

