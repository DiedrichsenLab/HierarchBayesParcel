# Example Models
import os  # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
from decimal import Decimal
from numpy import exp,log,sqrt

import sys
sys.path.insert(0, "D:/python_workspace/")


class ArrangementModel:
    """Abstract arrangement model
    """
    def __init__(self, K, P):
        self.K = K  # Number of states
        self.P = P  # Number of nodes

    def get_params(self):
        """Returns the vectorized verion of the parameters
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        pass

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        pass


class ArrangeIndependent(ArrangementModel):
    """ Arrangement model for spatially independent assignment
        Either with a spatially uniform prior
        or a spatially-specific prior. Pi is saved in form of log-pi.
    """
    def __init__(self, K=3, P=100, spatial_specific=False, remove_redundancy=True):
        super().__init__(K, P)
        # In this model, the spatially independent arrangement has
        # two options, spatially uniformed and spatially specific prior
        # If 
        self.spatial_specific = spatial_specific
        if spatial_specific:
            pi = np.ones((K, P)) / K
        else:
            pi = np.ones((K, 1)) / K
        self.logpi = np.log(pi)
        # Remove redundancy in parametrization 
        self.rem_red = remove_redundancy
        if self.rem_red: 
            self.logpi = self.logpi - self.logpi[-1, :]
            self.nparams = self.logpi.size - self.logpi[-1, :].size
        else: 
            self.nparams = self.logpi.size

    def get_params(self):
        """ Get the parameters (log-pi) for the Arrangement model
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        if self.rem_red:
            return self.logpi[:-1, :].flatten()
        else:
            return self.logpi.flatten()

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        if self.spatial_specific:
            P = self.P
        else:
            P = 1
        if self.rem_red:
            self.logpi[:-1,:] = theta.reshape((self.K-1, P))
        else: 
            self.logpi = theta.reshape((self.K-1, P))

    def Estep(self, emloglik):
        """ Estep for the spatial arrangement model

        Parameters:
            emloglik (np.array):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (np.array):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (np.array):
                Expected log-liklihood of the arrangement model
        """
        numsubj, K, P = emloglik.shape
        logq = emloglik + self.logpi
        self.estep_Uhat = np.empty(logq.shape)
        for s in range(numsubj):
            self.estep_Uhat[s] = loglik2prob(logq[s])

        # The log likelihood for arrangement model p(U|theta_A) is sum_i sum_K Uhat_(K)*log pi_i(K)

        ll_A = np.sum(self.estep_Uhat * self.logpi,axis=(1,2))
        return self.estep_Uhat, ll_A

    def Mstep(self):
        """ M-step for the spatial arrangement model
            Update the pi for arrangement model
            uses the epos_Uhat statistic that is put away from the last e-step.  
        
        Parameters:
        Returns: 
        """
        pi = np.mean(self.estep_Uhat, axis=0)  # Averarging over subjects
        if not self.spatial_specific:
            pi = pi.mean(axis=1).reshape(-1, 1)
        self.logpi = log(pi)
        if self.rem_red:
            self.logpi = self.logpi-self.logpi[-1,:]

    def sample(self, num_subj=10):
        """
        Samples a number of subjects from the prior.
        In this i.i.d arrangement model we assume each node has
        no relation with other nodes, which means it equals to
        sample from the prior pi.

        :param num_subj: the number of subjects to sample
        :return: the sampled data for subjects
        """
        U = np.zeros((num_subj, self.P))
        pi = loglik2prob(self.logpi)
        for i in range(num_subj):
            for p in range(self.P):
                if self.spatial_specific:
                    np.testing.assert_array_equal(self.K, pi.shape[1])
                    U[i, p] = np.random.choice(self.K, p=pi[i])
                else:
                    U[i, p] = np.random.choice(self.K, p=pi.reshape(-1))
        return U


class PottsModel(ArrangementModel):
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states
    Potential function is determined by linkages
    parameterization is joint between all linkages, although it could be split
    into different parameter functions
    """
    def __init__(self,W,K=3,remove_redundancy=True):
        self.W = W
        self.K = K # Number of states
        self.P = W.shape[0]
        self.theta_w = 1 # Weight of the neighborhood relation - inverse temperature param
        self.rem_red = remove_redundancy
        pi = np.ones((K, self.P)) / K
        self.logpi = np.log(pi)
        if remove_redundancy:
            self.logpi = self.logpi - self.logpi[-1,:]
            self.nparams = self.logpi[:-1,:].size + 1
        else:
            self.nparams = self.logpi.size + 1
        # Inference parameters for persistence CD alogrithm via sampling
        self.epos_iter = 3
        self.epos_numchains = 20 # Chains per subject
        self.epos_U = None
        self.eneg_iter = 3
        self.eneg_numchains = 20 # Overall number of chains
        self.eneg_U = None
        self.fit_theta_w = True # Update smoothing parameter in Mstep 
 
    def get_params(self):
        """ Get the parameters (log-pi) for the Arrangement model
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        if self.rem_red:
            theta = np.concatenate([self.logpi[:-1,:].flatten(), np.array([self.theta_w])])
        else:
            theta = np.concatenate([self.logpi.flatten(), np.array([self.theta_w])])
        return theta

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        if self.rem_red:
            self.logpi[:-1,:] = theta[:-1].reshape((self.K-1, self.P))
        else:
            self.logpi = theta[:-1].reshape((self.K, self.P))
        self.theta_w = exp(theta[-1])

    def get_param_names(self):
        """Returns a list of strings referring to the parameter names
        in the theta vector
        """
        if self.rem_red:
            n = self.K-1
        else:
            n = self.K
        names = []
        return names

    def random_smooth_pi(self, Dist, theta_mu=1,centroids=None):
        """
            Defines pi (prior over parcels) using a Ising model with K centroids
            Needs the Distance matrix to define the prior probability
        """
        if centroids is None:
            centroids = np.random.choice(self.P,(self.K,))
        d2 = Dist[centroids,:]**2
        pi = np.exp(-d2/theta_mu)
        pi = pi / pi.sum(axis=0)
        self.logpi = np.log(pi)
        if self.rem_red:
            self.logpi = self.logpi - self.logpi[-1,:]


    def potential(self,y):
        """
        returns the potential functions for the log-linear form of the model
        """
        if y.ndim==1:
            y=y.reshape((-1,1))
        # Potential on states
        N = y.shape[0] # Number of observations
        phi = np.zeros((self.numparam,N))
        for i in range(N):
           S = np.equal(y[i,:],y[i,:].reshape((-1,1)))
           phi[0,i]=np.sum(S*self.W)
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
        la = np.empty((N,))
        lp = np.empty((N,))
        for n in range(N):
            phi=np.equal(U[n,:],U[n,:].reshape((-1,1)))
            la[n] = np.sum(self.theta_w * self.W * phi)
            lp[n] = np.sum(self.logpi(U[n,:],range(self.P)))
        return(la + lp)

    def cond_prob(self,U,node,bias):
        """Returns the conditional probabity vector for node x, given U

        Args:
            U (ndarray): Current state of the network
            node (int): Number of node to get the conditional prov for
            bias (ndarray): (1,P) Log-Bias term for the node
        Returns:
            p (ndarray): (K,) vector of conditional probabilities for the node
        """
        x = np.arange(self.K)
        ind = np.where(self.W[node,:]>0) # Find all the neighbors for node x (precompute!)
        nb_x = U[ind] # Neighbors to node x
        same = np.equal(x,nb_x.reshape(-1,1))
        loglik = self.theta_w * np.sum(same,axis=0) + bias
        return(loglik2prob(loglik))

    def sample_gibbs(self,U0 = None, num_chains=None, bias = None, iter=5, return_hist=False):
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
            U0 = np.empty((num_chains,self.P))
            prob = loglik2prob(bias)
            for p in range(self.P):
                U0[:,p] = np.random.choice(self.K,p = prob[:,p],size = (num_chains,))
        else:
            num_chains = U0.shape[0]

        if return_hist:
            # Initilize array of full history of sample
            Uhist = np.zeros((iter+1,num_chains,self.P))

        # Start the chains
        U = U0
        for i in range(iter):
            if return_hist:
                Uhist[i,:,:]=U
            # Now loop over chains: This loop can maybe be replaced
            for c in range(num_chains):
                for p in range(self.P):
                    prob = self.cond_prob(U[c,:],p,bias=bias[:,p])
                    U[c,p]=np.random.choice(self.K,p=prob)
        if return_hist:
            Uhist[-1,:,:]=U
            return U,Uhist
        else:
            return U

    def sample(self,num_subj = 10,burnin=20):
        """Samples new subjects from prior: wrapper for sample_gibbs
        Args:
            num_subj (int): Number of subjects. Defaults to 10.
            burnin (int): Number of . Defaults to 20.
        Returns:
            U (ndarray): Labels for all subjects (numsubj x P) array
        """
        U = self.sample_gibbs(bias = self.logpi, iter = burnin,
            num_chains = num_subj)
        return U

    def epos_sample(self, emloglik):
        """ Positive phase of getting p(U|Y) for the spatial arrangement model
        Gets the expectations.

        Parameters:
            emloglik (np.array):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (np.array):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (np.array):
                Unnormalized log-likelihood of the arrangement model for each subject. Note that this does not contain the partition function
        """
        numsubj, K, P = emloglik.shape
        bias = emloglik + self.logpi
        if self.epos_U is None: # No current state of MC chains
            self.epos_U = np.empty((numsubj,self.epos_numchains,P))
            for s in range(numsubj):
                self.epos_U[s,:,:] = self.sample_gibbs(num_chains=self.epos_numchains,
                    bias = bias[s],iter=self.epos_iter)
        else:
            for s in range(numsubj):
                self.epos_U[s,:,:] = self.sample_gibbs(self.epos_U[s],
                    bias = bias[s],iter=self.epos_iter)

        # Get Uhat from the sampled examples
        self.epos_Uhat = np.empty((numsubj,self.K,self.P))
        for k in range(self.K):
            self.epos_Uhat[:,k,:]=np.sum(self.epos_U==k,axis=1)/self.epos_numchains

        # Get the sufficient statistics for the potential functions
        self.epos_phihat = np.zeros((numsubj,))
        for s in range(numsubj):
            phi = np.zeros((self.P,self.P))
            for n in range(self.epos_numchains):
                phi=phi+np.equal(self.epos_U[s,n,:],self.epos_U[s,n,:].reshape((-1,1)))
            self.epos_phihat[s] = np.sum(self.W * phi)/self.epos_numchains

        # The log likelihood for arrangement model p(U|theta_A) is not trackable-
        # So we can only return the unormalized potential functions
        if P>2:
            ll_Ae = self.theta_w * self.epos_phihat
            ll_Ap = np.sum(self.epos_Uhat*self.logpi,axis=(1,2))
            ll_A=ll_Ae+ll_Ap
        else:
            # Calculate Z in the case of P=2
            pp=exp(self.logpi[:,0]+self.logpi[:,1].reshape((-1,1))+np.eye(self.K)*self.theta_w)
            Z = np.sum(pp)
            ll_A = self.theta_w * self.epos_phihat + np.sum(self.epos_Uhat*self.logpi,axis=(1,2)) - log(Z)
        return self.epos_Uhat,ll_A

    def eneg_sample(self):
        """Negative phase of the learning: uses persistent contrastive divergence
        with sampling from the spatial arrangement model (not clampled to data)
        Uses persistence across negative smapling steps
        """
        if self.eneg_U is None:
            self.eneg_U = self.sample_gibbs(num_chains=self.eneg_numchains,
                    bias = self.logpi,iter=self.eneg_iter)
        else:
            self.mstep_state = self.sample_gibbs(self.eneg_U,
                    bias = self.logpi,iter=self.eneg_iter)

        # Get Uhat from the sampled examples
        self.eneg_Uhat = np.empty((self.K,self.P))
        for k in range(self.K):
            self.eneg_Uhat[k,:]=np.sum(self.eneg_U==k,axis=0)/self.eneg_numchains

        # Get the sufficient statistics for the potential functions
        phi = np.zeros((self.P,self.P))
        for n in range(self.eneg_numchains):
            phi=phi+np.equal(self.eneg_U[n,:],self.eneg_U[n,:].reshape((-1,1)))
        self.eneg_phihat = np.sum(self.W * phi)/self.eneg_numchains
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
            gradpos_logpi = self.epos_Uhat[:,:-1,:].mean(axis=0)
            gradneg_logpi = self.eneg_Uhat[:-1,:]
            self.logpi[:-1,:] = self.logpi[:-1,:] + stepsize * (gradpos_logpi - gradneg_logpi)
        else:
            gradpos_logpi = self.epos_Uhat.mean(axis=0)
            gradneg_logpi = self.eneg_Uhat
            self.logpi = self.logpi + stepsize * (gradpos_logpi - gradneg_logpi)
        if self.fit_theta_w:
            grad_theta_w = self.epos_phihat.mean() - self.eneg_phihat
            self.theta_w = self.theta_w + stepsize* grad_theta_w
        return


def loglik2prob(loglik):
    """Safe transformation and normalization of
    logliklihood (along axis 0)

    Args:
        loglik (ndarray): Log likelihood (not normalized)
        axis (int): Number of axis (or axes), along which the probability is being standardized
    Returns:
        prob (ndarray): Probability
    """
    loglik = loglik-np.max(loglik,axis=0)+10
    prob = np.exp(loglik)
    prob = prob/np.sum(prob,axis=0)
    return prob

