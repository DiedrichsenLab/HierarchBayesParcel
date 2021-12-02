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
    def __init__(self, K=3, P=100, spatial_specific=False):
        super().__init__(K, P)
        # In this model, the spatially independent arrangement has
        # two options, spatially uniformed and spatially specific prior
        self.spatial_specific = spatial_specific
        if spatial_specific:
            pi = np.ones((K, P)) / K
        else:
            pi = np.ones((K, 1)) / K
        self.logpi = np.log(pi)
        self.nparams = self.logpi.size

    def get_params(self):
        """ Get the parameters (log-pi) for the Arrangement model
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        return self.logpi.flatten()

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        if self.spatial_specific:
            self.logpi = theta.reshape((self.K, self.P))
        else:
            self.logpi = theta.reshape((self.K, 1))

    def Estep(self, emloglik):
        """ Estep for the spatial arrangement model

        Parameters:
            emloglik (np.array):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
        Returns:
            Uhat (np.array):
                posterior p(U|Y) a numsubj x K x P matrix
        """
        numsubj, K, P = emloglik.shape
        logq = emloglik + self.logpi
        Uhat = np.exp(np.apply_along_axis(lambda x: x - np.min(x), 1, logq))
        Uhat = Uhat / np.sum(Uhat, axis=1).reshape((numsubj, 1, P))

        # The log likelihood for arrangement model p(U|theta_A) is sum_i sum_K Uhat_(K)*log pi_i(K)
        ll_A = Uhat * self.logpi
        return Uhat, ll_A

    def Mstep(self, Uhat):
        """ M-step for the spatial arrangement model
            Update the pi for arrangement model
        """
        pi = np.mean(Uhat, axis=0)  # Averarging over subjects
        if not self.spatial_specific:
            pi = pi.mean(axis=1).reshape(-1, 1)
        self.logpi = log(pi)

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
        pi = np.exp(self.logpi)
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
    def __init__(self,W,K=3):
        self.W = W
        self.K = K # Number of states
        self.P = W.shape[0]
        pi = np.ones((K, self.P)) / K
        self.logpi = np.log(pi)
        self.theta_w = 1 # Weight of the neighborhood relation - inverse temperature param
        self.nparams = self.logpi.size + 1

    def get_params(self):
        """ Get the parameters (log-pi) for the Arrangement model
        Returns:
            theta (1-d np.array): Vectorized version of parameters
        """
        return np.concatenate([self.logpi.flatten(), log(self.theta_w)])

    def set_params(self,theta):
        """Sets the parameters from a vector
        """
        self.logpi = theta[:-1].reshape((self.K, self.P))
        self.theta_w = exp(theta[-1])

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

    def loglike(self,Y):
        """Returns the energy term of the network
        up to a constant the loglikelihood of the state
        Params:
            Y ([np-array]): 1d or 2d array of network states
        """
        phi=self.potential(Y)
        l = self.theta_w @ phi
        return(l)

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
        loglik = loglik +self.logMu[:,node] + bias
        p = np.exp(loglik)
        p = p / np.sum(p)
        return(p)

    def sample_gibbs(self,U0 = None, num_chains=None, bias = None, iter=5, return_all=False):
        """Samples a number of gibbs-chains simulatenously
        using the same bias term

        Args:
            U0 (nd-array): Initial starting point (num_chains x P): Defaults to None.
            bias (nd-array): Bias term (in log-probability for each of the Pi
                             (K,P). Defaults to None.
            iter (int): Number of iterations. Defaults to 5.
        Returns:
            U (nd-array): Either a (num_chains )
        Comments:
            This probably can be made more efficient by doing some of the sampling un bulk?
        """
        # Check for initialization of chains
        if U0 is None:
            for p in range(self.P):
                U0[:,p] = np.random.choice(self.K,p = exp(self.logpi[:,p]),size = (num_chains,))
        else:
            num_chains = U0.shape[0]

        if return_all:
            # Initilize array of full history of sample
            Uhist = np.zeros((iter+1,num_chains,self.P))

        # Start the chains
        U = U0
        for i in range(iter):
            if return_all:
                Uhist[i,:,:]=U
                # Now loop over chains: This loop can maybe be replaced
                for c in range(num_chains):
                    for p in range(self.P):
                        prob = self.cond_prob(U,p,bias=bias[:,p])
                        U[c,p]=np.random.choice(self.K,p=prob)
        return(U)

    def sample_new(self,num_subj = 10,burnin=20):
        """Samples new subjects from prior
        Args:
            num_subj (int): Number of subjects. Defaults to 10.
            burnin (int): Number of . Defaults to 20.
        Returns:
            U (ndarray): Labels for all subjects (numsubj x P) array
        """
        U = np.zeros((num_subj,self.P))
        for i in range(num_subj):
            Us = self.sample_gibbs(bias = self.logpi, iter = burnin)
            U[i,:]=Us[-1,:]
        return U

    def Estep(self,emloglik):
        """Estep using Gibbs sampling

        """

