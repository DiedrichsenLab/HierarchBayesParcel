# Example Models
import os  # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
from decimal import Decimal
from torch import exp,log,sqrt
from generativeMRF.model import Model
import torch as pt

import sys

class ArrangementModel(Model):
    """Abstract arrangement model
    """
    def __init__(self, K, P):
        self.K = K  # Number of states
        self.P = P  # Number of nodes

    def clear(self): 
        """Removes temporary fields from 
        object: Use this before saving with pickle to 
        redice memory size
        """
        if hasattr(self,'estep_Uhat'):
            delattr(self,'estep_Uhat')

class ArrangeIndependent(ArrangementModel):
    """ Arrangement model for spatially independent assignment
        Either with a spatially uniform prior
        or a spatially-specific prior. Pi is saved in form of log-pi.
    """
    def __init__(self, K=3, P=100,
                 spatial_specific=False,
                 remove_redundancy=True):
        super().__init__(K, P)
        # In this model, the spatially independent arrangement has
        # two options, spatially uniformed and spatially specific prior
        # If
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

    def random_params(self):
        """ Sets prior parameters to random starting values 
        """
        self.logpi = pt.normal(0,1,size=self.logpi.shape)

    def Estep(self, emloglik, gather_ss=True):
        """ Estep for the spatial arrangement model

        Parameters:
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
        if type(emloglik) is np.ndarray:
            emloglik=pt.tensor(emloglik,dtype=pt.get_default_dtype())
        logq = emloglik + self.logpi
        Uhat = pt.softmax(logq,dim=1)
        if gather_ss:
            self.estep_Uhat = Uhat
        # The log likelihood for arrangement model p(U|theta_A) is sum_i sum_K Uhat_(K)*log pi_i(K)
        pi = pt.softmax(self.logpi,dim=0)
        lpi = pt.nan_to_num(pt.log(pi),neginf=0) # Prevent underflow 
        ll_A = pt.sum(Uhat * lpi)
        if pt.isnan(ll_A):
            raise(NameError('likelihood is nan'))
        return Uhat, ll_A

    def Mstep(self):
        """ M-step for the spatial arrangement model
            Update the pi for arrangement model
            uses the epos_Uhat statistic that is put away from the last e-step.

        Parameters:
        Returns:
        """
        pi = pt.mean(self.estep_Uhat, dim=0)  # Averarging over subjects
        if not self.spatial_specific:
            pi = pi.mean(dim=1).reshape(-1, 1)
        self.logpi = log(pi)
        if self.rem_red:
            self.logpi = self.logpi-self.logpi[-1,:]
        self.logpi=pt.nan_to_num(self.logpi)

    def sample(self, num_subj=10):
        """
        Samples a number of subjects from the prior.
        In this i.i.d arrangement model we assume each node has
        no relation with other nodes, which means it equals to
        sample from the prior pi.

        :param num_subj: the number of subjects to sample
        :return: the sampled data for subjects
        """
        U = pt.zeros(num_subj, self.P)
        pi = pt.softmax(self.logpi,dim=0)
        for i in range(num_subj):
            if self.spatial_specific:
                U = sample_multinomial(pi,shape=(num_subj,self.K,self.P),compress=True)
            else:
                pi=pi.expand(self.K,self.P)
                U = sample_multinomial(pi,shape=(num_subj,self.K,self.P),compress=True)
        return U

    def marginal_prob(self):
        """Returns marginal probabilty for every node under the model
        Returns: p[] marginal probability under the model
        """
        return pt.softmax(self.logpi,dim=0)


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
        pi = pt.ones((K, self.P)) / K
        self.logpi = np.log(pi)
        if remove_redundancy:
            self.logpi = self.logpi - self.logpi[-1,:]
        # Inference parameters for persistence CD alogrithm via sampling
        self.epos_U = None
        self.eneg_U = None
        self.fit_theta_w = True # Update smoothing parameter in Mstep
        self.update_order=None
        self.nparams = 10
        self.set_param_list(['logpi', 'theta_w'])

    def random_smooth_pi(self, Dist, theta_mu=1,centroids=None):
        """
            Defines pi (prior over parcels) using a Ising model with K centroids
            Needs the Distance matrix to define the prior probability
        """
        if centroids is None:
            centroids = np.random.choice(self.P,(self.K,))
        d2 = Dist[centroids,:]**2
        pi = pt.exp(-d2/theta_mu)
        pi = pi / pi.sum(dim=0)
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
            # Now loop over noes for all chains at the same tim
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


class mpRBM(ArrangementModel):
    """multinomial (categorial) restricted Boltzman machine
    for learning of brain parcellations for probabilistic input
    Uses Contrastive-Divergence k for learning
    Outer nodes (U):
        The outer (most peripheral nodes) are
        categorical with K possible categories.
        There are three different representations:
        a) N x nv: integers between 0 and K-1 (u)
        b) N x K x nv : indicator variables or probabilities (U)
        c) N x (K * nv):  Vectorized version of b- with all nodes of category 1 first, etc,
        If not otherwise noted, we will use presentation b)
    Hidden nodes (h):
        In this version we will use binary hidden nodes - so to get the same capacity as a mmRBM, one would need to set the number of hidden nodes to nh
    """
    def __init__(self, K, P, nh):
        super().__init__(K, P)
        self.K = K
        self.P = P
        self.nh = nh
        self.W = pt.randn(nh,P*K)
        self.bh = pt.randn(nh)
        self.bu = pt.randn(K,P)
        self.eneg_U = None
        self.Etype = 'prob'
        self.alpha = 0.01
        self.epos_iter = 5
        self.set_param_list(['W','bh','bu'])

    def sample_h(self, U):
        """Sample hidden nodes given an activation state of the outer nodes
        Args:
            U (NxKxP tensor): Indicator or probability tensor of outer layer
        Returns:
            p_h: (N x nh tensor): probability of the hidden nodes
            sample_h (N x nh tensor): 0/1 values of discretely sampled hidde nodes
        """
        wv = pt.mm(U.reshape(U.shape[0],-1), self.W.t())
        activation = wv + self.bh
        p_h = pt.sigmoid(activation)
        sample_h = pt.bernoulli(p_h)
        return p_h, sample_h

    def sample_U(self, h):
        """ Returns a sampled U as a unpacked indicator variable
        Args:
            h tensor: Hidden states
        Returns:
            p_u: Probability of each node [N,K,nv] array
            sample_U: One-hot encoding of random sample [N,K,nv] array
        """
        N = h.shape[0]
        wh = pt.mm(h, self.W).reshape(N,self.K,self.P)
        p_u = pt.softmax(wh + self.bu,1)
        sample = sample_multinomial(p_u,kdim=1)
        return p_u, sample

    def sample(self,num_subj,iter=10):
        """Draw new subjects from the model

        Args:
            num_subj (int): Number of subjects
            iter (int): Number of iterations until burn in
        """
        p = pt.ones(self.K)
        u = pt.multinomial(p,num_subj*self.P,replacement=True)
        u = u.reshape(num_subj,self.P)
        U = expand_mn(u,self.K)
        for i in range (iter):
            _,h = self.sample_h(U)
            _,U = self.sample_U(h)
        u = compress_mn(U)
        return u

    def Estep(self, emloglik,gather_ss=True,iter=None):
        """ Positive Estep for the multinomial boltzman model
        Uses mean field approximation to posterior to U and hidden parameters.
        Parameters:
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
            emloglik=pt.tensor(emloglik,dtype=pt.get_default_dtype())
        if iter is None:
            iter = self.epos_iter
        N=emloglik.shape[0]
        Uhat = pt.softmax(emloglik + self.bu,dim=1) # Start with hidden = 0
        for i in range(iter):
            wv = pt.mm(Uhat.reshape(N,-1), self.W.t())
            Eh = pt.sigmoid(wv + self.bh)
            wh = pt.mm(Eh, self.W).reshape(N,self.K,self.P)
            Uhat = pt.softmax(wh + self.bu + emloglik,1)
        if gather_ss:
            if self.Etype=='vis': # This is incorrect, but a understandable and information error
                self.epos_U = pt.softmax(emloglik,dim=1)
            elif self.Etype=='prob': # This is correct and isthe olny version that should be used.
                self.epos_U = Uhat
            self.epos_Eh = Eh
        return Uhat, pt.nan

    def marginal_prob(self):
        # If not given, then initialize: 
        if self.eneg_U is None:
            U,Eh=self.Eneg()
        pi  = pt.mean(self.eneg_U,dim=0)
        return pi

    def Mstep(self):
        """Performs gradient step on the parameters

        Args:
            alpha (float, optional): [description]. Defaults to 0.8.
        """
        N = self.epos_Eh.shape[0]
        M = self.eneg_Eh.shape[0]
        epos_U=self.epos_U.reshape(N,-1)
        eneg_U=self.eneg_U.reshape(M,-1)
        self.W += self.alpha * (pt.mm(self.epos_Eh.t(),epos_U) - N / M * pt.mm(self.eneg_Eh.t(),eneg_U))
        self.bu += self.alpha * (pt.sum(self.epos_U,0) - N / M * pt.sum(self.eneg_U,0))
        self.bh += self.alpha * (pt.sum(self.epos_Eh,0) - N / M * pt.sum(self.eneg_Eh, 0))


class mpRBM_pCD(mpRBM):
    """multinomial (categorial) restricted Boltzman machine
    for learning of brain parcellations for probabilistic input
    Uses persistent Contrastive-Divergence k for learning
    """

    def __init__(self, K, P, nh, eneg_iter=3,eneg_numchains=77):
        super().__init__(K, P,nh)
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains


    def Eneg(self, U=None):
        if (self.eneg_U is None):
            U = pt.empty(self.eneg_numchains,self.K,self.P).uniform_(0,1)
        else:
            U = self.eneg_U
        for i in range(self.eneg_iter):
            Eh,h = self.sample_h(U)
            EU,U = self.sample_U(h)
        self.eneg_Eh = Eh
        self.eneg_U = EU
        return self.eneg_U,self.eneg_Eh

class mpRBM_CDk(mpRBM):
    """multinomial (categorial) restricted Boltzman machine
    for learning of brain parcellations for probabilistic input
    Uses persistent Contrastive-Divergence k for learning
    """

    def __init__(self, K, P, nh, eneg_iter=1):
        super().__init__(K, P,nh)
        self.eneg_iter = eneg_iter

    def Eneg(self,U):
        for i in range(self.eneg_iter):
            Eh,h = self.sample_h(U)
            EU,U = self.sample_U(h)
        self.eneg_Eh = Eh
        self.eneg_U = EU
        return self.eneg_U,self.eneg_Eh


class cmpRBM(mpRBM):

    def __init__(self, K, P, nh=None, Wc = None, theta=None, eneg_iter=10,epos_iter=10,eneg_numchains=77):
        """convolutional multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses variational stochastic maximum likelihood for learning

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
        if Wc is None:
            if nh is None:
                raise(NameError('Provide Connectivty kernel (Wc) matrix or number of hidden nodes (nh)'))
            self.nh = nh
            self.W = pt.randn(nh,P)
            self.theta = None
            self.set_param_list(['bu','W'])
        else:
            if Wc.ndim==2:
                self.Wc= Wc.view(Wc.shape[0],Wc.shape[1],1)
            self.nh = Wc.shape[0]
            if theta is None:
                self.theta = pt.randn((self.Wc.shape[2],))
            else:
                self.theta = pt.tensor(theta)
                if self.theta.ndim ==0:
                    self.theta = self.theta.view(1)
            self.W = (self.Wc * self.theta).sum(dim=2)
            self.set_param_list(['bu','theta'])
        self.gibbs_U = None # samples from the hidden layer for negative phase
        self.alpha = 0.01
        self.epos_iter = epos_iter
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains
        self.fit_bu = True
        self.fit_W = True


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
        sample_h = sample_multinomial(p_h,kdim=1)
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
        if emloglik is not None:
            act += emloglik
        p_u = pt.softmax(act ,1)
        sample = sample_multinomial(p_u,kdim=1)
        return p_u, sample

    def marginal_prob(self):
        # If not given, then initialize: 
        if self.gibbs_U is None:
            return pt.softmax(self.bu,0)
        else:
            pi  = pt.mean(self.gibbs_U,dim=0)
        return pi


    def Estep(self, emloglik,gather_ss=True,iter=None):
        """ Positive Estep for the multinomial boltzman model
        Uses mean field approximation to posterior to U and hidden parameters.
        Parameters:
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
            emloglik=pt.tensor(emloglik,dtype=pt.get_default_dtype())
        if iter is None:
            iter = self.epos_iter
        N=emloglik.shape[0]
        Uhat = pt.softmax(emloglik + self.bu,dim=1) # Start with hidden = 0
        for i in range(iter):
            wv = pt.matmul(Uhat,self.W.t())
            Hhat = pt.softmax(wv,1)
            wh = pt.matmul(Hhat, self.W)
            Uhat = pt.softmax(wh + self.bu + emloglik,1)
        if gather_ss:
            self.epos_Uhat = Uhat
            self.epos_Hhat = Hhat
        return Uhat, pt.nan

    def Eneg(self, iter=None, use_chains=None, emission_model=None):
        # If no iterations specified - use standard
        if iter is None:
            iter = self.eneg_iter
        # If no markov chain are initialized, start them off
        if (self.gibbs_U is None):
            p = pt.softmax(self.bu,0)
            self.gibbs_U = sample_multinomial(p,
                    shape=(self.eneg_numchains,self.K,self.P),
                    kdim=0,
                    compress=False)
        # Grab the current chains 
        if use_chains is None:
            use_chains = pt.arange(self.eneg_numchains)

        U = self.gibbs_U[use_chains]
        U0 = U.detach().clone()
        for i in range(iter):
            Y = emission_model.sample(compress_mn(U))
            emloglik = emission_model.Estep(Y)
            _,H = self.sample_h(U)
            _,U = self.sample_U(H,emloglik)
        self.eneg_H = H
        self.eneg_U = U
        # Persistent: Keep the new gibbs samples around
        self.gibbs_U[use_chains]=U 
        return self.eneg_U,self.eneg_H

    def Mstep(self):
        """Performs gradient step on the parameters

        Args:
            alpha (float, optional): [description]. Defaults to 0.8.
        """
        N = self.epos_Hhat.shape[0]
        M = self.eneg_H.shape[0]
        # Update the connectivity 
        if self.fit_W:
            gradW = pt.matmul(pt.transpose(self.epos_Hhat,1,2),self.epos_Uhat).sum(dim=0)/N
            gradW -= pt.matmul(pt.transpose(self.eneg_H,1,2),self.eneg_U).sum(dim=0)/M
            # If we are dealing with component Wc:
            if self.Wc is not None:
                gradW = gradW.view(gradW.shape[0],gradW.shape[1],1)
                weights = pt.sum(self.Wc,dim=(0,1))
                gradT   = pt.sum(gradW*self.Wc,dim=(0,1))/weights
                self.theta += self.alpha * gradT
                self.W = (self.Wc * self.theta).sum(dim=2)
            else:
                self.W += self.alpha * gradW
        
        # Update the bias term
        if self.fit_bu: 
            gradBU =   1 / N * pt.sum(self.epos_Uhat,0) 
            gradBU -=  1 / M * pt.sum(self.eneg_U,0)
            self.bu += self.alpha * gradBU

def sample_multinomial(p,shape=None,kdim=0,compress=False):
    """Samples from a multinomial distribution
    Fast smpling from matrix probability without looping

    Args:
        p (tensor): Tensor of probilities, which sums to 1 on the dimension kdim
        shape (tuple): Shape of the output data (in uncompressed form): Smaller p will be broadcasted to target shape
        kdim (int): Number of dimension of p that indicates the different categories (default 0)
        compress: Return as int (True) or indicator (false)
    Returns: Samples either in indicator coding (compress = False)
            or as ints (compress = False)
    """
    if shape is None:
        shape = p.shape
    out_kdim = len(shape) - p.dim() + kdim
    K = p.shape[kdim]
    shapeR = list(shape)
    shapeR[out_kdim]=1 # Set the probability dimension to 1
    r = pt.empty(shapeR).uniform_(0,1)
    cdf_v = p.cumsum(dim=kdim)
    sample = (r < cdf_v).float()
    if compress:
        return sample.argmax(dim=out_kdim)
    else:
        for k in np.arange(K-1,0,-1):
            a=sample.select(out_kdim,k) # Get view of slice
            a-=sample.select(out_kdim,k-1)
        return sample

def expand_mn(u,K):
    """Expands a N x P multinomial vector
    to an N x K x P tensor of indictor variables
    Args:
        u (2d-tensor): N x nv matrix of samples from [int]
        K (int): Number of categories
    Returns
        U (3d-tensor): N x K x nv matrix of indicator variables [default float]
    """
    N = u.shape[0]
    P = u.shape[1]
    U = pt.zeros(N,K,P)
    U[np.arange(N).reshape((N,1)),u,np.arange(P)]=1
    return U

def compress_mn(U):
    """Compresses a N x K x P tensor of indictor variables
    to a N x P multinomial tensor
    Args:
        U (3d-tensor): N x K x P matrix of indicator variables
    Returns
        u (2d-tensor): N x P matrix of category labels [int]
    """
    u=U.argmax(1)
    return u
