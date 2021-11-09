# Example Models
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import copy



class MixGaussianExponential:
    """
    Mixture of Gaussians with an exponential
    scaling factor on the signal and a fixed noise variance
    """
    def __init__(self,K=4):
        self.K = K # Number of states

    def initialize(self, data):
        """

        """
        self.Y = data # This is assumed to be (num_sub,N,P)
        self.num_subj, self.N, self.P = data.shape

    def loglike(self,Y,sub = None):
        """
            Returns log p(Y|U) for each value of U, up to a constant
        """
        if sub is None:
            sub = range(self.num_subj)
        L = np.empty(self.num_subj
        uVVu = np.sum(self.V**2,axis=0) # This is u.T V.T V u for each u
        for i in sub:
            YV = self.Y[i,:,:].T @ self.V
            self.s[i]=YV/uVVu-self.beta*self.sigma2  # Maximized g
            YY = np.sum(self.Y[i,:,:]**2,axis=0)
            self.res[i] = YY - 2 *YV + uVVu
            LL = -1/(2*self.sigma)*self.res[i]+ self.beta*self.s[i]

    def cond_prob(self,U,node,prior = False):
        """
            Returns the conditional probabity vector for node x, given U
        """
        x = np.arange(self.K)
        ind = np.where(self.W[node,:]>0) # Find all the neighbors for node x (precompute!)
        nb_x = U[ind] # Neighbors to node x
        same = np.equal(x,nb_x.reshape(-1,1))
        loglik = self.theta_w * np.sum(same,axis=0)
        if prior:
            loglik = loglik +self.logMu[:,node]
        p = np.exp(loglik)
        p = p / np.sum(p)
        return(p)

    def sample_gibbs(self,U0 = None,evidence = None,prior = False, iter=5):
        # Get initial starting point if required
        U = np.zeros((iter+1,self.P))
        if U0 is None:
            for p in range(self.P):
                U[0,p] = np.random.choice(self.K,p = self.mu[:,p])
        else:
            U[0,:] = U0
        for i in range(iter):
            U[i+1,:]=U[i,:] # Start the new sample from the old one
            for p in range(self.P):
                prob = self.cond_prob(U[i+1,:],p,prior = prior)
                U[i+1,p]=np.random.choice(self.K,p=prob)
        return(U)

    def generate_subjects(self,num_subj = 10):
        """
            Samples a number of subjects from the prior
        """
        U = np.zeros((num_subj,self.P))
        for i in range(num_subj):
            Us = self.sample_gibbs(prior=True,iter=10)
            U[i,:]=Us[-1,:]
        return U

    def generate_emission (self,U,V = None, N = 30, num_subj=10,theta_alpha = 2, theta_beta=0.5):
        """
            Generates a specific experimental data set
        """
        num_subj = U.shape[0]

        if V is None:
            V = np.random.normal(0,1,(N,self.K))
            # Make zero mean, unit length
            V = V - V.mean(axis=0)
            V = V / np.sqrt(np.sum(V**2,axis=0))
        else:
            N,K = V.shape
            if K != self.K:
                raise(NameError('Number of columns in V need to match Model.K'))
        Y = np.empty((num_subj,N,self.P))
        signal = np.empty((num_subj,self.P))
        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            signal[s,:] = np.random.gamma(theta_alpha,theta_beta,(self.P,))
            # Generate mean signal
            # One -hot encoding could be done:
            # UI[U[0,:].astype('int'),np.arange(self.P)]=1
            Y[s,:,:] = V[:,U[s,:].astype('int')] * signal[s,:]
            # And add noise of variance 1
            Y[s,:,:] = Y[s,:,:] + np.random.normal(0,np.sqrt(1/N),(N,self.P))
        param = {'theta_alpha':theta_alpha,
                 'theta_beta':theta_beta,
                 'V':V,
                 'signal':signal}
        return(Y,param)
