# Example Models
import numpy as np
import matplotlib.pyplot as plt
import copy
# import seaborn as sns


class PottsModel:
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states
    Potential function is determined by linkages
    parameterization is joint between all linkages, although it could be split
    into different parameter functions
    """
    def __init__(self,K=3,width=5,height=5):
        self.width = width
        self.height = height
        self.dim = (height,width)
        self.K = K # Number of states
        # Get the grid and neighbourhood relationship
        self.define_grid()
        self.P = self.xx.shape[0]
        self.theta_w = 1 # Weight of the neighborhood relation - inverse temperature param

    def define_grid(self):
        """
        Makes a connectivity matrix and a mappin matrix for a simple rectangular grid
        """
        XX, YY = np.meshgrid(range(self.width),range(self.height))
        self.xx = XX.reshape(-1)
        self.yy = YY.reshape(-1)
        self.D = (self.xx - self.xx.reshape((-1,1)))**2 + (self.yy - self.yy.reshape((-1,1)))**2
        self.W = np.double(self.D==1) # Nearest neighbour connectivity

    def define_mu(self, theta_mu=1):
        """
            Defines pi (prior over parcels) using a Ising model with K centroids
        """
        mx = np.random.uniform(0, self.width,(self.K,))
        my = np.random.uniform(0, self.width,(self.K,))
        d2 = (self.xx-mx.reshape(-1,1))**2 + (self.yy-my.reshape(-1,1))**2
        self.mu = np.exp(-d2/theta_mu)
        self.mu = self.mu / self.mu.sum(axis=0)
        self.logMu = np.log(self.mu)

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
        """
            Returns the energy term of the network
            up to a constant the loglikelihood of the state
        Args:
            Y ([np-array]): 1d or 2d array of network states
        """
        phi=self.potential(Y)
        l = self.theta_w @ phi
        return(l)

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

    def plot_maps(self,Y,cmap='tab20',vmin=0,vmax=19,grid=None):
        """
            Plots a set of map samples as an image grid
            N X P
        """
        if Y.ndim == 1:
            ax = plt.imshow(Y.reshape(self.dim),cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
            ax.axes.yaxis.set_visible(False)
            ax.axes.xaxis.set_visible(False)
        else:
            N,P = Y.shape
            if grid is None:
                grid = np.zeros((2,),np.int32)
                grid[0] = np.ceil(np.sqrt(N))
                grid[1] = np.ceil(N/grid[0])
            for n in range(N):
                ax = plt.subplot(grid[0],grid[1],n+1)
                ax.imshow(Y[n,:].reshape(self.dim),cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

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
            V = V / np.sqrt(np.mean(V**2,axis=0))
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
            Y[s,:,:] = Y[s,:,:] + np.random.normal(0,1,(N,self.P))
        param = {'theta_alpha':theta_alpha,
                 'theta_beta':theta_beta,
                 'V':V,
                 'signal':signal}
        return(Y,param)


if __name__ == '__main__':
    M = PottsModel(5,30,30)
    M.define_mu(200)
    # plt.imshow(cluster.reshape(M.dim),cmap='tab10')
    U = M.generate_subjects(num_subj = 4)
    [Y,param] = M.generate_emission(U)
    pass 
