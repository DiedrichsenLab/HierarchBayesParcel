# Example Models
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import copy
import nibabel as nb
from nilearn import plotting

import sys
sys.path.insert(0, "D:/python_workspace/")


def eucl_distance(coord):
    """
    Calculates euclediand distances over some cooordinates
    Args:
        coord (ndarray)
            Nx3 array of x,y,z coordinates
    Returns:
        dist (ndarray)
            NxN array pf distances
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points,num_points))
    for i in range(3):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)


class ArrangementModel: 
    """Abstract arrangement model
    """
    def __init__(self, K, P):
        self.K = K  # Number of states
        self.P = P  # Number of nodes


class ArrangeIndependent(ArrangementModel):
    """ Arrangement model for spatially independent assignment
        Either with a spatially uniform prior 
        or a spatially-specific prior  
    """
    def __init__(self, K=3, P=100, spatial_specific=False):
        super(ArrangeIndependent, self).__init__(K, P)
        # In this model, the spatially independent arrangement has
        # two options, spatially uniformed and spatially specific prior
        if spatial_specific:
            self.pi = np.ones((P, K)) / K
        else:
            self.pi = np.ones((K,)) / K
    
    def Estep(self, emloglik):
        """ Estep for the spatial arrangement model

        Parameters: 
            emloglik (np.array):
                emission log likelihood log p(Y|u,theta_E) a numsubjxPxK matrix
        Returns: 
            Uhat (np.array):
                posterior p(U|Y) a numsubj x P x K matrix 
        """
        numsubj, P, K = emloglik.shape
        logq = emloglik + np.log(self.pi)
        Uhat = np.exp(logq)
        Uhat = Uhat / np.sum(Uhat, axis=2).reshape((numsubj, P, 1))
        return Uhat 

    def Mstep(self, Uhat):
        """ M-step for the spatial arrangement model
        """
        pi = np.mean(Uhat, axis=0)  # Averarging over subjects
        if self.pi.ndim == 1:
            self.pi = pi.mean(axis=1)
        else: 
            self.pi = pi

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
        for i in range(num_subj):
            for p in range(self.P):
                U[i, p] = np.random.choice(self.K, p=self.pi)

        return U


class PottsModel:
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states
    Potential function is determined by linkages
    parameterization is joint between all linkages, although it could be split
    into different parameter functions
    """
    def __init__(self,K=3,W = None):
        self.W = W
        self.K = K # Number of states
        self.P = W.shape[0]
        self.theta_w = 1 # Weight of the neighborhood relation - inverse temperature param

    def define_mu(self, theta_mu=1,centroids=None):
        """
            Defines pi (prior over parcels) using a Ising model with K centroids
            Needs the Distance matrix to define the prior probability
        """
        if centroids is None:
            centroids = np.random.choice(self.P,(self.K,))
        d2 = self.Dist[centroids,:]**2
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


class PottsModelGrid(PottsModel):
    """
        Potts model defined on a regular grid
    """
    def __init__(self,K=3,width=5,height=5):
        self.width = width
        self.height = height
        self.dim = (height,width)
        # Get the grid and neighbourhood relationship
        W = self.define_grid()
        super().__init__(K,W)

    def define_grid(self):
        """
        Makes a connectivity matrix and a mappin matrix for a simple rectangular grid
        """
        XX, YY = np.meshgrid(range(self.width),range(self.height))
        self.xx = XX.reshape(-1)
        self.yy = YY.reshape(-1)
        self.Dist = np.sqrt((self.xx - self.xx.reshape((-1,1)))**2 + (self.yy - self.yy.reshape((-1,1)))**2)
        W = np.double(self.Dist==1) # Nearest neighbour connectivity
        return W

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


baseDir = '/Users/jdiedrichsen/Data/fs_LR_32'
hemN = ['L','R']
hem_name = ['CortexLeft','CortexRight']


class PottsModelCortex(PottsModel):
    """
        Potts models on a single hemisphere on an isocahedron Grid
    """
    def __init__(self,K=3,hem=[0], roi_name='Icosahedron-162'):
        self.flatsurf = []
        self.inflsurf = []
        self.roi_label = []
        self.hem = hem
        self.roi_name = roi_name
        self.P = 0
        vertex = []
        for i,h in enumerate(hem):
            flatname = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.flat.surf.gii')
            inflname = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.inflated.surf.gii')
            labname = os.path.join(baseDir,roi_name + '.32k.' + hemN[h] + '.label.gii')
            sphere_name = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.sphere.surf.gii')

            # Get the inflate and flat surfaces and stor for later use
            self.flatsurf.append(nb.load(flatname))
            self.inflsurf.append(nb.load(inflname))

            # Get the labels and append
            roi_gifti = nb.load(labname)
            L = roi_gifti.agg_data()
            num_roi = L.max()
            L[L>0]=L[L>0]+self.P # Add the number of parcels from other hemisphere - but not to zero
            self.roi_label.append(L)
            self.P = self.P + num_roi

            # Get the vertices for the sphere for distance matrix
            sphere = nb.load(sphere_name)
            vertex.append(sphere.darrays[0].data)
            vertex[i][:,0] = vertex[i][:,0]+(h*2-1)*500

        self.coord = np.zeros((self.P,3))
        for i,h in enumerate(hem):
            for p in np.unique(self.roi_label[i]):
                if p > 0:
                    self.coord[p-1,:]= vertex[i][self.roi_label[i]==p,:].mean(axis=0)

        self.Dist = eucl_distance(self.coord)
        thresh = 1
        self.W = np.logical_and(self.Dist>0,self.Dist< thresh)
        while np.all(self.W.sum(axis=1)<=6):
            thresh = thresh+1
            self.W = np.logical_and(self.Dist>0,self.Dist< thresh)
        thresh = thresh-1
        W = np.logical_and(self.Dist>0,self.Dist< thresh)
        super().__init__(K,W)

    def map_data(self,data,out_value=0):
        """
            Args:
                data (np-arrray): 1d-array
                out_value = 0(default) or np.nan
            Returns:
                List of 1-d np-arrays (per hemisphere)
        """
        data = np.insert(data,0,out_value)
        mapped_data = []
        for i,h in enumerate(self.hem):
            mapped_data.append(data[self.roi_label[i]])
        return mapped_data

    def plot_map(self,data,cmap='tab20',vmin=0,vmax=19,grid=None):
        """
        """
        d = self.map_data(data)
        coords = self.inflsurf[0].agg_data('pointset')
        faces = self.inflsurf[0].agg_data('triangle')
        view = plotting.view_surf([coords,faces],d[0],
                cmap=cmap,vmin=vmin,vmax=vmax,symmetric_cmap = False,
                colorbar=False)
        # plotting.plot_surf([coords,faces],data[0],darkness=0.3,hemi='left')
        return view