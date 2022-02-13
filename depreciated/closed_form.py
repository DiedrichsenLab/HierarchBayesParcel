# ClosedForm solutions to Ising and Potts models
def combinations(K,P):
    """
        Returns all possible states of a system

    Args:
        K ([int]): Number of states (or list of states)
        P ([int]): Number of nodes

    Returns:
        Y : np-array: NxP array of possible states
    """
    N = K**P  # This is the number of combinations
    Y = np.zeros((N,P),np.int8)
    k = np.arange(K)
    for p in range (P):
        Y[:,p]=np.tile(np.repeat(k,K**p),K**(P-p-1))
    return Y



class Ising:
    def __init__(self,width=5,height=5):
        self.xx, self.yy, self.D, self.W = get_grid(width,height)
        self.P = self.xx.shape[0]
        self.b = np.random.normal(0,1,(self.P,))
        self.b = np.zeros((self.N,))

    def loglike(self,Y):
        """
            Returns the energy term of the network
            up to a constant the loglikelihood of the state
        Args:
            Y ([np-array]): 1d or 2d array of network states
        """
        return -0.5 * np.sum((Y @ self.W)* Y,axis=1)  + np.sum(self.b * Y,axis=1)

class Potts:
    """
    Potts models (Markov random field on multinomial variable)
    with K possible states
    Potential function is determined by linkages
    parameterization is joint between all linkages, although it could be split
    into different parameter functions
    """
    def __init__(self,K=3,width=5,height=5):
        self.xx, self.yy, self.D, self.W = get_grid(width,height)
        self.dim = (height,width)
        self.P = self.xx.shape[0]
        self.K = K # Number of states
        self.b = np.random.normal(0,1,(self.P,self.K))
        self.numparam = 1 # Number of weight parameters
        self.theta = np.ones((self.numparam,))

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

    def possible_states(self,fixed=None):
        """
            returns a matrix of all possible states
        """
        if fixed is None:
            fixed = np.empty((self.P,))
            fixed[:]= np.nan
        fixIn = np.where(np.logical_not(np.isnan(fixed)))[0]
        freeIn = np.where(np.isnan(fixed))[0]
        num_states = freeIn.shape[0]
        Y = np.empty((self.K**num_states,self.P))
        Y[:,freeIn] = combinations(self.K,num_states)
        Y[:,fixIn] = fixed[fixIn]
        return Y

    def loglike(self,Y):
        """
            Returns the energy term of the network
            up to a constant the loglikelihood of the state
        Args:
            Y ([np-array]): 1d or 2d array of network states
        """
        phi=self.potential(Y)
        l = self.theta @ phi
        return(l)

    def cond_prob(self,Y,node):
        """
            Returns the conditional probabity vector for node x, given Y
        """
        x = np.arange(self.K)
        ind = np.where(self.W[node,:]>0) # Find all the neighbors for node x (precompute!)
        nb_x = Y[ind] # Neighbors to node x
        same = np.equal(x,nb_x.reshape(-1,1))
        loglik = self.theta * np.sum(same,axis=0)
        p = np.exp(loglik)
        p = p / np.sum(p)
        return(p)

    def sample_gibbs(self,num_chains=10,Y0=None,fixed=None,iter=100):
        if fixed is None:
            fixed = np.empty((self.P,))
            fixed[:]= np.nan
        fixIn = np.where(np.logical_not(np.isnan(fixed)))[0]
        samIn = np.where(np.isnan(fixed))[0]
        Y = np.zeros((iter,self.P,num_chains))
        if Y0 is None:
            Y0 = np.random.choice(self.K,(self.P,num_chains))
        Y[0,:,:] = Y0
        Y[0,fixIn,:] = fixed[fixIn].reshape((-1,1)) # Set the fixed elements to the right values
        for i in range(iter):
            for s in samIn:
                for n in range(num_chains): # Loop over chains
                    p = self.cond_prob(Y[i,:,n],s)
                    Y[i,s,n]=np.random.choice(self.K,p=p)
            if i+1<iter:
                Y[i+1,:,:]=Y[i,:,:] # Start the new sample from the old one
        return(Y)

    def plot_samples(self,Y):
        """
            Plots a set of map samples as an image grid
        """
        N,P = Y.shape
        rows = np.ceil(np.sqrt(N))
        for n in range(N):
            plt.subplot(rows,rows,n+1)
            plt.imshow(Y[n,:].reshape(self.dim))

    def fit(self,Y,iter=100,fixed=None):
        """
            Fitting the parameters of the model using divergent gradients on fully observed data
            This version relies on the closed form
        """
        N = Y.shape[0] # Number of data points
        Yp = self.possible_states(fixed) # Get array of possible states to integrate over
        Phi_emp = self.potential(Y) - 20
        Phi_th  = self.potential(Yp) - 20
        theta = np.empty((self.numparam,iter))
        ll = np.empty((iter,))
        for i in range(iter):
            theta[:,i]=self.theta
            p = np.exp(np.sum(self.theta*Phi_th,axis=0))
            Z = np.sum(p) # Partition function (normalizing constant)
            p = p / Z
            ll[i] = np.sum(np.sum(self.theta*Phi_emp,axis=0) - np.log(Z))
            E_th = np.sum(Phi_th * p,axis=1) # Theoretical expectation of potentials
            E_emp = np.sum(Phi_emp,axis=1)/N   # Empirical expectaction of potentials
            grad = E_emp - E_th
            self.theta = self.theta + 0.01 * grad
        RES = {'theta':theta,'ll':ll}
        return RES
