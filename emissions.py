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
    def __init__(self,K=4,N=10,P=20):
        self.K = K # Number of states
        self.N = N
        self.P = P 
        self.alpha = 1
        self.beta = 1
        self.sigma = 1

    def initialize(self, data):
        """

        """
        self.Y = data # This is assumed to be (num_sub,N,P)
        self.num_subj = data.shape[0]

    def loglike(self,Y,sub = None):
        """
            Returns log p(Y|U) for each value of U, up to a constant
        """
        if sub is None:
            sub = range(self.num_subj)
        L = np.empty((self.num_subj,self.P))
        uVVu = np.sum(self.V**2,axis=0) # This is u.T V.T V u for each u
        for i in sub:
            YV = self.Y[i,:,:].T @ self.V
            self.s[i]=YV/uVVu-self.beta*self.sigma2  # Maximized g
            YY = np.sum(self.Y[i,:,:]**2,axis=0)
            self.res[i] = YY - 2 *YV + uVVu
            LL = -1/(2*self.sigma)*self.res[i]+ self.beta*self.s[i]

    def random_V(self): 
        V = np.random.normal(0,1,(self.N,self.K))
        # Make zero mean, unit length
        V = V - V.mean(axis=0)
        V = V / np.sqrt(np.sum(V**2,axis=0))
        return V

    def generate_data(self,U):
        num_subj = U.shape[0]
        Y = np.empty((num_subj,self.N,self.P))
        signal = np.empty((num_subj,self.P))
        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            signal[s,:] = np.random.gamma(self.alpha,self.beta,(self.P,))
            Y[s,:,:] = self.V @ U [s,:,:] * signal[s,:]
            # And add noise of variance 1
            Y[s,:,:] = Y[s,:,:] + np.random.normal(0,self.sigma,(self.N,self.P))
        return(Y,signal)

def test_MixGaussian(K=5,N=10,P=25): 
    M = MixGaussianExponential(K,N,P)
    V = M.random_V()
    U = np.kron(np.eye(K),np.ones(np.int(P/K),))
    U = U.reshape((1,K,P))    
    M.V = V
    Y, s = M.generate_data(U)
    M.initialize

if __name__ == '__main__':
    test_MixGaussian()