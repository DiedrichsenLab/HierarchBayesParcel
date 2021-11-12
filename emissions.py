# Example Models
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import copy

from numpy.lib.index_tricks import RClass



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
        self.sigma2 = 1

    def initialize(self, data):
        """

        """
        self.Y = data # This is assumed to be (num_sub,N,P)
        self.num_subj = data.shape[0]
        self.s = np.empty((self.num_subj,self.K,self.P))
        self.rss = np.empty((self.num_subj,self.K,self.P))

    def Estep(self,sub = None):
        """
            Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        """
        if sub is None:
            sub = range(self.num_subj)
        LL = np.empty((self.num_subj, self.K, self.P))
        uVVu = np.sum(self.V**2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = self.Y[i, :, :].T @ self.V
            self.s[i,:,:]=(YV/uVVu-self.beta*self.sigma2).T  # Maximized g
            self.s[i][self.s[i]<0]=0  # Limit to 0
            YY = np.sum(self.Y[i,:,:]**2, axis=0)
            self.rss[i, :, :] = YY - 2 *YV.T * self.s[i,:,:] + self.s[i,:,:]**2 * uVVu.reshape((self.K,1))
            LL[i,:,:] = -0.5*self.sigma2 * self.rss[i,:,:] + self.beta * self.s[i,:,:]
        return LL

    def Mstep(self, U_hat):
        """
            Performs the M-step on a specific U-hat
        """
        SU = self.s * U_hat
        YU = np.zeros((self.N, self.K))
        UU = np.zeros((self.K, self.K))
        for i in range(self.num_subj):
            YU = YU + self.Y[i,:,:] @ SU[i,:,:].T
            UU = UU + SU[i,:,:] @ SU[i,:,:].T
        self.V = np.linalg.solve(UU,YU.T).T
        ERSS = np.sum(U_hat * self.rss)
        self.sigma2 = ERSS/(self.N*self.P)

    def random_V(self):
        V = np.random.normal(0,1,(self.N,self.K))
        # Make zero mean, unit length
        V = V - V.mean(axis=0)
        V = V / np.sqrt(np.sum(V**2,axis=0))
        return V

    def generate_data(self, U):
        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        signal = np.empty((num_subj, self.P))
        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            signal[s, :] = np.random.gamma(self.alpha, self.beta, (self.P,))
            Y[s, :, :] = self.V @ U[s, :, :] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + np.random.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        return Y, signal


def MixGaussian(K=5,N=10,P=25):
    # Generate a simple iid emission model
    Mtrue = MixGaussianExponential(K, N, P)
    Mtrue.V = Mtrue.random_V()
    Utrue = np.kron(np.eye(K),np.ones(np.int(P/K),))
    Utrue = Utrue.reshape((1, K, P))

    Y, s = Mtrue.generate_data(Utrue)

    # Estimate the model back
    M = MixGaussianExponential(K, N, P)
    M.V = M.random_V()
    M.initialize(Y)
    LL = M.Estep()
    Uhat = np.exp(LL)
    Uhat = Uhat / np.sum(Uhat,axis=1)
    M.Mstep(Uhat)       # Update the parameters
    pass


if __name__ == '__main__':
    MixGaussian()
