# Example Models
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import copy

from numpy.lib.index_tricks import RClass


class EmissionModel: 
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        self.K = K  # Number of states
        self.N = N
        self.P = P
        if data is not None:
            self.initialize(data)
        else:
            pass

        if params is None:
            self.set_params()
        else:
            self.params = params
    
    def initialize(self, data):
        """Stores the data in emission model itself 

        """
        self.Y = data # This is assumed to be (num_sub,P,N)
        self.num_subj = data.shape[0]

    def Estep(self, sub=None):
        """Implemnents E-step and returns 
            Parameters: 
                sub (list): 
                    List of indices of subjects to use. Default=all (None)
            Returns:
                emloglik (np.array):
                    emission log likelihood log p(Y|u,theta_E) a numsubjxPxK matrix
        """
        pass 

    def Mstep(self,U_hat): 
        """Implements M-step for the model 
        """
        pass

    def get_params(self):
        """[summary]
        """
        pass

    def set_params(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        pass


class MixGaussianExponential(EmissionModel):
    """
    Mixture of Gaussians with an exponential
    scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super(MixGaussianExponential, self).__init__(K, N, P, data, params)
        self.alpha = 1
        self.beta = 1

    def initialize(self, data):
        """Stores the data in emission model itself 
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super(MixGaussianExponential, self).initialize(data)
        self.YY = self.Y**2
        self.s = np.empty((self.num_subj, self.K, self.P))
        self.rss = np.empty((self.num_subj, self.K, self.P))

    def get_params(self):
        """ Get the parameters for the Gaussian mixture model

        :return: the parcel-specific mean and uniformed sigma square
        """
        params = []
        np.testing.assert_array_equal(self.K, self.V.shape[1])
        for i in range(self.K):
            params = np.array([params, self.V[:, i]])

        # np.testing.assert_array_equal(self.K, self.sigma2.shape[1])
        for i in range(self.K):
            params = np.array([params, self.sigma2])

        return params

    def set_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
        and variance. Here, we assume the variance is equal across different parcels.
        Therefore, there are total k+1 parameters in this mixture model

        set the initial parameters for gaussian mixture
        """
        V = np.random.normal(0,1, (self.N, self.K))
        # standardise V to unit length
        V = V - V.mean(axis=0)
        self.V = V / np.sqrt(np.sum(V**2, axis=0))

        self.sigma2 = 1
        self.nparams = self.K + 1

    def Estep(self, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        :param Y: the real data
        :param sub: specify which subject to optimize

        :return: the expected log likelihood for emission model
        """
        if sub is None:
            sub = range(self.Y.shape[0])
        LL = np.empty((self.Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V**2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = self.Y[i, :, :].T @ self.V
            # self.s[i,:,:]=(YV/uVVu-self.beta*self.sigma2).T  # Maximized g
            # self.s[i][self.s[i]<0]=0  # Limit to 0
            self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - 2*YV.T + uVVu.reshape((self.K, 1))

            LL[i, :, :] = -0.5*self.P*self.N*(np.log(2*np.pi) + np.log(self.sigma2))-0.5*(1/self.sigma2) * self.rss[i, :, :]
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

    def sample(self, U, V=None):
        if V is None:
            V = np.random.normal(0, 1, (self.N, self.K))
            # Make zero mean, unit length
            V = V - V.mean(axis=0)
            V = V / np.sqrt(np.sum(V ** 2, axis=0))
        else:
            this_n, this_k = V.shape
            if this_k != self.K:
                raise(NameError('Number of columns in V need to match Model.K'))

        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        for s in range(num_subj):
            Y[s, :, :] = V[:, U[s, :].astype('int')]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + np.random.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        return Y

    def sample_with_gamma(self, U):
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


def MixGaussian(K=5, N=10, P=25):
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
