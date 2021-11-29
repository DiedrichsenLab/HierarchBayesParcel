# Example Models
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

from numpy import log, exp, sqrt


class EmissionModel:
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        self.K = K  # Number of states
        self.N = N
        self.P = P
        self.nparams = 0
        if data is not None:
            self.initialize(data)
        if params is None:
            self.random_params()  # Random parameter state
        else:
            self.set_params(params)

    def initialize(self, data):
        """Stores the data in emission model itself
        """
        self.Y = data  # This is assumed to be (num_sub,P,N)
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

    def Mstep(self, U_hat):
        """Implements M-step for the model
        """
        pass

    def get_params(self):
        """Returns all parameters as a vector
        """
        pass

    def set_params(self, params):
        """Sets all parameters from a vector
        """
        pass

    def random_params(self):
        """Sets all random parameters from a vector
        """
        pass


class MixGaussian(EmissionModel):
    """
    Mixture of Gaussians with isotropic noise
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super().__init__(K, N, P, data, params)
        self.nparams = self.N * self.K + 1

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super().initialize(data)
        self.YY = self.Y**2
        self.rss = np.empty((self.num_subj, self.K, self.P))

    def get_params(self):
        """ Get the parameters for the Gaussian mixture model

        :return: the parcel-specific mean and log sigma2
        """
        np.testing.assert_array_equal(self.K, self.V.shape[1])
        return np.append(self.V.flatten('F'), log(self.sigma2))

    def set_params(self, theta):
        """ Set the model parameters by the given input thetas

        :param theta: input parameters
        :return: None
        """
        self.V = theta[0:self.N*self.K].reshape(self.N, self.K)
        self.sigma2 = exp(theta[-1])

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        V = np.random.normal(0, 1, (self.N, self.K))
        # standardise V to unit length
        V = V - V.mean(axis=0)
        self.V = V / sqrt(np.sum(V**2, axis=0))
        self.sigma2 = exp(np.random.normal(0, 0.3))

    def Estep(self, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        :param Y: the real data
        :param sub: specify which subject to optimize

        :return: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if sub is None:
            sub = range(self.Y.shape[0])
        LL = np.empty((self.Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V**2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = np.dot(self.Y[i, :, :].T, self.V)
            self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - 2*YV.T + uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5*self.N*(np.log(2*np.pi) + np.log(self.sigma2))-0.5*(1/self.sigma2) * self.rss[i, :, :]

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat.
            In this emission model, the parameters need to be updated
            are V and sigma2.
        """
        # SU = self.s * U_hat
        YU = np.zeros((self.N, self.K))
        UU = np.zeros((self.K, self.P))
        for i in range(self.num_subj):
            YU = YU + self.Y[i, :, :] @ U_hat[i, :, :].T
            UU = UU + U_hat[i, :, :]
        # self.V = np.linalg.solve(UU,YU.T).T
        # Here we update the v_k, which is sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k))
        self.V = (YU.T / np.sum(UU, axis=1).reshape(-1, 1)).T
        ERSS = np.sum(U_hat * self.rss)
        self.sigma2 = ERSS/(self.N*self.P*self.num_subj)

    def sample(self, U):
        """ Generate random data given this emission model

        :param U: The prior arrangement U from arragnment model
        :return: sampled data Y
        """
        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        for s in range(num_subj):
            Y[s, :, :] = self.V[:, U[s, :].astype('int')]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + np.random.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        return Y


class MixGaussianGamma(EmissionModel):
    """
    Mixture of Gaussians with signal strength (fit gamma distribution)
    for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super(MixGaussianGamma, self).__init__(K, N, P, data, params)
        self.nparams = self.N * self.K + 3  # V shape is (N, K) + sigma2 + alpha + beta

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super(MixGaussianGamma, self).initialize(data)
        self.YY = self.Y ** 2
        self.s = np.empty((self.num_subj, self.K, self.P))
        self.rss = np.empty((self.num_subj, self.K, self.P))

    def get_params(self):
        """ Get the parameters for the Gaussian mixture model

        :return: the parcel-specific mean and uniformed sigma square, alpha, and beta
        """
        np.testing.assert_array_equal(self.K, self.V.shape[1])
        return np.append(self.V.flatten('F'), (self.sigma2, self.alpha, self.beta))

    def set_params(self, theta):
        """ Set the model parameters by the given input thetas

        :param theta: input parameters by fixed order
                      N*K Vs + sigma2 + alpha + beta
        :return: None
        """
        self.V = theta[0:self.N*self.K].reshape(self.N, self.K)
        self.sigma2 = theta[-3]
        self.alpha = theta[-2]
        self.beta = theta[-1]

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        V = np.random.normal(0, 1, (self.N, self.K))
        # standardise V to unit length
        V = V - V.mean(axis=0)
        self.V = V / sqrt(np.sum(V**2, axis=0))
        self.sigma2 = exp(np.random.normal(0, 0.3))
        # The initial random alpha and beta values are fitted to 24 subjects data of MDTB
        # Run 'data_estimation.py' to see evidence
        # self.alpha = np.random.normal(2.5, 0.46)
        # self.beta = -0.13 * self.alpha + 0.6
        self.alpha = 1
        self.beta = 1

    def _norm_pdf_multivariate(self, x, mu, sigma):
        """ pdf function for multivariate normal distribution

        Note: X and mu are assumed to be column vector
        :param x: multivariate data. Shape (n_dim, 1)
        :param mu: theta mu of the distribution. Shape (n_dim, 1)
        :param sigma: theta cov of the distribution. Shape (n_dim, n_dim)

        :return: the probability of a data point in the given normal distribution
        """
        size = len(x)
        if size == len(mu) and (size, size) == sigma.shape:
            det = np.linalg.det(sigma)
            if det == 0:
                raise NameError("The covariance matrix can't be singular")
            norm_const = 1.0 / (np.math.pow((2 * np.pi), float(size) / 2) * np.math.pow(det, 1.0 / 2))
            x_mu = np.matrix(x - mu)
            inv_ = np.linalg.inv(sigma)
            result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")

    def Estep(self, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        :param Y: the real data
        :param sub: specify which subject to optimize

        :return: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if sub is None:
            sub = range(self.Y.shape[0])
        LL = np.empty((self.Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V ** 2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = np.dot(self.Y[i, :, :].T, self.V)
            self.s[i, :, :] = (YV / uVVu - self.beta * self.sigma2).T  # Maximized g
            self.s[i][self.s[i] < 0] = 0  # Limit to 0
            self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - 2*YV.T*self.s[i, :, :] + self.s[i, :, :]**2 * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (np.log(2 * np.pi) + np.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + self.alpha*log(self.beta) - log(gamma(self.alpha))  - self.beta * self.s[i, :, :]

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. U_hat = E[u_i ^(k), s_i]
            In this emission model, the parameters need to be updated
            are V, sigma2, alpha, and beta
        """
        # SU = self.s * U_hat
        YU = np.zeros((self.N, self.K))
        UU = np.zeros((self.K, self.P))
        for i in range(self.num_subj):
            YU = YU + np.dot(self.Y[i, :, :], U_hat[i, :, :].T)
            UU = UU + U_hat[i, :, :]
        # self.V = np.linalg.solve(UU,YU.T).T
        # 1. Updating the V
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.V = (YU.T / np.sum(UU**2, axis=1).reshape(-1, 1)).T
        # 2. Updating the sigma squared.
        ERSS = np.sum(U_hat * self.rss)
        self.sigma2 = ERSS / (self.N * self.P * self.num_subj)
        # 3. Updating the alpha
        self.alpha = np.sum(0.5 / (log(UU * self.s) - UU * log(self.s))) / (self.N*self.P*self.num_subj)
        # 4. Updating the beta
        # second moment E(X^2) = (alpha+1)alpha/beta^2
        self.beta = self.P*self.num_subj*self.alpha / np.sum(U_hat*self.s)

    def sample(self, U, V=None):
        """ Generate random data given this emission model

        :param U: The prior arrangement U from arrangement model
        :param V: given the initial V. If None, then randomly generate

        :return: sampled data Y
        """
        if V is None:
            V = np.random.normal(0, 1, (self.N, self.K))
            # Make zero mean, unit length
            V = V - V.mean(axis=0)
            V = V / np.sqrt(np.sum(V ** 2, axis=0))
        else:
            this_n, this_k = V.shape
            if this_k != self.K:
                raise (NameError('Number of columns in V need to match Model.K'))

        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        signal = np.empty((num_subj, self.P))
        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            signal[s, :] = np.random.gamma(self.alpha, self.beta, (self.P,))
            Y[s, :, :] = V[:, U[s, :].astype('int')] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + np.random.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        return Y

