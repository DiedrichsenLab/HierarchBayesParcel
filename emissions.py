# Example Models
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
from numpy import log, exp, sqrt
from sample_vmf import rand_von_mises_fisher, rand_von_Mises


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
        super(MixGaussian, self).__init__(K, N, P, data, params)
        self.nparams = self.N * self.K + 1

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super(MixGaussian, self).initialize(data)
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
        self.V = YU / np.sum(UU, axis=1)
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

    def _loglikelihood(self, Y, sub=None):
        """ Compute the log likelihood given current parameters in the model

            Returns: The current log likelihood
        """
        if sub is None:
            sub = range(Y.shape[0])
        LL = np.empty((Y.shape[0], self.K, self.P))
        rss = np.empty((Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V ** 2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = np.dot(Y[i, :, :].T, self.V)
            YY = Y ** 2
            rss[i, :, :] = np.sum(YY[i, :, :], axis=0) - 2 * YV.T + uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (np.log(2 * np.pi) + np.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * rss[i, :, :]

        return LL


class MixGaussianExp(EmissionModel):
    """
    Mixture of Gaussians with signal strength (fit gamma distribution)
    for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super(MixGaussianExp, self).__init__(K, N, P, data, params)
        self.nparams = self.N * self.K + 3  # V shape is (N, K) + sigma2 + alpha + beta

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super(MixGaussianExp, self).initialize(data)
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
        self.V = theta[0]
        self.sigma2 = theta[1]
        self.alpha = theta[2]
        self.beta = theta[3]

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

        Args:
            sub: specify which subject to optimize

        Returns: the expected log likelihood for emission model, shape (nSubject * K * P)

        """
        if sub is None:
            sub = range(self.Y.shape[0])
        LL = np.empty((self.Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V ** 2, axis=0) # This is u.T V.T V u for each u
        VV = np.dot(self.V.T, self.V)
        for i in sub:
            YV = np.dot(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i)
            # First try sample from uniformed distribution
            # plt.figure()
            for k in range(self.K):
                for p in range(self.P):
                    # Here try to sampling the posterior of p(s_i|y_i, u_i) for each
                    # given y_i and u_i(k)
                    x = np.sort(np.random.uniform(0, 5, 1000))
                    loglike = - 0.5 * (1 / self.sigma2)*(-2*YV[p, k]*x + uVVu[k]*x**2) - self.beta*x
                    # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                    post = exp(loglike)/np.sum(exp(loglike))
                    self.s[i, k, p] = np.sum(x * post)
                    # plt.plot(x, post)
                # plt.show()

            self.s[i][self.s[i] < 0] = 0  # set all to non-negative
            # self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - 2 * YV.T * self.s[i, :, :] + \
            #                     self.s[i, :, :] ** 2 * uVVu.reshape((self.K, 1))
            self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, self.s[i, :, :])) + \
                                np.dot(VV, self.s[i, :, :]**2)
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (log(2 * np.pi) + log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + log(self.beta) - self.beta * self.s[i, :, :]

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. U_hat = E[u_i ^(k), s_i]
            In this emission model, the parameters need to be updated
            are V, sigma2, alpha, and beta
        Args:
            U_hat: The expected emission log likelihood

        Returns: Update all model parameters, self attributes

        """
        # SU = self.s * U_hat
        YUs = np.zeros((self.N, self.K))
        US = np.zeros((self.K, self.P))
        US2 = np.zeros((self.K, self.P))
        ERSS = np.zeros((self.num_subj, self.K, self.P))
        for i in range(self.num_subj):
            YV = np.dot(self.Y[i, :, :].T, self.V)
            YUs = YUs + np.dot(self.Y[i, :, :], (U_hat[i, :, :]*self.s[i, :, :]).T)
            US = US + U_hat[i, :, :] * self.s[i, :, :]
            US2 = US2 + U_hat[i, :, :] * (self.s[i, :, :]**2)
            # ERSS[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - 2 * YV.T * U_hat[i, :, :] * self.s[i, :, :] + \
            #                 U_hat[i, :, :] * (self.s[i, :, :]**2) * np.sum(self.V ** 2, axis=0).reshape((self.K, 1))
            ERSS[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, U_hat[i, :, :]*self.s[i, :, :])) + \
                                np.dot(self.V.T @ self.V, U_hat[i, :, :]*self.s[i, :, :]**2)

        # 1. Updating the sigma squared.
        # rss = np.sum(self.YY, axis=1).reshape(self.num_subj, -1, self.P) - 2*np.transpose(np.dot(np.transpose(self.Y, (0, 2, 1)), self.V), (0,2,1))*U_hat*self.s + \
        #       U_hat * self.s**2 * np.sum(self.V ** 2, axis=0).reshape((self.K, 1))
        self.sigma2 = np.sum(ERSS) / (self.N * self.P * self.num_subj)

        # 2. Updating the V
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.V = YUs / np.sum(US2, axis=1)

        # 3. Updating the beta (Since this is an exponential model)
        self.beta = self.P*self.num_subj / np.sum(US)

    def sample(self, U):
        """ Generate random data given this emission model and parameters

        Args:
            U: The prior arrangement U from the arrangement model
            V: Given the initial V. If None, then randomly generate

        Returns: Sampled data Y

        """
        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        signal = np.empty((num_subj, self.P))
        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            signal[s, :] = np.random.exponential(self.beta, (self.P,))
            Y[s, :, :] = self.V[:, U[s, :].astype('int')] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + np.random.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        return Y, signal

    def _loglikelihood(self, Y, signal, sub=None):
        """ Compute the log likelihood given current parameters in the model

        Returns: The current log likelihood
        """
        if sub is None:
            sub = range(Y.shape[0])
        LL = np.empty((Y.shape[0], self.K, self.P))
        rss = np.empty((Y.shape[0], self.K, self.P))
        uVVu = np.sum(self.V ** 2, axis=0)  # This is u.T V.T V u for each u
        for i in sub:
            YV = np.dot(Y[i, :, :].T, self.V)
            YY = Y**2

            rss[i, :, :] = np.sum(YY[i, :, :], axis=0) - 2*YV.T*signal[i, :] + \
                                signal[i, :]**2 * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (log(2 * np.pi) + log(self.sigma2)) - 0.5 * (1 / self.sigma2) * rss[i, :, :] \
                          + log(self.beta) - self.beta * signal[i, :]

        return LL


class MixVMF(EmissionModel):
    """ Mixture of Gaussians with isotropic noise

    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None, uniform=True):
        self.uniform = uniform
        super(MixVMF, self).__init__(K, N, P, data, params)
        self.nparams = self.N * self.K + self.K

    def initialize(self, data):
        """ Calculates the sufficient stats on the data that does not depend on U,
        and allocates memory for the sufficient stats that does.

        Args:
            data: the input data array.

        Returns: None. Store the data in emission model itself.

        """
        super(MixVMF, self).initialize(data)
        self.YY = self.Y**2
        self.rss = np.empty((self.num_subj, self.K, self.P))

    def get_params(self):
        """ Get the parameters for the vmf mixture model

        Returns: return all parameters that stores in the current emission model

        """
        np.testing.assert_array_equal(self.K, self.V.shape[1])
        return np.append(self.V.flatten('C'), self.kappa)

    def set_params(self, theta):
        """ Set the model parameters by the given input thetas
        Args:
            theta: The input parameters if any

        Returns: pass the given input params to the object

        """
        # np.testing.assert_array_equal(theta.size, self.nparams)
        # self.V = theta[0:self.N*self.K].reshape(self.N, self.K)
        # self.kappa = exp(theta[-self.K:])
        self.V = theta[0]
        self.kappa = theta[1]

    def random_params(self):
        """ In this mixture vmf model, the parameters are parcel-specific direction V_k
            and concentration value kappa_k.

        Returns: None, just passes the random parameters to the model

        """
        V = np.random.uniform(0, 1, (self.N, self.K))
        # standardise V to unit length
        # V = V - V.mean(axis=0)
        self.V = V / sqrt(np.sum(V**2, axis=0))
        if self.uniform:
            self.kappa = np.repeat(np.random.uniform(30, 100), self.K)
        else:
            self.kappa = np.random.uniform(30, 100, (self.K, ))

    def _bessel_function(self, order, kappa):
        """ The modified bessel function of the first kind of real order

        Args:
            order: the real order
            kappa: the input value

        Returns: The values of modified bessel function

        """
        # res = np.empty(kappa.shape)
        res = special.iv(order, kappa)
        return res

    def _log_bessel_function(self, order, kappa):
        """ The log of modified bessel function of the first kind of real order

        Args:
            order: the real order
            kappa: the input value

        Returns: The values of log of modified bessel function

        """
        frac = kappa / order
        square = 1 + frac**2
        root = np.sqrt(square)
        eta = root + np.log(frac) - np.log(1 + root)
        approx = - np.log(np.sqrt(2 * np.pi * order)) + order * eta - 0.25*np.log(square)

        return approx

    def Estep(self, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        Args:
            sub: specify which subject to optimize

        Returns: the expected log likelihood for emission model, shape (nSubject * K * P)

        """
        if sub is None:
            sub = range(self.Y.shape[0])
        LL = np.empty((self.Y.shape[0], self.K, self.P))
        # CnK = np.empty((self.Y.shape[0], self.K))

        # log likelihood is sum_i sum_k(u_i(k))*log C_n(kappa_k) + sum_i sum_k (u_i(k))kappa*V_k.T*Y_i
        for i in sub:
            YV = np.dot(self.Y[i, :, :].T, self.V)
            # CnK[i, :] = self.kappa**(self.N/2-1) / ((2*np.pi)**(self.N/2) * self._bessel_function(self.N/2 - 1, self.kappa))
            logCnK = (self.N / 2 - 1) * log(self.kappa) - (self.N / 2) * log(2 * np.pi) - \
                     self._log_bessel_function(self.N / 2 - 1, self.kappa)
            # the log likelihood for emission model (VMF in this case)
            LL[i, :, :] = (logCnK + self.kappa * YV).T

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are Vs (unit norm projected on
            the N-1 sphere) and kappa (concentration value).

        Args:
            U_hat: the expected log likelihood from the arrangement model

        Returns: Update all the object's parameters

        """
        YU = np.zeros((self.N, self.K))
        UU = np.zeros((self.K, self.P))
        for i in range(self.num_subj):
            YU = YU + np.dot(self.Y[i, :, :], U_hat[i, :, :].T)
            UU = UU + U_hat[i, :, :]

        # 1. Updating the V_k, which is || sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k)) ||
        self.V = YU / sqrt(np.sum(YU ** 2, axis=0))

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2), where r_bar = ||V_k||/N*Uhat
        if self.uniform:
            r_bar = sqrt(np.sum(YU ** 2, axis=0)) / np.sum(UU, axis=1)
            r_bar[r_bar > 0.95] = 0.95
            r_bar[r_bar < 0.05] = 0.05
            r_bar = np.repeat(np.mean(r_bar), self.K)
        else:
            r_bar = sqrt(np.sum(YU**2, axis=0)) / np.sum(UU, axis=1)
            r_bar[r_bar > 0.95] = 0.95
            r_bar[r_bar < 0.05] = 0.05

        self.kappa = (r_bar * self.N - r_bar**3) / (1 - r_bar**2)

    def sample(self, U):
        """ Draw data sample from this model and given parameters

        Args:
            U: The prior arrangement U from arragnment model

        Returns: The samples data form this distribution

        """
        num_subj = U.shape[0]
        Y = np.empty((num_subj, self.N, self.P))
        for s in range(num_subj):
            for p in range(self.P):
                # Draw sample from the vmf distribution given the input U
                # sample = np.random.vonmises(self.V[:, U[s, p].astype('int')], self.kappa[U[s, p].astype('int')], (self.N,))
                # Y[s, :, p] = sample / np.linalg.norm(sample)
                Y[s, :, p] = rand_von_mises_fisher(self.V[:, U[s, p].astype('int')], self.kappa[U[s, p].astype('int')])

        return Y

    def _loglikelihood(self, Y, sub=None):
        """ Compute the log likelihood given current parameters in the model

        Args:
            Y: The given data form the sample
            sub: the number or the index of the subject needs to be compute the log likelihood

        Returns: The log likelihood given by the current settings

        """
        if sub is None:
            sub = range(Y.shape[0])
        LL = np.empty((Y.shape[0], self.K, self.P))
        # CnK = np.empty((self.Y.shape[0], self.K))

        # log likelihood is sum_i sum_k(u_i(k))*log C_n(kappa_k) + sum_i sum_k (u_i(k))kappa*V_k.T*Y_i
        for i in sub:
            YV = np.dot(Y[i, :, :].T, self.V)
            # CnK[i, :] = self.kappa**(self.N/2-1) / ((2*np.pi)**(self.N/2) * self._bessel_function(self.N/2 - 1, self.kappa))
            logCnK = (self.N / 2 - 1) * log(self.kappa) - log(2 * np.pi) * (self.N / 2) - \
                     self._log_bessel_function(self.N / 2 - 1, self.kappa)
            # the log likelihood for emission model (VMF in this case)
            LL[i, :, :] = (logCnK + self.kappa * YV).T

        return LL