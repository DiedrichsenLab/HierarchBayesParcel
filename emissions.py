# Example Models
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from scipy import stats, special
from torch import log, exp, sqrt
from sample_vmf import rand_von_mises_fisher, rand_von_Mises
from model import Model


class EmissionModel(Model):
    def __init__(self, K=4, N=10, P=20, data=None):
        self.K = K  # Number of states
        self.N = N
        self.P = P
        self.nparams = 0
        if data is not None:
            self.initialize(data)

    def initialize(self, data):
        """Stores the data in emission model itself
        """
        if type(data) is np.ndarray:
            data = pt.tensor(data, dtype=pt.get_default_dtype())
        elif type(data) is pt.Tensor:
            pass
        else:
            raise ValueError("The input data must be a numpy.array or torch.tensor.")

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

    def random_params(self):
        """Sets all random parameters from a vector
        """
        pass


class MixGaussian(EmissionModel):
    """
    Mixture of Gaussians with isotropic noise
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super().__init__(K, N, P, data)
        self.random_params()
        self.set_param_list(['V', 'sigma2'])
        self.name = 'GMM'
        if params is not None:
            self.set_params(params)

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super().initialize(data)
        self.YY = self.Y**2
        self.rss = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        self.V = pt.randn(self.N, self.K)
        # standardise V to unit length
        # self.V = V / pt.sqrt(pt.sum(V**2, dim=0)) # Not clear why this should be constraint for GMM, but ok
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)), dtype=pt.get_default_dtype())

    def Estep(self, Y=None, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        specify which subject to optimize
        return: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)
        n_subj = self.Y.shape[0]
        if sub is None:
            sub = range(n_subj)

        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(self.V**2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2*YV.T + uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5*self.N*(log(pt.as_tensor(2*np.pi)) + log(self.sigma2))-0.5/self.sigma2 * self.rss[i, :, :]

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat.
            In this emission model, the parameters need to be updated
            are V and sigma2.
        """
        # SU = self.s * U_hat
        YU = pt.zeros((self.N, self.K))
        ERSS = pt.zeros(self.rss.shape)
        # JD: YOU CAN CAN USE pt.matmult to prevent looping over subjects
        for i in range(self.num_subj):
            YU = YU + pt.mm(self.Y[i, :, :], U_hat[i, :, :].T)
        UU = U_hat.sum(dim=(0,2))
        # self.V = np.linalg.solve(UU,YU.T).T
        # 1. Here we update the v_k, which is sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k))
        self.V = YU / UU

        # 2. Updating sigma2 (rss is calculated using updated V)
        uVVu = pt.sum(self.V ** 2, dim=0)
        # JD: Again, you should be able to avoid looping over subjects here entirely.
        for i in range(self.num_subj):
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            ERSS[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2*YV.T + uVVu.reshape((self.K, 1))
        ERSS = pt.sum(U_hat * ERSS)
        self.sigma2 = ERSS/(self.N*self.P*self.num_subj)

        # rss is calculated using V at (t-1) iteration
        # ERSS = pt.sum(U_hat * self.rss)
        # self.sigma2 = ERSS / (self.N * self.P * self.num_subj)

    def sample(self, U):
        """ Generate random data given this emission model
        :param U: The prior arrangement U from arragnment model
        :return: sampled data Y
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        num_subj = U.shape[0]
        Y = pt.empty((num_subj, self.N, self.P))
        for s in range(num_subj):
            Y[s, :, :] = self.V[:, U[s, :].long()]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + pt.normal(0, pt.sqrt(self.sigma2), (self.N, self.P))
        return Y


class MixGaussianExp(EmissionModel):
    """
    Mixture of Gaussians with signal strength (fit gamma distribution)
    for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, num_signal_bins=88, data=None, params=None):
        super().__init__(K, N, P, data)
        self.random_params()
        self.set_param_list(['V', 'sigma2', 'alpha', 'beta'])
        self.name = 'GMM_exp'
        if params is not None:
            self.set_params(params)
        self.num_signal_bins = num_signal_bins  # Bins for approximation of signal strength
        self.std_V = True  # Standardize mean vectors?

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super().initialize(data)
        self.YY = self.Y ** 2
        self.maxlength =pt.max(pt.sqrt(pt.sum(self.YY, dim=1)))
        self.s = pt.empty((self.num_subj, self.K, self.P))
        self.s2 = pt.empty((self.num_subj, self.K, self.P))
        self.rss = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        self.V = pt.randn(self.N, self.K)
        # standardise V to unit length
        self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)), dtype=pt.get_default_dtype())
        self.alpha = pt.tensor(1, dtype=pt.get_default_dtype())
        self.beta = pt.tensor(1, dtype=pt.get_default_dtype())

    def Estep_max(self, Y=None, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        :param Y: the real data
        :param sub: specify which subject to optimize
        :return: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)
        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            self.s[i, :, :] = (YV / uVVu - self.beta * self.sigma2).T  # Maximized g
            self.s[i][self.s[i] < 0] = 0  # Limit to 0
            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] + self.s[i, :, :] ** 2 * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (pt.log(pt.tensor(2*np.pi, dtype=pt.get_default_dtype())) + pt.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + self.alpha * pt.log(self.beta) - pt.special.gammaln(self.alpha) - self.beta * self.s[i, :, :]

        return LL

    def Estep_old(self, Y=None, signal=None, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        Args:
            sub: specify which subject to optimize
        Returns: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i). First try sample from uniformed distribution
            # plt.figure()
            if signal is None:

                for k in range(self.K):
                    for p in range(self.P):
                        # Here try to sampling the posterior of p(s_i|y_i, u_i) for each
                        # given y_i and u_i(k)
                        # x = pt.tensor(np.sort(np.random.uniform(0, 10, 200)), dtype=pt.get_default_dtype())
                        x = pt.linspace(0,self.maxlength*1.1,77)
                        loglike = - 0.5 * (1 / self.sigma2) * (-2 * YV[p, k] * x + uVVu[k] * x ** 2) - self.beta * x
                        # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                        # post = pt.exp(loglike) / pt.sum(pt.exp(loglike))
                        post = loglik2prob(loglike)
                        self.s[i, k, p] = pt.sum(x * post)
                        self.s2[i, k, p] = pt.sum(x**2 * post)
                        if pt.isnan(self.s[i, k, p]) or pt.isnan(self.s2[i, k, p]):
                            print(i, k, p)
                        # plt.plot(x, post)
                    # plt.show()

            else:
                self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
                self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] + self.s2[i, :, :] * uVVu.reshape((self.K, 1))
            # self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, self.s[i, :, :])) + \
            #                     np.dot(VV, self.s2[i, :, :])
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (pt.log(pt.tensor(2*np.pi, dtype=pt.get_default_dtype())) + pt.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + pt.log(self.beta) - self.beta * self.s[i, :, :]

        return LL

    def Estep(self, Y=None, signal=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        Args:
            sub: specify which subject to optimize
        Returns: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        n_subj= self.Y.shape[0]

        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        YV = pt.matmul(self.V.T,self.Y)
        signal_max = self.maxlength*1.2  # Make the maximum signal 1.2 times the max data magnitude
        signal_bin = signal_max / self.num_signal_bins
        x = pt.linspace(signal_bin/2,signal_max,self.num_signal_bins)
        logpi = pt.log(pt.tensor(2*np.pi, dtype=pt.get_default_dtype()))
        if signal is None:
                # This is p(y,s|u)
            loglike = - 0.5/self.sigma2 * (-2 * YV.view(n_subj,self.K,self.P,1) * x + uVVu.view(self.K,1,1) * (x ** 2)) - self.beta * x
                # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
            post = pt.softmax(loglike,dim=3)
            self.s = pt.sum(x * post,dim=3)
            self.s2 = pt.sum(x**2 * post,dim=3)

        else:
            self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
            self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

        self.rss = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s + self.s2 * uVVu.reshape((self.K, 1))
        # the log likelihood for emission model (GMM in this case)
        LL = -0.5 * self.N * (logpi + pt.log(self.sigma2)) - 0.5/ self.sigma2 * self.rss + pt.log(self.beta) - self.beta * self.s

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
        ERSS = pt.zeros((self.num_subj, self.K, self.P))
        YU = pt.matmul(self.Y, pt.transpose(U_hat * self.s,1,2))
        US = (U_hat * self.s).sum(dim=2)
        US2 = (U_hat * self.s2).sum(dim=2)

        # 1. Updating the V - standardize if requested
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.V = pt.sum(YU,dim=0) / pt.sum(US2, dim=0)
        if self.std_V:
            self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))

        # 2. Updating the sigma squared.
        YV = pt.matmul(self.V.T,self.Y)
        ERSS = pt.sum(self.YY, dim=1,keepdim=True) - 2 * YV * self.s + self.s2 * pt.sum(self.V ** 2, dim=0).view((self.K, 1))

        self.sigma2 = pt.sum(U_hat * ERSS) / (self.N * self.P * self.num_subj)

        # 3. Updating the beta (Since this is an exponential model)
        self.beta = self.P * self.num_subj / pt.sum(US)

    def sample(self, U, signal=None, return_signal=False):
        """ Generate random data given this emission model and parameters
        Args:
            U: The prior arrangement U from the arrangement model
            V: Given the initial V. If None, then randomly generate
            return_signal (bool): Return signal as well? False by default for compatibility
        Returns: Sampled data Y
        """
        num_subj = U.shape[0]
        Y = pt.empty((num_subj, self.N, self.P))
        if signal is not None:
            np.testing.assert_equal((signal.shape[0], signal.shape[1]), (num_subj, self.P),
                                    err_msg='The given signal must with a shape of (num_subj, P)')
        else:
            # Draw the signal strength for each node from an exponential distribution
            signal = pt.distributions.exponential.Exponential(self.beta).sample((num_subj, self.P))

        for s in range(num_subj):
            Y[s, :, :] = self.V[:, U[s, :].long()] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + pt.normal(0, np.sqrt(self.sigma2), (self.N, self.P))
        # Only return signal when asked: compatibility with other models
        if return_signal:
            return Y,signal
        else:
            return Y


class MixVMF(EmissionModel):
    """ Mixture of Gaussians with isotropic noise
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None, uniform_kappa=True):
        self.uniform_kappa = uniform_kappa
        super().__init__(K, N, P, data)
        self.random_params()
        self.set_param_list(['V', 'kappa'])
        self.name = 'VMF'
        if params is not None:
            self.set_params(params)

    def initialize(self, data):
        """ Calculates the sufficient stats on the data that does not depend on U,
        and allocates memory for the sufficient stats that does.
        Args:
            data: the input data array.
        Returns: None. Store the data in emission model itself.
        """
        super().initialize(data)
        self.YY = self.Y**2
        self.rss = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture vmf model, the parameters are parcel-specific direction V_k
            and concentration value kappa_k.
        Returns: None, just passes the random parameters to the model
        """
        # standardise V to unit length
        V = pt.randn(self.N, self.K)
        self.V = V / pt.sqrt(pt.sum(V ** 2, dim=0))

        # TODO: VMF doesn't work porperly for small kappa (let's say smaller than 8),
        # so right now the kappa is sampled from 0 to 50

        if self.uniform_kappa:
            self.kappa = pt.distributions.uniform.Uniform(8, 50).sample()
        else:
            self.kappa = pt.distributions.uniform.Uniform(8, 50).sample((self.K, ))

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
        LL = pt.empty((self.Y.shape[0], self.K, self.P))

        if type(self.V) is np.ndarray:
            self.V = pt.tensor(self.V, dtype=pt.get_default_dtype())
        if type(self.kappa) is np.ndarray:
            self.kappa = pt.tensor(self.kappa, dtype=pt.get_default_dtype())

        # log likelihood is sum_i sum_k(u_i(k))*log C_n(kappa_k) + sum_i sum_k (u_i(k))kappa*V_k.T*Y_i
        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # CnK[i, :] = self.kappa**(self.N/2-1) / ((2*np.pi)**(self.N/2) * self._bessel_function(self.N/2 - 1, self.kappa))
            logCnK = (self.N / 2 - 1) * pt.log(self.kappa) - (self.N / 2) * log(pt.as_tensor(2*np.pi)) - \
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
        if type(U_hat) is np.ndarray:
            U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

        YU = pt.zeros((self.N, self.K))
        UU = pt.zeros((self.K, self.P))
        for i in range(self.num_subj):
            YU = YU + pt.mm(self.Y[i, :, :], U_hat[i, :, :].T)
            UU = UU + U_hat[i, :, :]

        # 1. Updating the V_k, which is || sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k)) ||
        self.V = YU / pt.sqrt(pt.sum(YU ** 2, dim=0))

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2), where r_bar = ||V_k||/N*Uhat
        if self.uniform_kappa:
            r_bar = pt.sqrt(pt.sum(YU ** 2, dim=0)) / pt.sum(UU, dim=1)
            # r_bar[r_bar > 0.95] = 0.95
            # r_bar[r_bar < 0.05] = 0.05
            r_bar = pt.mean(r_bar)
        else:
            r_bar = pt.sqrt(pt.sum(YU**2, dim=0)) / pt.sum(UU, dim=1)
            # r_bar[r_bar > 0.95] = 0.95
            # r_bar[r_bar < 0.05] = 0.05

        self.kappa = (r_bar * self.N - r_bar**3) / (1 - r_bar**2)

    def sample(self, U):
        """ Draw data sample from this model and given parameters
        Args:
            U: The prior arrangement U from arragnment model (tensor)
        Returns: The samples data form this distribution
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        num_subj = U.shape[0]
        Y = pt.empty((num_subj, self.N, self.P))
        for s in range(num_subj):
            for p in range(self.P):
                # Draw sample from the vmf distribution given the input U
                # JD: Ideally re-write this routine to native Pytorch...
                if self.uniform_kappa:
                    Y[s, :, p] = pt.tensor(rand_von_mises_fisher(self.V[:, U[s, p]], self.kappa), dtype=pt.get_default_dtype())
                else:
                    Y[s, :, p] = pt.tensor(rand_von_mises_fisher(self.V[:, U[s, p]], self.kappa[U[s, p]]), dtype=pt.get_default_dtype())
        return Y


class MixGaussianGamma(EmissionModel):
    """
    Mixture of Gaussians with signal strength (fit gamma distribution)
    for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        super().__init__(K, N, P, data)
        self.random_params()
        self.set_param_list(['V', 'sigma2', 'alpha', 'beta'])
        self.name = 'GMM_gamma'
        if params is not None:
            self.set_params(params)

    def initialize(self, data):
        """Stores the data in emission model itself
        Calculates sufficient stats on the data that does not depend on u,
        and allocates memory for the sufficient stats that does.
        """
        super().initialize(data)
        self.YY = self.Y ** 2
        self.s = pt.empty((self.num_subj, self.K, self.P))
        self.s2 = pt.empty((self.num_subj, self.K, self.P))
        self.rss = pt.empty((self.num_subj, self.K, self.P))
        self.logs = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        V = pt.randn(self.N, self.K)
        # standardise V to unit length
        V = V - V.mean(dim=0)
        self.V = V / pt.sqrt(pt.sum(V ** 2, dim=0))
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)), dtype=pt.get_default_dtype())
        self.alpha = pt.tensor(1, dtype=pt.get_default_dtype())
        self.beta = pt.tensor(1, dtype=pt.get_default_dtype())

    def Estep(self, signal=None, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step
        Args:
            sub: specify which subject to optimize
        Returns: the expected log likelihood for emission model, shape (nSubject * K * P)
        """
        n_subj = self.Y.shape[0]
        if sub is None:
            sub = range(n_subj)

        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i). First try sample from uniformed distribution
            # plt.figure()
            if signal is None:
                for k in range(self.K):
                    for p in range(self.P):
                        # Here try to sampling the posterior of p(s_i|y_i, u_i) for each
                        # given y_i and u_i(k)
                        x = pt.tensor(np.sort(np.random.uniform(0, 10, 1000)), dtype=pt.get_default_dtype())
                        loglike = - 0.5 * (1 / self.sigma2) * (-2 * YV[p, k] * x + uVVu[k] * x ** 2) + \
                                  (self.alpha - 1) * log(x) - self.beta * x
                        # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                        post = pt.exp(loglike) / pt.sum(pt.exp(loglike))
                        self.s[i, k, p] = pt.sum(x * post)
                        self.s2[i, k, p] = pt.sum(x**2 * post)
                        self.logs[i, k, p] = pt.sum(pt.log(x) * post)
                        # plt.plot(x, post)
                    # plt.show()

                self.s[i][self.s[i] < 0] = 0  # set all to non-negative
                self.s2[i][self.s2[i] < 0] = 0  # set all to non-negative
            else:
                self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
                self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2
                self.logs = pt.log(signal.unsqueeze(1).repeat(1, self.K, 1))

            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] + \
                                self.s2[i, :, :] * uVVu.reshape((self.K, 1))
            # self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, self.s[i, :, :])) + \
            #                     np.dot(VV, self.s2[i, :, :])
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (pt.log(pt.tensor(2*np.pi, dtype=pt.get_default_dtype()))
                          + pt.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + self.alpha * pt.log(self.beta) - pt.special.gammaln(self.alpha) \
                          + (self.alpha - 1) * self.logs[i, :, :] - self.beta * self.s[i, :, :]

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
        YUs = pt.zeros((self.N, self.K))
        US = pt.zeros((self.K, self.P))
        US2 = pt.zeros((self.K, self.P))
        UlogS = pt.zeros((self.K, self.P))
        ERSS = pt.zeros((self.num_subj, self.K, self.P))

        for i in range(self.num_subj):
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            YUs = YUs + pt.mm(self.Y[i, :, :], (U_hat[i, :, :] * self.s[i, :, :]).T)
            US = US + U_hat[i, :, :] * self.s[i, :, :]
            US2 = US2 + U_hat[i, :, :] * self.s2[i, :, :]
            UlogS = UlogS + U_hat[i, :, :] * self.logs[i, :, :]
            ERSS[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * U_hat[i, :, :] * self.s[i, :, :] + \
                            U_hat[i, :, :] * self.s2[i, :, :] * pt.sum(self.V ** 2, dim=0).reshape((self.K, 1))
            # ERSS[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, U_hat[i, :, :]*self.s[i, :, :])) + \
            #                     np.dot(self.V.T @ self.V, U_hat[i, :, :]*self.s2[i, :, :])

        # 1. Updating the V
        # rss = np.sum(self.YY, axis=1).reshape(self.num_subj, -1, self.P) \
        # - 2*np.transpose(np.dot(np.transpose(self.Y, (0, 2, 1)), self.V), (0,2,1))*U_hat*self.s + \
        # U_hat * self.s**2 * np.sum(self.V ** 2, axis=0).reshape((self.K, 1))
        self.V = YUs / pt.sum(US2, dim=1)

        # 2. Updating the sigma squared.
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.sigma2 = pt.sum(ERSS) / (self.N * self.P * self.num_subj)

        # 3. Updating the alpha
        self.alpha = 0.5 / (pt.log(pt.sum(US) / (self.num_subj * self.P)) - pt.sum(UlogS) / (self.num_subj * self.P))
        # self.alpha = self.alpha / self.beta  # first moment

        # 4. Updating the beta
        self.beta = (self.P * self.num_subj * self.alpha) / pt.sum(US)
        # self.beta = (self.alpha+1)*self.alpha / self.beta**2  # second moment

    def sample(self, U, signal=None, return_signal=False):
        """ Generate random data given this emission model and parameters
        Args:
            U: The prior arrangement U from the arrangement model
            V: Given the initial V. If None, then randomly generate
            return_signal: If true, return both data and signal; Otherwise, only data
        Returns: Sampled data Y
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        num_subj = U.shape[0]
        Y = pt.empty((num_subj, self.N, self.P))

        if signal is not None:
            np.testing.assert_equal(signal.shape, (num_subj, self.P),
                                    err_msg='The given signal must with a shape of (num_subj, P)')
        else:
            signal = pt.distributions.gamma.Gamma(self.alpha, self.beta).sample((num_subj, self.P))

        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            Y[s, :, :] = self.V[:, U[s, :].long()] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + pt.normal(0, np.sqrt(self.sigma2), (self.N, self.P))

        if return_signal:
            return Y, signal
        else:
            return Y


def loglik2prob(loglik, dim=0):
    """Safe transformation and normalization of
    logliklihood

    Args:
        loglik (ndarray): Log likelihood (not normalized)
        axis (int): Number of axis (or axes), along which the probability is being standardized
    Returns:
        prob (ndarray): Probability
    """
    if dim==0:
        ml, _ = pt.max(loglik, dim=0)
        loglik = loglik - ml + 10
        prob = np.exp(loglik)
        prob = prob / pt.sum(prob, dim=0)
    else:
        a = pt.tensor(loglik.shape)
        a[dim] = 1  # Insert singleton dimension
        ml, _ = pt.max(loglik, dim=0)
        loglik = loglik - ml.reshape(a) + 10
        prob = pt.exp(loglik)
        prob = prob/pt.sum(prob, dim=1).reshape(a)
    return prob
